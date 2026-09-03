//! JSON Schema → GBNF translator.
//!
//! A fail-closed generator-oriented subset used by OpenAI
//! `response_format`, structured outputs, and family tool-call schemas.
//!
//!   - Primitive types: `string`, `number`, `integer`, `boolean`, `null`.
//!   - `object` with `properties` and `required`.
//!   - `array` with `items`.
//!   - `const` and `enum` for every JSON value kind.
//!   - Exact sibling intersection for finite values and composition.
//!   - `type` as either a single string or an array of strings (unions).
//!   - Type-agnostic schema (bare `{}`) → `value` primitive.
//!
//! Every encountered assertion is either compiled exactly or rejected with a
//! [`SchemaError`]. Assertions are never accepted and silently discarded.
//!
//! `additionalProperties: false` IS enforced (iter 75): the grammar rejects
//! any key not declared in `properties`. `additionalProperties: true` or
//! unset (the default per JSON Schema spec) allows only undeclared extra keys
//! through an exact exclusion trie. `additionalProperties: {schema}` applies
//! that schema to those extra values.
//!
//! Object key order (iter 75): grammar accepts keys in ANY order for up to
//! 12 required keys. For N_req > 12 the emitter
//! returns a `SchemaError` → HTTP 400; no sequential fallback is provided
//! because sorted-order is a semantic downgrade (Moshier & Rounds ACL 1987).
//! Previously keys were required alphabetically (iter 8 simplification, never
//! a feature). The algorithm generates O(2^N_req) permutation sub-rules for
//! N required keys and O(N^2) optional-chain sub-rules for optional keys.
//!
//! The output is a GBNF string that can be parsed by
//! `super::parser::parse(...)` and consumed by `super::sampler::GrammarRuntime`.
//! The root rule is always named `root`.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde_json::Value;

use super::regex_gbnf::{Surface, regex_to_gbnf_body, regex_to_gbnf_full_match};

const MAX_SCHEMA_DEPTH: usize = 64;
const MAX_LOCAL_REFS: usize = 1024;
const MAX_ENUM_VALUES: usize = 1024;
const MAX_LITERAL_BYTES: usize = 1024 * 1024;
const MAX_INTEGER_MAGNITUDE: i64 = 9_999_999_999_999_999;

/// Return the XGrammar/llama.cpp-compatible lexical pattern for a supported
/// JSON Schema string format. These are generator constraints: they validate
/// the same lexical shapes upstream constrained decoders use, not calendar or
/// DNS existence.
fn string_format_pattern(format: &str) -> Option<&'static str> {
    match format {
        "email" => Some(
            r#"^([a-zA-Z0-9_!#$%&'*+/=?^`{|}~-]+(\.[a-zA-Z0-9_!#$%&'*+/=?^`{|}~-]+)*|\"[^\"\r\n]*\")@[A-Za-z0-9]([-A-Za-z0-9]*[A-Za-z0-9])?(\.[A-Za-z0-9]([-A-Za-z0-9]*[A-Za-z0-9])?)*$"#,
        ),
        "date" => Some(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2]\d|3[01])$"),
        "time" => {
            Some(r"^([01]\d|2[0-3]):[0-5]\d:([0-5]\d|60)(\.\d+)?(Z|[+-]([01]\d|2[0-3]):[0-5]\d)$")
        }
        "date-time" => Some(
            r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2]\d|3[01])T([01]\d|2[0-3]):[0-5]\d:([0-5]\d|60)(\.\d+)?(Z|[+-]([01]\d|2[0-3]):[0-5]\d)$",
        ),
        "duration" => Some(
            r"^P((\d+D|\d+M(\d+D)?|\d+Y(\d+M(\d+D)?)?)(T(\d+S|\d+M(\d+S)?|\d+H(\d+M(\d+S)?)?))?|T(\d+S|\d+M(\d+S)?|\d+H(\d+M(\d+S)?)?)|\d+W)$",
        ),
        "ipv4" => Some(r"^((25[0-5]|2[0-4]\d|[0-1]?\d?\d)\.){3}(25[0-5]|2[0-4]\d|[0-1]?\d?\d)$"),
        "ipv6" => Some(
            r"^(([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))$",
        ),
        "hostname" => Some(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*$"),
        "uuid" | "uuid1" | "uuid2" | "uuid3" | "uuid4" | "uuid5" => {
            Some(r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$")
        }
        "uri" => Some(
            r#"^[a-zA-Z][a-zA-Z+\.-]*:(//(([\w\.~!$&'()*+,;=:-]|%[0-9A-Fa-f][0-9A-Fa-f])*@)?([\w\.~!$&'()*+,;=-]|%[0-9A-Fa-f][0-9A-Fa-f])*(\:\d*)?(/([\w\.~!$&'()*+,;=:@-]|%[0-9A-Fa-f][0-9A-Fa-f])*)*|/?(([\w\.~!$&'()*+,;=:@-]|%[0-9A-Fa-f][0-9A-Fa-f])+(/([\w\.~!$&'()*+,;=:@-]|%[0-9A-Fa-f][0-9A-Fa-f])*)*)?)(\?([\w\.~!$&'()*+,;=:@/\?-]|%[0-9A-Fa-f][0-9A-Fa-f])*)?(#([\w\.~!$&'()*+,;=:@/\?-]|%[0-9A-Fa-f][0-9A-Fa-f])*)?$"#,
        ),
        "uri-reference" => Some(
            r#"^(//(([\w\.~!$&'()*+,;=:-]|%[0-9A-Fa-f][0-9A-Fa-f])*@)?([\w\.~!$&'()*+,;=-]|%[0-9A-Fa-f][0-9A-Fa-f])*(\:\d*)?(/([\w\.~!$&'()*+,;=:@-]|%[0-9A-Fa-f][0-9A-Fa-f])*)*|/(([\w\.~!$&'()*+,;=:@-]|%[0-9A-Fa-f][0-9A-Fa-f])+(/([\w\.~!$&'()*+,;=:@-]|%[0-9A-Fa-f][0-9A-Fa-f])*)*)?|([\w\.~!$&'()*+,;=@-]|%[0-9A-Fa-f][0-9A-Fa-f])+(/([\w\.~!$&'()*+,;=:@-]|%[0-9A-Fa-f][0-9A-Fa-f])*)*)?(\?([\w\.~!$&'()*+,;=:@/\?-]|%[0-9A-Fa-f][0-9A-Fa-f])*)?(#([\w\.~!$&'()*+,;=:@/\?-]|%[0-9A-Fa-f][0-9A-Fa-f])*)?$"#,
        ),
        "uri-template" => Some(
            r#"^([^\s\"\\{}]|\{[+#./;?&=,!@|]?[a-zA-Z0-9_%]+([.:*][a-zA-Z0-9_%]+)*(,[a-zA-Z0-9_%]+([.:*][a-zA-Z0-9_%]+)*)*\})*$"#,
        ),
        "json-pointer" | "relative-json-pointer" => None,
        _ => None,
    }
}

fn is_supported_string_format(format: &str) -> bool {
    string_format_pattern(format).is_some()
        || matches!(format, "json-pointer" | "relative-json-pointer")
}

/// Compile a supported JSON Schema format for a particular string surface.
/// Family-native tool emitters use this same helper so no model family gets a
/// weaker interpretation than response JSON.
pub fn string_format_gbnf(format: &str, surface: Surface) -> Result<String, SchemaError> {
    match format {
        "json-pointer" => {
            let char_class = match surface {
                Surface::JsonString => r#"[^"\\/~\x00-\x1F]"#,
                Surface::QwenJsonString => r#"[^"\\</~\x00-\x1F]"#,
                Surface::QwenRawString | Surface::GemmaMarkerString => r#"[^</~]"#,
                Surface::DeepSeekRawString | Surface::RawOutput => r#"[^/~]"#,
            };
            Ok(format!(r#"( "/" ( {char_class} | "~" [01] )* )*"#))
        }
        "relative-json-pointer" => {
            let pointer = string_format_gbnf("json-pointer", surface)?;
            Ok(format!(r##"( "0" | [1-9] [0-9]* ) ( "#" | {pointer} )"##))
        }
        other => {
            let pattern = string_format_pattern(other)
                .ok_or_else(|| schema_error("/format", format!("unsupported format {other:?}")))?;
            regex_to_gbnf_full_match(pattern, surface)
                .map_err(|error| schema_error("/format", error.to_string()))
        }
    }
}

// ---------------------------------------------------------------------------
// Primitive rule library.
// ---------------------------------------------------------------------------

/// GBNF body for the `space` rule — 0+ whitespace characters. Kept identical
/// to the peer's `SPACE_RULE` so output grammars are byte-for-byte
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

// The "object" primitive body above has a subtle issue: the braces { } are
// treated as literals in the body but that's not valid GBNF. The peer's
// version is:
//   "\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? \"}\" space"
// Let me use the quoted-brace form.

/// The peer's actual primitives body uses quoted braces for object. This is
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
const _UNUSED_PRIMITIVE: fn(&str) -> Option<(&'static str, &'static str, &'static [&'static str])> =
    primitive;

// ---------------------------------------------------------------------------
// Literal escape helpers
// ---------------------------------------------------------------------------

/// Escape a string so it can be embedded as a GBNF literal between double
/// quotes.
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

#[derive(Default)]
struct JsonKeyTrie {
    terminal: bool,
    children: BTreeMap<char, JsonKeyTrie>,
}

/// Match a JSON string whose decoded value is not one of `excluded`.
///
/// Declared object keys may otherwise be consumed by an open object's
/// wildcard key rule, bypassing the declared value schema.  The trie works on
/// decoded ASCII characters and accounts for both raw and `\u00XX` spellings
/// (plus JSON's short escapes), so alternate JSON escaping cannot reopen that
/// path. Non-ASCII declared names fail closed rather than being approximated.
pub(crate) fn json_string_excluding_gbnf(
    excluded: &[String],
    char_rule: &str,
    allow_escaped_slash: bool,
    allow_raw_del: bool,
) -> Result<String, String> {
    let mut trie = JsonKeyTrie::default();
    let mut total_characters = 0usize;
    for name in excluded {
        if !name.is_ascii() {
            return Err(format!(
                "declared property {name:?} is non-ASCII; exact wildcard key exclusion is unavailable"
            ));
        }
        let characters = name.chars().count();
        total_characters = total_characters.saturating_add(characters);
        if characters > 256 || total_characters > 4096 {
            return Err(
                "declared property names exceed the exact wildcard exclusion budget (256 characters per key, 4096 total)"
                    .to_string(),
            );
        }
        let mut node = &mut trie;
        for character in name.chars() {
            node = node.children.entry(character).or_default();
        }
        node.terminal = true;
    }
    let body = json_key_trie_body(&trie, char_rule, allow_escaped_slash, allow_raw_del);
    Ok(format!(r#""\"" {body} "\"""#))
}

fn json_key_trie_body(
    node: &JsonKeyTrie,
    char_rule: &str,
    allow_slash: bool,
    allow_raw_del: bool,
) -> String {
    let mut alternatives = Vec::new();
    if !node.terminal {
        alternatives.push(empty_expression());
    }
    for (character, child) in &node.children {
        alternatives.push(format!(
            "{} {}",
            json_character_encoding(*character, allow_slash, allow_raw_del),
            json_key_trie_body(child, char_rule, allow_slash, allow_raw_del)
        ));
    }
    alternatives.push(format!(
        "{} {char_rule}*",
        json_character_except(node.children.keys().copied(), allow_slash, allow_raw_del,)
    ));
    format!("( {} )", alternatives.join(" | "))
}

fn json_character_encoding(character: char, allow_slash: bool, allow_raw_del: bool) -> String {
    let mut alternatives = Vec::new();
    if character >= ' '
        && character != '"'
        && character != '\\'
        && (allow_raw_del || character != '\u{7f}')
    {
        alternatives.push(format_literal(&character.to_string()));
    }
    let short = match character {
        '"' => Some("\\\""),
        '\\' => Some("\\\\"),
        '/' if allow_slash => Some("\\/"),
        '\u{0008}' => Some("\\b"),
        '\u{000c}' => Some("\\f"),
        '\n' => Some("\\n"),
        '\r' => Some("\\r"),
        '\t' => Some("\\t"),
        _ => None,
    };
    if let Some(short) = short {
        alternatives.push(format_literal(short));
    }
    alternatives.push(json_unicode_escape(character as u8));
    format!("( {} )", alternatives.join(" | "))
}

fn json_unicode_escape(byte: u8) -> String {
    let digits = format!("{byte:04x}");
    format!(
        "{} {}",
        format_literal("\\u"),
        digits
            .chars()
            .map(hex_exact_symbol)
            .collect::<Vec<_>>()
            .join(" ")
    )
}

fn hex_exact_symbol(symbol: char) -> String {
    if symbol.is_ascii_digit() {
        format_literal(&symbol.to_string())
    } else {
        format!("[{}{}]", symbol, symbol.to_ascii_uppercase())
    }
}

fn json_character_except(
    excluded: impl Iterator<Item = char>,
    allow_slash: bool,
    allow_raw_del: bool,
) -> String {
    let excluded: HashSet<char> = excluded.collect();
    let mut raw = if allow_raw_del {
        String::from(r#"[^"\\\x00-\x1F"#)
    } else {
        String::from(r#"[^"\\\x7F\x00-\x1F"#)
    };
    for character in &excluded {
        if *character >= ' '
            && *character != '"'
            && *character != '\\'
            && (allow_raw_del || *character != '\u{7f}')
        {
            match character {
                ']' | '-' => {
                    raw.push('\\');
                    raw.push(*character);
                }
                _ => raw.push(*character),
            }
        }
    }
    raw.push(']');

    let mut alternatives = vec![raw];
    let short_escapes = [
        ('"', '"'),
        ('\\', '\\'),
        ('/', '/'),
        ('b', '\u{0008}'),
        ('f', '\u{000c}'),
        ('n', '\n'),
        ('r', '\r'),
        ('t', '\t'),
    ];
    let mut short_class = String::new();
    for (encoded, decoded) in short_escapes {
        if (encoded != '/' || allow_slash) && !excluded.contains(&decoded) {
            if encoded == '\\' {
                short_class.push_str("\\\\");
            } else {
                short_class.push(encoded);
            }
        }
    }
    if !short_class.is_empty() {
        alternatives.push(format!(r#"[\\] [{short_class}]"#));
    }
    let excluded_hex: Vec<[u8; 4]> = excluded
        .iter()
        .map(|character| {
            let text = format!("{:04x}", *character as u8);
            text.as_bytes().try_into().expect("four hex digits")
        })
        .collect();
    alternatives.push(format!(
        "{} {} {}",
        r#"[\\]"#,
        format_literal("u"),
        hex4_except(&excluded_hex, 0)
    ));
    format!("( {} )", alternatives.join(" | "))
}

fn hex4_except(excluded: &[[u8; 4]], position: usize) -> String {
    if excluded.is_empty() {
        return format!("[0-9a-fA-F]{{{}}}", 4 - position);
    }
    let mut alternatives = Vec::new();
    let mut child_symbols = excluded
        .iter()
        .map(|value| value[position])
        .collect::<Vec<_>>();
    child_symbols.sort_unstable();
    child_symbols.dedup();
    let allowed = b"0123456789abcdef"
        .iter()
        .copied()
        .filter(|symbol| !child_symbols.contains(symbol))
        .collect::<Vec<_>>();
    if !allowed.is_empty() {
        let mut class = String::from("[");
        for symbol in allowed {
            class.push(symbol as char);
            if symbol.is_ascii_lowercase() {
                class.push((symbol as char).to_ascii_uppercase());
            }
        }
        class.push(']');
        if position + 1 < 4 {
            class.push_str(&format!(" [0-9a-fA-F]{{{}}}", 3 - position));
        }
        alternatives.push(class);
    }
    if position + 1 < 4 {
        for symbol in child_symbols {
            let descendants = excluded
                .iter()
                .copied()
                .filter(|value| value[position] == symbol)
                .collect::<Vec<_>>();
            alternatives.push(format!(
                "{} {}",
                hex_exact_symbol(symbol as char),
                hex4_except(&descendants, position + 1)
            ));
        }
    }
    format!("( {} )", alternatives.join(" | "))
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Structured error returned by `schema_to_gbnf` (→ HTTP 400).
///
/// # Variants
///
/// - `TooManyRequiredKeys` — the object's `required` array exceeds
///   `ANY_ORDER_MAX_REQUIRED` (12).  Carries the function/path name, the
///   actual count, and the cap so callers can format a precise 400 body.
///   Introduced in ADR-005 W-ζ (wave-2.7) to replace the generic struct
///   that the audit (commit 5110dc0) implied but never created.
///
/// - `Generic` — all other schema errors; preserves the pre-W-ζ
///   `{ path, message }` structure so no existing call-site is broken.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaError {
    /// Object schema has more required properties than the any-order
    /// grammar supports (> `ANY_ORDER_MAX_REQUIRED` = 12).
    TooManyRequiredKeys {
        /// Dot-path of the offending object in the schema (empty = root).
        fn_name: String,
        /// Number of required keys found.
        count: usize,
        /// The cap that was exceeded.
        max: usize,
    },
    /// Any other schema error: unsupported feature, malformed schema, etc.
    Generic {
        /// Dot-path of the offending node in the JSON Schema.
        path: String,
        /// Human-readable description of what went wrong.
        message: String,
    },
}

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchemaError::TooManyRequiredKeys {
                fn_name,
                count,
                max,
            } => write!(
                f,
                "json-schema-to-grammar error at {}: object has {} required \
                 properties; ADR-005 grammar enforcement supports at most {} \
                 required properties per object (CFG-for-permutation is \
                 provably exponential per Moshier & Rounds ACL 1987). \
                 Reduce required properties or split the schema.",
                if fn_name.is_empty() { "root" } else { fn_name },
                count,
                max,
            ),
            SchemaError::Generic { path, message } => {
                write!(f, "json-schema-to-grammar error at {}: {}", path, message)
            }
        }
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
    schema_to_gbnf_with_whitespace(schema, None)
}

/// Compile a schema with an optional full-match regex for inter-token JSON
/// whitespace. `Some("")` means no whitespace; `None` keeps the default
/// llama.cpp-compatible whitespace rule.
pub fn schema_to_gbnf_with_whitespace(
    schema: &Value,
    whitespace_pattern: Option<&str>,
) -> Result<String, SchemaError> {
    schema_to_gbnf_with_options(schema, whitespace_pattern, true)
}

/// Compile a schema with explicit object-order semantics.
///
/// The historical public helpers preserve hf2q's any-order response-format
/// behavior. Structural tags need XGrammar's distinct `any_order` switch:
/// false follows the declaration order carried by serde_json's
/// `preserve_order` map, while true retains the existing permutation grammar.
/// This deliberately remains a narrow compiler option rather than a new API
/// surface for request handling.
pub fn schema_to_gbnf_with_options(
    schema: &Value,
    whitespace_pattern: Option<&str>,
    any_order: bool,
) -> Result<String, SchemaError> {
    validate_schema_profile(schema)?;
    let space_rule = match whitespace_pattern {
        None => SPACE_RULE.to_string(),
        Some("") => r#""""#.to_string(),
        Some(pattern) => regex_to_gbnf_full_match(pattern, Surface::RawOutput)
            .map_err(|error| schema_error("/whitespace_pattern", error.to_string()))?,
    };
    let mut conv = Converter {
        rules: BTreeMap::new(),
        added_primitives: HashSet::new(),
        root_schema: schema.clone(),
        ref_rules: HashMap::new(),
        resolving_refs: HashSet::new(),
        resolved_refs: 0,
        any_order,
    };
    conv.rules.insert("space".to_string(), space_rule);
    let root_body = conv.visit(schema, "", 0)?;
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
    root_schema: Value,
    ref_rules: HashMap<String, String>,
    resolving_refs: HashSet<String>,
    resolved_refs: usize,
    /// Whether object properties may be emitted in arbitrary order.  The
    /// structural-tag surface sets this from XGrammar's `any_order`; existing
    /// callers retain the historical any-order behavior.
    any_order: bool,
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
    fn visit(&mut self, schema: &Value, path: &str, depth: usize) -> Result<String, SchemaError> {
        if depth > MAX_SCHEMA_DEPTH {
            return Err(schema_error(
                path,
                format!("schema nesting exceeds {MAX_SCHEMA_DEPTH}"),
            ));
        }
        if let Some(allowed) = schema.as_bool() {
            if allowed {
                self.add_primitive("value");
                return Ok("value".into());
            }
            return Ok(self.uninhabited_rule(path));
        }
        let obj = match schema.as_object() {
            Some(o) => o,
            None => {
                return Err(SchemaError::Generic {
                    path: path.to_string(),
                    message: "schema must be a JSON object".into(),
                });
            }
        };

        if let Some(Value::String(reference)) = obj.get("$ref") {
            return self.visit_ref(reference, path, depth);
        }

        if let Some(values) = finite_candidates(obj) {
            let mut siblings = obj.clone();
            siblings.remove("const");
            siblings.remove("enum");
            let sibling_schema = Value::Object(siblings);
            let mut narrowed_values = Vec::new();
            for value in values {
                if instance_matches_schema(&value, &sibling_schema, &self.root_schema, depth + 1)? {
                    narrowed_values.push(value);
                }
            }
            let values = narrowed_values;
            if values.is_empty() {
                return Ok(self.uninhabited_rule(path));
            }
            self.rules
                .entry("space".to_string())
                .or_insert_with(|| SPACE_RULE.to_string());
            let mut alternatives = Vec::with_capacity(values.len());
            let mut literal_bytes = 0usize;
            for value in values {
                let text = serde_json::to_string(&value).map_err(|error| {
                    schema_error(path, format!("cannot serialize finite value: {error}"))
                })?;
                literal_bytes = literal_bytes.saturating_add(text.len());
                if literal_bytes > MAX_LITERAL_BYTES {
                    return Err(schema_error(
                        path,
                        format!("literal bytes exceed {MAX_LITERAL_BYTES}"),
                    ));
                }
                alternatives.push(format_literal(&text));
            }
            return Ok(format!("({}) space", alternatives.join(" | ")));
        }

        if let Some(Value::Array(branches)) = obj.get("allOf") {
            return self.visit_all_of(obj, branches, path, depth);
        }

        for keyword in ["anyOf", "oneOf"] {
            if let Some(Value::Array(branches)) = obj.get(keyword) {
                if branches.is_empty() {
                    return Ok(self.uninhabited_rule(&format!("{path}/{keyword}")));
                }
                let mut siblings = obj.clone();
                siblings.remove(keyword);
                let sibling_schema = Value::Object(siblings);
                let narrowed = branches
                    .iter()
                    .enumerate()
                    .map(|(index, branch)| {
                        merge_schemas(
                            &sibling_schema,
                            branch,
                            &format!("{path}/{keyword}/{index}"),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if keyword == "oneOf" {
                    for left in 0..narrowed.len() {
                        for right in left + 1..narrowed.len() {
                            if !schemas_provably_disjoint(&narrowed[left], &narrowed[right]) {
                                return Err(schema_error(
                                    &format!("{path}/oneOf"),
                                    format!(
                                        "branches {left} and {right} are not provably disjoint; exact oneOf cannot be lowered"
                                    ),
                                ));
                            }
                        }
                    }
                }
                let slug = path_slug(path);
                let mut alternatives = Vec::with_capacity(narrowed.len());
                for (index, branch) in narrowed.iter().enumerate() {
                    let body =
                        self.visit(branch, &format!("{path}/{keyword}/{index}"), depth + 1)?;
                    let name = format!("{slug}-{keyword}-{index}");
                    self.rules.insert(name.clone(), body);
                    alternatives.push(name);
                }
                return Ok(alternatives.join(" | "));
            }
        }

        // `type`: the dominant dispatch.
        let inferred_type = if obj.contains_key("format")
            || obj.contains_key("pattern")
            || obj.contains_key("minLength")
            || obj.contains_key("maxLength")
        {
            Some("string")
        } else if obj.contains_key("properties")
            || obj.contains_key("required")
            || obj.contains_key("additionalProperties")
            || obj.contains_key("minProperties")
            || obj.contains_key("maxProperties")
        {
            Some("object")
        } else if obj.contains_key("items")
            || obj.contains_key("prefixItems")
            || obj.contains_key("minItems")
            || obj.contains_key("maxItems")
        {
            Some("array")
        } else {
            None
        };
        let type_val = obj.get("type");
        let type_str = match type_val {
            None if inferred_type.is_none() => {
                // Untyped — accept any JSON value.
                self.add_primitive("value");
                return Ok("value".into());
            }
            None => inferred_type.expect("checked").to_string(),
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(types)) => {
                // A type union retains every sibling assertion that applies
                // to the selected branch. Dropping them would silently widen
                // schemas such as `{type:["string","null"],pattern:...}`.
                let mut alts: Vec<String> = Vec::with_capacity(types.len());
                for (i, t) in types.iter().enumerate() {
                    let tstr = t.as_str().ok_or_else(|| SchemaError::Generic {
                        path: format!("{}/type/{}", path, i),
                        message: "type array entries must be strings".into(),
                    })?;
                    let mut stub = serde_json::Map::new();
                    stub.insert("type".into(), Value::String(tstr.into()));
                    for (key, value) in obj {
                        if key != "type" && !is_schema_annotation(key) {
                            stub.insert(key.clone(), value.clone());
                        }
                    }
                    let body = self.visit(
                        &Value::Object(stub),
                        &format!("{}/type/{}", path, i),
                        depth + 1,
                    )?;
                    alts.push(body);
                }
                return Ok(alts.join(" | "));
            }
            Some(other) => {
                return Err(SchemaError::Generic {
                    path: format!("{}/type", path),
                    message: format!("type must be a string or array of strings, got {:?}", other),
                });
            }
        };

        match type_str.as_str() {
            "string" => {
                if let Some(format) = obj.get("format").and_then(Value::as_str) {
                    let body =
                        string_format_gbnf(format, Surface::JsonString).map_err(|error| {
                            schema_error(&format!("{path}/format"), error.to_string())
                        })?;
                    return Ok(format!(r#""\"" {} "\"" space"#, body));
                }
                if let Some(Value::String(pattern)) = obj.get("pattern") {
                    let body =
                        regex_to_gbnf_body(pattern, Surface::JsonString).map_err(|error| {
                            schema_error(&format!("{path}/pattern"), error.to_string())
                        })?;
                    return Ok(format!(r#""\"" {} "\"" space"#, body));
                }
                let min = obj.get("minLength").and_then(Value::as_u64).unwrap_or(0);
                let max = obj.get("maxLength").and_then(Value::as_u64);
                if min > 0 || max.is_some() {
                    if max.is_some_and(|upper| upper < min) {
                        return Ok(self.uninhabited_rule(path));
                    }
                    self.add_primitive("char");
                    let repetition = match max {
                        Some(upper) if upper == min => format!("char{{{min}}}"),
                        Some(upper) => format!("char{{{min},{upper}}}"),
                        None => format!("char{{{min},}}"),
                    };
                    return Ok(format!(r#""\"" {} "\"" space"#, repetition));
                }
                self.add_primitive("string");
                Ok("string".into())
            }
            "number" => {
                self.add_primitive("number");
                Ok("number".into())
            }
            "integer" => {
                if ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]
                    .iter()
                    .any(|keyword| obj.contains_key(*keyword))
                {
                    return Ok(format!("{} space", integer_range_gbnf(obj)?));
                }
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
            "object" => self.visit_object(obj, path, depth),
            "array" => self.visit_array(obj, path, depth),
            other => Err(SchemaError::Generic {
                path: format!("{}/type", path),
                message: format!("unsupported type '{}'", other),
            }),
        }
    }

    fn uninhabited_rule(&mut self, path: &str) -> String {
        let name = format!("{}-uninhabited", path_slug(path));
        self.rules
            .entry(name.clone())
            .or_insert_with(|| r#"[^\U00000000-\U0010FFFF]"#.to_string());
        name
    }

    fn visit_ref(
        &mut self,
        reference: &str,
        path: &str,
        depth: usize,
    ) -> Result<String, SchemaError> {
        if !reference.starts_with('#') || reference.contains('%') {
            return Err(schema_error(
                &format!("{path}/$ref"),
                "only unescaped local JSON Pointer references are supported",
            ));
        }
        if let Some(rule) = self.ref_rules.get(reference) {
            return Ok(rule.clone());
        }
        self.resolved_refs += 1;
        if self.resolved_refs > MAX_LOCAL_REFS {
            return Err(schema_error(
                &format!("{path}/$ref"),
                format!("resolved references exceed {MAX_LOCAL_REFS}"),
            ));
        }
        let target = if reference == "#" {
            self.root_schema.clone()
        } else {
            self.root_schema
                .pointer(reference.trim_start_matches('#'))
                .cloned()
                .ok_or_else(|| {
                    schema_error(
                        &format!("{path}/$ref"),
                        format!("unresolved local reference {reference:?}"),
                    )
                })?
        };
        let rule = ref_rule_name(reference);
        self.ref_rules.insert(reference.to_string(), rule.clone());
        self.rules.insert(rule.clone(), String::new());
        self.resolving_refs.insert(reference.to_string());
        let body = self.visit(&target, reference, depth + 1)?;
        self.resolving_refs.remove(reference);
        self.rules.insert(rule.clone(), body);
        Ok(rule)
    }

    fn visit_all_of(
        &mut self,
        object: &serde_json::Map<String, Value>,
        branches: &[Value],
        path: &str,
        depth: usize,
    ) -> Result<String, SchemaError> {
        if branches.is_empty() {
            let mut base = object.clone();
            base.remove("allOf");
            return self.visit(&Value::Object(base), path, depth + 1);
        }
        let mut base = Value::Object(object.clone());
        base.as_object_mut().expect("object").remove("allOf");
        let conditional_count = branches
            .iter()
            .filter(|branch| branch.get("if").is_some())
            .count();
        if conditional_count > 0 {
            if conditional_count != branches.len() || branches.len() != 1 {
                return Err(schema_error(
                    &format!("{path}/allOf"),
                    "conditional allOf currently supports exactly one if/then/else entry",
                ));
            }
            let expanded = expand_conditional(&base, &branches[0], path)?;
            let slug = path_slug(path);
            let mut alternatives = Vec::with_capacity(expanded.len());
            for (index, branch) in expanded.iter().enumerate() {
                let body =
                    self.visit(branch, &format!("{path}/allOf/0/branch/{index}"), depth + 1)?;
                let name = format!("{slug}-conditional-{index}");
                self.rules.insert(name.clone(), body);
                alternatives.push(name);
            }
            return Ok(alternatives.join(" | "));
        }

        for (index, branch) in branches.iter().enumerate() {
            base = merge_schemas(&base, branch, &format!("{path}/allOf/{index}"))?;
        }
        self.visit(&base, path, depth + 1)
    }

    fn visit_object(
        &mut self,
        obj: &serde_json::Map<String, Value>,
        path: &str,
        depth: usize,
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

        // additionalProperties handling:
        //   - unset or true  → accept undeclared extra keys (JSON Schema
        //     default) through a wildcard whose key trie exactly excludes
        //     every declared property name.
        //   - false          → closed: grammar rejects keys not in properties.
        //     Implemented by omitting the wildcard rule from the optional
        //     chain — only declared property keys can appear, so extra keys
        //     cause the grammar stack to die.
        //   - {schema}       → apply that value grammar to undeclared keys.
        let additional_props = obj.get("additionalProperties");
        let additional_closed = matches!(additional_props, Some(Value::Bool(false)));
        let additional_value_rule = match additional_props {
            Some(Value::Object(_)) | Some(Value::Bool(true)) => Some(self.visit(
                additional_props.expect("present"),
                &format!("{path}/additionalProperties"),
                depth + 1,
            )?),
            _ => None,
        };
        let min_properties = obj
            .get("minProperties")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let max_properties = obj.get("maxProperties").and_then(Value::as_u64);

        if properties.is_empty() {
            if additional_closed {
                // additionalProperties:false + no declared properties means
                // only the empty object {} is valid.
                return if min_properties == 0 {
                    Ok(r#""{" space "}" space"#.into())
                } else {
                    Ok(self.uninhabited_rule(path))
                };
            }
            if additional_value_rule.is_some()
                || obj.contains_key("minProperties")
                || obj.contains_key("maxProperties")
            {
                let name = format!("{}-typed-extra-kv", path_slug(path));
                self.rules.insert(
                    name.clone(),
                    format!(
                        "string \":\" space {}",
                        additional_value_rule.as_deref().unwrap_or("value")
                    ),
                );
                if max_properties.is_some_and(|maximum| maximum < min_properties) {
                    return Ok(self.uninhabited_rule(path));
                }
                return Ok(format!(
                    r#""{{" space {} "}}" space"#,
                    repeated_sequence(&name, min_properties, max_properties)
                ));
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
        let declared_keys: Vec<String> = properties.keys().cloned().collect();
        let mut all_keys: Vec<&String> = properties.keys().collect();
        if self.any_order {
            all_keys.sort();
        }

        let slug = path_slug(path);

        // Map: property name → kv rule name.
        let mut kv_rule_name: HashMap<String, String> = HashMap::new();
        let mut required_keys: Vec<String> = Vec::new();
        let mut optional_keys: Vec<String> = Vec::new();

        for k in &all_keys {
            let v = &properties[*k];
            let vbody = self.visit(v, &format!("{}/properties/{}", path, k), depth + 1)?;
            // Value rule: path-slug prefix avoids collisions when two
            // different object schemas share a property name.
            let val_rule = format!("{}-{}", slug, sanitize_rule_name(k));
            self.rules.insert(val_rule.clone(), vbody);

            // kv rule: literal key + ":" + space + value-rule.
            let key_json = serde_json::to_string(k).map_err(|error| {
                schema_error(
                    &format!("{path}/properties/{k}"),
                    format!("cannot serialize property name: {error}"),
                )
            })?;
            let quoted_key = format_literal(&key_json);
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
            return Err(SchemaError::Generic {
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
        // practical for small N_req but intractable at N_req > ~16.
        // Threshold = 12 keeps the worst-case bounded at 4096 subset rules
        // and covers the nine-key r2c ReviewLens contract.
        //
        // For N_req > threshold a hard SchemaError (→ HTTP 400) is returned.
        // A sequential-sorted fallback would silently change semantics from
        // "any-position" to "fixed-sorted" — a semantic downgrade prohibited by
        // the no-shortcuts mantra.  Production engines (the peer, llguidance,
        // xgrammar, outlines-core) all enforce declaration order for the same
        // reason: Moshier & Rounds ACL 1987 prove CFG-for-permutations is
        // exponential; Barton 1985 proves ID/LP recognition is NP-complete.
        const ANY_ORDER_MAX_REQUIRED: usize = 12;

        // Build extra-kv wildcard rule (shared across all states if allowed).
        if !additional_closed {
            let declared_keys = properties.keys().cloned().collect::<Vec<_>>();
            let extra_key = json_string_excluding_gbnf(&declared_keys, "char", false, false)
                .map_err(|message| schema_error(path, message))?;
            let extra_key_name = format!("{}-extra-key", slug);
            self.rules.insert(extra_key_name.clone(), extra_key);
            let extra_kv_name = format!("{}-extra-kv", slug);
            self.rules.entry(extra_kv_name).or_insert_with(|| {
                format!(
                    "{extra_key_name} \":\" space {}",
                    additional_value_rule.as_deref().unwrap_or("value")
                )
            });
        }

        if !self.any_order {
            if obj.contains_key("minProperties") || obj.contains_key("maxProperties") {
                return Err(schema_error(
                    path,
                    "minProperties/maxProperties with ordered structural-tag objects is not yet representable",
                ));
            }
            let inner = self.build_ordered_object_inner(
                &slug,
                &declared_keys,
                &required_list,
                &kv_rule_name,
                !additional_closed,
            );
            return Ok(format!(r#""{{" space {} "}}" space"#, inner));
        }

        if obj.contains_key("minProperties") || obj.contains_key("maxProperties") {
            if required_keys.len() > ANY_ORDER_MAX_REQUIRED {
                return Err(SchemaError::TooManyRequiredKeys {
                    fn_name: path.to_string(),
                    count: required_keys.len(),
                    max: ANY_ORDER_MAX_REQUIRED,
                });
            }
            let required_count = required_keys.len() as u64;
            let declared_count = n_total as u64;
            let effective_min = min_properties.max(required_count);
            let effective_max = if additional_closed {
                Some(max_properties.unwrap_or(declared_count).min(declared_count))
            } else {
                max_properties
            };
            if effective_max.is_some_and(|maximum| maximum < effective_min)
                || (additional_closed && effective_min > declared_count)
            {
                return Ok(self.uninhabited_rule(path));
            }
            if effective_min == 0 && effective_max == Some(0) {
                return Ok(r#""{" space "}" space"#.into());
            }
            let req_full = if required_keys.is_empty() {
                0
            } else {
                u32::MAX >> (32 - required_keys.len())
            };
            let opt_full = if optional_keys.is_empty() {
                0
            } else {
                u32::MAX >> (32 - optional_keys.len())
            };
            let counted = self.build_counted_object_inner(
                &slug,
                req_full,
                opt_full,
                0,
                effective_min,
                effective_max,
                &required_keys,
                &optional_keys,
                &kv_rule_name,
                !additional_closed,
                path,
            )?;
            let inner = if effective_min == 0 {
                format!("( {counted} )?")
            } else {
                counted
            };
            return Ok(format!(r#""{{" space {} "}}" space"#, inner));
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
            // Too many required keys: the bitmask any-position grammar grows as
            // O(2^N_req), which is provably exponential for permutations of N
            // keys (Moshier & Rounds ACL 1987).  Sequential-order fallback would
            // silently change semantics from "any-position" to "fixed-sorted",
            // which is a semantic downgrade.
            //
            // Hard error.  Operator must reduce required parameters or split the
            // function.  The threshold is ANY_ORDER_MAX_REQUIRED = 12 (4096
            // subset rules worst-case).  Propagates as HTTP 400.
            return Err(SchemaError::TooManyRequiredKeys {
                fn_name: path.to_string(),
                count: required_keys.len(),
                max: ANY_ORDER_MAX_REQUIRED,
            });
        };

        Ok(format!(r#""{{" space {} "}}" space"#, inner))
    }

    /// Build the declared-order object grammar used by XGrammar
    /// `any_order=false`.  A state carries whether a previous item was
    /// emitted, so omitted optional properties never leave a stray comma.
    fn build_ordered_object_inner(
        &mut self,
        slug: &str,
        keys: &[String],
        required: &HashSet<String>,
        kv_rule_name: &HashMap<String, String>,
        allow_extra_kv: bool,
    ) -> String {
        self.build_ordered_object_state(
            slug,
            keys,
            required,
            kv_rule_name,
            allow_extra_kv,
            0,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_ordered_object_state(
        &mut self,
        slug: &str,
        keys: &[String],
        required: &HashSet<String>,
        kv_rule_name: &HashMap<String, String>,
        allow_extra_kv: bool,
        index: usize,
        emitted: bool,
    ) -> String {
        let state = format!("{slug}-ordered-{index}-{}", u8::from(emitted));
        if self.rules.contains_key(&state) {
            return state;
        }
        self.rules.insert(state.clone(), String::new());
        let body = if index == keys.len() {
            if !allow_extra_kv {
                r#""""#.to_owned()
            } else {
                let extra = format!("{slug}-extra-kv");
                if emitted {
                    format!(r#"( "," space {extra} )*"#)
                } else {
                    format!(r#"( {extra} ("," space {extra})* )?"#)
                }
            }
        } else {
            let key = &keys[index];
            let kv = kv_rule_name
                .get(key)
                .expect("declared key must have a key/value rule");
            let next = self.build_ordered_object_state(
                slug,
                keys,
                required,
                kv_rule_name,
                allow_extra_kv,
                index + 1,
                true,
            );
            let emit = if emitted {
                format!(r#""," space {kv} {next}"#)
            } else {
                format!("{kv} {next}")
            };
            if required.contains(key) {
                emit
            } else {
                let skip = self.build_ordered_object_state(
                    slug,
                    keys,
                    required,
                    kv_rule_name,
                    allow_extra_kv,
                    index + 1,
                    emitted,
                );
                format!("( {emit} | {skip} )")
            }
        };
        self.rules.insert(state.clone(), body);
        state
    }

    #[allow(clippy::too_many_arguments)]
    fn build_counted_object_inner(
        &mut self,
        slug: &str,
        req_remaining: u32,
        opt_remaining: u32,
        emitted: u64,
        minimum: u64,
        maximum: Option<u64>,
        required_keys: &[String],
        optional_keys: &[String],
        kv_rule_name: &HashMap<String, String>,
        allow_extra_kv: bool,
        path: &str,
    ) -> Result<String, SchemaError> {
        // With no upper bound, all counts at or above the minimum are
        // equivalent. Saturation gives the extra-property branch a finite,
        // right-recursive state.
        let count_state = maximum.map_or(emitted.min(minimum), |_| emitted);
        let rule_name =
            format!("{slug}-count-r{req_remaining:08x}-o{opt_remaining:08x}-n{count_state}");
        if self.rules.contains_key(&rule_name) {
            return Ok(rule_name);
        }
        if self.rules.len() >= 8192 {
            return Err(schema_error(
                path,
                "object count grammar exceeds 8192-rule budget",
            ));
        }
        self.rules.insert(rule_name.clone(), String::new());

        let may_continue = |next_count: u64| maximum.is_none_or(|limit| next_count < limit);
        let may_close = |next_req: u32, next_count: u64| {
            next_req == 0
                && next_count >= minimum
                && maximum.is_none_or(|limit| next_count <= limit)
        };
        let mut alternatives = Vec::new();

        for (index, key) in required_keys.iter().enumerate() {
            if req_remaining & (1u32 << index) == 0 {
                continue;
            }
            let next_req = req_remaining & !(1u32 << index);
            let next_count = emitted + 1;
            let kv = &kv_rule_name[key];
            if may_close(next_req, next_count) {
                alternatives.push(kv.clone());
            }
            if may_continue(next_count) {
                let next = self.build_counted_object_inner(
                    slug,
                    next_req,
                    opt_remaining,
                    next_count,
                    minimum,
                    maximum,
                    required_keys,
                    optional_keys,
                    kv_rule_name,
                    allow_extra_kv,
                    path,
                )?;
                alternatives.push(format!("{kv} \",\" space {next}"));
            }
        }

        for (index, key) in optional_keys.iter().enumerate() {
            if opt_remaining & (1u32 << index) == 0 {
                continue;
            }
            let next_opt = opt_remaining & !(1u32 << index);
            let next_count = emitted + 1;
            let kv = &kv_rule_name[key];
            if may_close(req_remaining, next_count) {
                alternatives.push(kv.clone());
            }
            if may_continue(next_count) {
                let next = self.build_counted_object_inner(
                    slug,
                    req_remaining,
                    next_opt,
                    next_count,
                    minimum,
                    maximum,
                    required_keys,
                    optional_keys,
                    kv_rule_name,
                    allow_extra_kv,
                    path,
                )?;
                alternatives.push(format!("{kv} \",\" space {next}"));
            }
        }

        if allow_extra_kv {
            let next_count = emitted + 1;
            let kv = format!("{slug}-extra-kv");
            if may_close(req_remaining, next_count) {
                alternatives.push(kv.clone());
            }
            if may_continue(next_count) {
                let next = self.build_counted_object_inner(
                    slug,
                    req_remaining,
                    opt_remaining,
                    next_count,
                    minimum,
                    maximum,
                    required_keys,
                    optional_keys,
                    kv_rule_name,
                    allow_extra_kv,
                    path,
                )?;
                alternatives.push(format!("{kv} \",\" space {next}"));
            }
        }

        let body = if alternatives.is_empty() {
            r#"[^\U00000000-\U0010FFFF]"#.to_string()
        } else {
            alternatives.join(" | ")
        };
        self.rules.insert(rule_name.clone(), body);
        Ok(rule_name)
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
        let rule_name = format!("{}-up-r{:08x}-o{:08x}", slug, req_remaining, opt_remaining);

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
        for (i, (kv, is_wildcard)) in entries.iter().enumerate() {
            // For non-wildcard entries: remove this entry from remaining so
            // declared optional keys appear at most once (no duplicate keys).
            // For wildcard entries (extra-kv, `is_wildcard == true`): keep
            // the wildcard in remaining so the emitted rule is self-referential
            // and accepts multiple extra keys (Kleene-star semantics via GBNF
            // optional recursion).
            let keep_self = *is_wildcard; // wildcard stays in remaining; non-wildcard is removed
            let remaining: Vec<(String, bool)> = entries
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i || keep_self)
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

    fn visit_array(
        &mut self,
        obj: &serde_json::Map<String, Value>,
        path: &str,
        depth: usize,
    ) -> Result<String, SchemaError> {
        self.rules
            .entry("space".to_string())
            .or_insert_with(|| SPACE_RULE.to_string());
        let min = obj.get("minItems").and_then(Value::as_u64).unwrap_or(0);
        let max = obj.get("maxItems").and_then(Value::as_u64);
        if max.is_some_and(|upper| upper < min) {
            return Ok(self.uninhabited_rule(path));
        }

        if let Some(Value::Array(prefix)) = obj.get("prefixItems") {
            return self.visit_prefix_array(obj, prefix, path, depth, min, max);
        }

        let item_schema = obj.get("items").unwrap_or(&Value::Bool(true));
        let item_rule = self.visit(item_schema, &format!("{path}/items"), depth + 1)?;
        if item_rule.contains("-uninhabited") {
            return if min == 0 {
                Ok(r#""[" space "]" space"#.to_string())
            } else {
                Ok(self.uninhabited_rule(path))
            };
        }
        let body = repeated_sequence(&item_rule, min, max);
        Ok(format!(r#""[" space {} "]" space"#, body))
    }

    fn visit_prefix_array(
        &mut self,
        obj: &serde_json::Map<String, Value>,
        prefix: &[Value],
        path: &str,
        depth: usize,
        min: u64,
        max: Option<u64>,
    ) -> Result<String, SchemaError> {
        if prefix.len() > 32 {
            return Err(schema_error(
                &format!("{path}/prefixItems"),
                format!("{} entries exceed 32", prefix.len()),
            ));
        }
        let mut prefix_rules = Vec::with_capacity(prefix.len());
        for (index, schema) in prefix.iter().enumerate() {
            prefix_rules.push(self.visit(
                schema,
                &format!("{path}/prefixItems/{index}"),
                depth + 1,
            )?);
        }
        let tail_schema = obj.get("items").unwrap_or(&Value::Bool(true));
        let tail_rule = self.visit(tail_schema, &format!("{path}/items"), depth + 1)?;
        let tail_allowed = !tail_rule.contains("-uninhabited");
        let prefix_len = prefix_rules.len() as u64;
        let maximum = max.unwrap_or(u64::MAX);
        let mut alternatives = Vec::new();

        let short_end = maximum.min(prefix_len.saturating_sub(1));
        if min <= short_end {
            for length in min..=short_end {
                alternatives.push(join_array_items(&prefix_rules[..length as usize]));
            }
        }

        if maximum >= prefix_len && min <= maximum {
            if prefix_len == 0
                || !prefix_rules
                    .iter()
                    .any(|rule| rule.contains("-uninhabited"))
            {
                let fixed = join_array_items(&prefix_rules);
                let min_tail = min.saturating_sub(prefix_len);
                let max_tail = max.map(|upper| upper.saturating_sub(prefix_len));
                if tail_allowed {
                    let tail = repeated_sequence(&tail_rule, min_tail, max_tail);
                    let combined = match (fixed.is_empty(), tail.is_empty()) {
                        (true, _) => tail,
                        (_, true) => fixed,
                        _ if min_tail == 0 => format!(
                            "{} ( \",\" space {} )?",
                            fixed,
                            repeated_sequence(&tail_rule, 1, max_tail)
                        ),
                        _ => format!("{} \",\" space {}", fixed, tail),
                    };
                    alternatives.push(combined);
                } else if min_tail == 0 {
                    alternatives.push(fixed);
                }
            }
        }

        if alternatives.is_empty() {
            return Ok(self.uninhabited_rule(path));
        }
        alternatives.sort();
        alternatives.dedup();
        Ok(format!(
            r#""[" space ( {} ) "]" space"#,
            alternatives.join(" | ")
        ))
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

fn schema_error(path: &str, message: impl Into<String>) -> SchemaError {
    SchemaError::Generic {
        path: if path.is_empty() { "/" } else { path }.to_string(),
        message: message.into(),
    }
}

fn ref_rule_name(reference: &str) -> String {
    // Stable FNV-1a suffix prevents two JSON Pointers that sanitize to the
    // same GBNF identifier from aliasing one another.
    let mut hash = 0xcbf29ce484222325u64;
    for byte in reference.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("ref-{}-{hash:016x}", sanitize_rule_name(reference))
}

fn empty_expression() -> String {
    "\"\"".to_string()
}

fn repeated_sequence(item: &str, min: u64, max: Option<u64>) -> String {
    if max == Some(0) {
        return empty_expression();
    }
    let comma_item = format!("( \",\" space {item} )");
    if min == 0 {
        return match max {
            None => format!("( {item} {comma_item}* )?"),
            Some(upper) => format!("( {item} {comma_item}{{0,{}}} )?", upper - 1),
        };
    }
    let required_tail = min - 1;
    let suffix = match max {
        None if required_tail == 0 => format!(" {comma_item}*"),
        None => format!(" {comma_item}{{{required_tail}}} {comma_item}*"),
        Some(upper) if upper == min && required_tail == 0 => String::new(),
        Some(upper) if upper == min => format!(" {comma_item}{{{required_tail}}}"),
        Some(upper) if required_tail == 0 => {
            format!(" {comma_item}{{0,{}}}", upper - min)
        }
        Some(upper) => format!(
            " {comma_item}{{{required_tail}}} {comma_item}{{0,{}}}",
            upper - min
        ),
    };
    format!("{item}{suffix}")
}

fn decimal_digit_range(low: u8, high: u8) -> String {
    debug_assert!(low <= high && high <= 9);
    if low == high {
        format!("[{}]", char::from(b'0' + low))
    } else {
        format!("[{}-{}]", char::from(b'0' + low), char::from(b'0' + high))
    }
}

fn fixed_width_decimal_range(low: &str, high: &str) -> String {
    debug_assert_eq!(low.len(), high.len());
    debug_assert!(low <= high);
    if low == high {
        return format_literal(low);
    }

    let prefix_len = low
        .bytes()
        .zip(high.bytes())
        .take_while(|(left, right)| left == right)
        .count();
    let prefix = &low[..prefix_len];
    let low_tail = &low[prefix_len..];
    let high_tail = &high[prefix_len..];
    let low_digit = low_tail.as_bytes()[0] - b'0';
    let high_digit = high_tail.as_bytes()[0] - b'0';
    let suffix_len = low_tail.len() - 1;
    let mut alternatives = Vec::new();

    if suffix_len == 0 {
        alternatives.push(decimal_digit_range(low_digit, high_digit));
    } else {
        let low_suffix = &low_tail[1..];
        let high_suffix = &high_tail[1..];
        let zero_suffix = "0".repeat(suffix_len);
        let nine_suffix = "9".repeat(suffix_len);
        alternatives.push(format!(
            "{} {}",
            decimal_digit_range(low_digit, low_digit),
            fixed_width_decimal_range(low_suffix, &nine_suffix)
        ));
        if low_digit + 1 < high_digit {
            alternatives.push(format!(
                "{} [0-9]{{{suffix_len}}}",
                decimal_digit_range(low_digit + 1, high_digit - 1)
            ));
        }
        alternatives.push(format!(
            "{} {}",
            decimal_digit_range(high_digit, high_digit),
            fixed_width_decimal_range(&zero_suffix, high_suffix)
        ));
    }

    let body = if alternatives.len() == 1 {
        alternatives.remove(0)
    } else {
        format!("( {} )", alternatives.join(" | "))
    };
    if prefix.is_empty() {
        body
    } else {
        format!("{} {body}", format_literal(prefix))
    }
}

fn nonnegative_integer_range(low: u64, high: u64) -> String {
    debug_assert!(low <= high);
    let low_text = low.to_string();
    let high_text = high.to_string();
    let mut alternatives = Vec::new();
    for width in low_text.len()..=high_text.len() {
        let lower = if width == low_text.len() {
            low_text.clone()
        } else if width == 1 {
            "0".to_string()
        } else {
            format!("1{}", "0".repeat(width - 1))
        };
        let upper = if width == high_text.len() {
            high_text.clone()
        } else {
            "9".repeat(width)
        };
        if lower <= upper {
            alternatives.push(fixed_width_decimal_range(&lower, &upper));
        }
    }
    if alternatives.len() == 1 {
        alternatives.remove(0)
    } else {
        format!("( {} )", alternatives.join(" | "))
    }
}

fn integer_bound(
    object: &serde_json::Map<String, Value>,
    inclusive: &str,
    exclusive: &str,
    lower: bool,
) -> Result<Option<i64>, SchemaError> {
    let inclusive = object.get(inclusive).map(|value| {
        value
            .as_i64()
            .ok_or_else(|| schema_error(&format!("/{inclusive}"), "integer bound must be an i64"))
    });
    let exclusive = object.get(exclusive).map(|value| {
        let value = value.as_i64().ok_or_else(|| {
            schema_error(
                &format!("/{exclusive}"),
                "exclusive integer bound must be an i64",
            )
        })?;
        if lower {
            value.checked_add(1).ok_or_else(|| {
                schema_error(
                    &format!("/{exclusive}"),
                    "exclusive lower bound overflows i64",
                )
            })
        } else {
            value.checked_sub(1).ok_or_else(|| {
                schema_error(
                    &format!("/{exclusive}"),
                    "exclusive upper bound overflows i64",
                )
            })
        }
    });
    match (inclusive.transpose()?, exclusive.transpose()?) {
        (Some(left), Some(right)) if lower => Ok(Some(left.max(right))),
        (Some(left), Some(right)) => Ok(Some(left.min(right))),
        (Some(value), None) | (None, Some(value)) => Ok(Some(value)),
        (None, None) => Ok(None),
    }
}

/// Exact bounded-integer GBNF body shared by response JSON and all native
/// tool-call wire emitters. The lexical domain intentionally matches hf2q's
/// existing 16-digit JSON integer primitive and llama.cpp's bounded emitter.
pub fn integer_range_gbnf(object: &serde_json::Map<String, Value>) -> Result<String, SchemaError> {
    let requested_min = integer_bound(object, "minimum", "exclusiveMinimum", true)?;
    let requested_max = integer_bound(object, "maximum", "exclusiveMaximum", false)?;
    let minimum = requested_min
        .unwrap_or(-MAX_INTEGER_MAGNITUDE)
        .max(-MAX_INTEGER_MAGNITUDE);
    let maximum = requested_max
        .unwrap_or(MAX_INTEGER_MAGNITUDE)
        .min(MAX_INTEGER_MAGNITUDE);
    if minimum > maximum {
        return Ok(r#"[^\U00000000-\U0010FFFF]"#.to_string());
    }

    let mut alternatives = Vec::new();
    if minimum < 0 {
        let negative_max = maximum.min(-1);
        let magnitude_low = negative_max.unsigned_abs();
        let magnitude_high = minimum.unsigned_abs();
        alternatives.push(format!(
            r#""-" ( {} )"#,
            nonnegative_integer_range(magnitude_low, magnitude_high)
        ));
    }
    if maximum >= 0 {
        alternatives.push(nonnegative_integer_range(
            minimum.max(0) as u64,
            maximum as u64,
        ));
    }
    Ok(if alternatives.len() == 1 {
        alternatives.remove(0)
    } else {
        format!("( {} )", alternatives.join(" | "))
    })
}

fn join_array_items(items: &[String]) -> String {
    if items.is_empty() {
        return empty_expression();
    }
    items.join(" \",\" space ")
}

fn is_schema_annotation(keyword: &str) -> bool {
    matches!(
        keyword,
        "$schema"
            | "$id"
            | "$anchor"
            | "$comment"
            | "title"
            | "description"
            | "default"
            | "examples"
            | "deprecated"
            | "readOnly"
            | "writeOnly"
            | "$defs"
            | "definitions"
    )
}

/// Validate the exact assertion profile accepted by [`schema_to_gbnf`].
/// This function is also the preflight entry point for family tool emitters:
/// a lowerer MUST NOT receive an assertion that it could silently weaken.
pub fn validate_schema_profile(schema: &Value) -> Result<(), SchemaError> {
    validate_schema_node(schema, "", 0)
}

/// Resolve local references and lower schema composition into the common
/// subset consumed by model-family wire emitters. Multiple returned schemas
/// are disjoint root variants (currently produced by finite conditionals).
pub fn normalize_schema_variants(schema: &Value) -> Result<Vec<Value>, SchemaError> {
    validate_schema_profile(schema)?;
    let mut active = HashSet::new();
    let mut references = 0usize;
    let normalized = normalize_schema_node(schema, schema, "", 0, &mut active, &mut references)?;
    if let Some(branches) = normalized
        .as_object()
        .and_then(|object| object.get("oneOf"))
        .and_then(Value::as_array)
    {
        if branches.iter().enumerate().all(|(left, branch)| {
            branches
                .iter()
                .skip(left + 1)
                .all(|right| schemas_provably_disjoint(branch, right))
        }) {
            return Ok(branches.clone());
        }
    }
    Ok(vec![normalized])
}

fn normalize_schema_node(
    schema: &Value,
    root: &Value,
    path: &str,
    depth: usize,
    active: &mut HashSet<String>,
    references: &mut usize,
) -> Result<Value, SchemaError> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(schema_error(
            path,
            format!("schema nesting exceeds {MAX_SCHEMA_DEPTH}"),
        ));
    }
    if schema.is_boolean() {
        return Ok(schema.clone());
    }
    let object = schema
        .as_object()
        .ok_or_else(|| schema_error(path, "schema must be an object or boolean"))?;
    if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
        *references += 1;
        if *references > MAX_LOCAL_REFS {
            return Err(schema_error(
                path,
                format!("resolved references exceed {MAX_LOCAL_REFS}"),
            ));
        }
        if !active.insert(reference.to_string()) {
            return Err(schema_error(
                &format!("{path}/$ref"),
                "recursive references are supported by response grammars but cannot be inlined into a family tool wire grammar",
            ));
        }
        let target = if reference == "#" {
            root
        } else {
            root.pointer(reference.trim_start_matches('#'))
                .ok_or_else(|| {
                    schema_error(
                        &format!("{path}/$ref"),
                        format!("unresolved local reference {reference:?}"),
                    )
                })?
        };
        let result = normalize_schema_node(target, root, reference, depth + 1, active, references);
        active.remove(reference);
        return result;
    }

    let mut normalized = object.clone();
    if !normalized.contains_key("type") {
        let inferred = if normalized.contains_key("format")
            || normalized.contains_key("pattern")
            || normalized.contains_key("minLength")
            || normalized.contains_key("maxLength")
        {
            Some("string")
        } else if normalized.contains_key("properties")
            || normalized.contains_key("required")
            || normalized.contains_key("additionalProperties")
            || normalized.contains_key("minProperties")
            || normalized.contains_key("maxProperties")
        {
            Some("object")
        } else if normalized.contains_key("items")
            || normalized.contains_key("prefixItems")
            || normalized.contains_key("minItems")
            || normalized.contains_key("maxItems")
        {
            Some("array")
        } else {
            None
        };
        if let Some(kind) = inferred {
            normalized.insert("type".into(), Value::String(kind.into()));
        }
    }
    normalized.remove("$defs");
    normalized.remove("definitions");
    if let Some(value) = normalized.remove("const") {
        normalized.insert("enum".into(), Value::Array(vec![value]));
    }

    if let Some(Value::Array(all_of)) = normalized.remove("allOf") {
        let base = Value::Object(normalized);
        if all_of.len() == 1 && all_of[0].get("if").is_some() {
            let branches = expand_conditional(&base, &all_of[0], path)?;
            let mut output = Vec::with_capacity(branches.len());
            for (index, branch) in branches.iter().enumerate() {
                output.push(normalize_schema_node(
                    branch,
                    root,
                    &format!("{path}/allOf/0/branch/{index}"),
                    depth + 1,
                    active,
                    references,
                )?);
            }
            return Ok(serde_json::json!({"oneOf": output}));
        }
        let mut merged = base;
        for (index, branch) in all_of.iter().enumerate() {
            merged = merge_schemas(&merged, branch, &format!("{path}/allOf/{index}"))?;
        }
        return normalize_schema_node(&merged, root, path, depth + 1, active, references);
    }

    if let Some(Value::Array(kinds)) = normalized.get("type") {
        let mut branches = Vec::with_capacity(kinds.len());
        for (index, kind) in kinds.iter().enumerate() {
            let kind = kind.as_str().expect("validated type union");
            let mut branch = serde_json::Map::new();
            for (keyword, value) in &normalized {
                if keyword == "type" {
                    branch.insert(keyword.clone(), Value::String(kind.to_string()));
                } else if is_schema_annotation(keyword) || keyword_applies_to_type(keyword, kind) {
                    branch.insert(keyword.clone(), value.clone());
                }
            }
            branches.push(normalize_schema_node(
                &Value::Object(branch),
                root,
                &format!("{path}/type/{index}"),
                depth + 1,
                active,
                references,
            )?);
        }
        return Ok(serde_json::json!({"anyOf": branches}));
    }

    for container in ["properties"] {
        if let Some(values) = normalized.get_mut(container).and_then(Value::as_object_mut) {
            for (name, child) in values.iter_mut() {
                *child = normalize_schema_node(
                    child,
                    root,
                    &format!("{path}/{container}/{name}"),
                    depth + 1,
                    active,
                    references,
                )?;
            }
        }
    }
    for keyword in ["items", "additionalProperties"] {
        if let Some(child) = normalized.get_mut(keyword) {
            *child = normalize_schema_node(
                child,
                root,
                &format!("{path}/{keyword}"),
                depth + 1,
                active,
                references,
            )?;
        }
    }
    if let Some(items) = normalized
        .get_mut("prefixItems")
        .and_then(Value::as_array_mut)
    {
        for (index, child) in items.iter_mut().enumerate() {
            *child = normalize_schema_node(
                child,
                root,
                &format!("{path}/prefixItems/{index}"),
                depth + 1,
                active,
                references,
            )?;
        }
    }
    for keyword in ["anyOf", "oneOf"] {
        if let Some(branches) = normalized.get_mut(keyword).and_then(Value::as_array_mut) {
            for (index, branch) in branches.iter_mut().enumerate() {
                *branch = normalize_schema_node(
                    branch,
                    root,
                    &format!("{path}/{keyword}/{index}"),
                    depth + 1,
                    active,
                    references,
                )?;
            }
        }
    }

    if let Some(values) = finite_candidates(&normalized) {
        let mut siblings = normalized.clone();
        siblings.remove("const");
        siblings.remove("enum");
        let sibling_schema = Value::Object(siblings);
        let mut narrowed_values = Vec::new();
        for value in values {
            if instance_matches_schema(&value, &sibling_schema, root, depth + 1)? {
                narrowed_values.push(value);
            }
        }
        let values = narrowed_values;
        return if values.is_empty() {
            Ok(Value::Bool(false))
        } else {
            Ok(serde_json::json!({"enum": values}))
        };
    }

    for keyword in ["anyOf", "oneOf"] {
        if normalized.contains_key(keyword) {
            let branches = normalized
                .remove(keyword)
                .and_then(|value| value.as_array().cloned())
                .expect("validated composition array");
            let base = Value::Object(normalized);
            let mut narrowed = Vec::with_capacity(branches.len());
            for (index, branch) in branches.iter().enumerate() {
                let merged = merge_schemas(&base, branch, &format!("{path}/{keyword}/{index}"))?;
                narrowed.push(normalize_schema_node(
                    &merged,
                    root,
                    &format!("{path}/{keyword}/{index}"),
                    depth + 1,
                    active,
                    references,
                )?);
            }
            if keyword == "oneOf" {
                for left in 0..narrowed.len() {
                    for right in left + 1..narrowed.len() {
                        if !schemas_provably_disjoint(&narrowed[left], &narrowed[right]) {
                            return Err(schema_error(
                                &format!("{path}/oneOf"),
                                format!(
                                    "branches {left} and {right} are not provably disjoint; exact oneOf cannot be lowered"
                                ),
                            ));
                        }
                    }
                }
            }
            let mut union = serde_json::Map::new();
            union.insert(keyword.to_string(), Value::Array(narrowed));
            return Ok(Value::Object(union));
        }
    }
    Ok(Value::Object(normalized))
}

fn keyword_applies_to_type(keyword: &str, kind: &str) -> bool {
    match keyword {
        "pattern" | "format" | "minLength" | "maxLength" => kind == "string",
        "minimum" | "maximum" | "exclusiveMinimum" | "exclusiveMaximum" => {
            kind == "integer" || kind == "number"
        }
        "items" | "prefixItems" | "minItems" | "maxItems" => kind == "array",
        "properties"
        | "required"
        | "additionalProperties"
        | "propertyNames"
        | "minProperties"
        | "maxProperties" => kind == "object",
        "enum" => true,
        "anyOf" | "oneOf" => true,
        _ => false,
    }
}

fn validate_schema_node(schema: &Value, path: &str, depth: usize) -> Result<(), SchemaError> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(schema_error(
            path,
            format!("schema nesting exceeds {MAX_SCHEMA_DEPTH}"),
        ));
    }
    if schema.is_boolean() {
        return Ok(());
    }
    let object = schema
        .as_object()
        .ok_or_else(|| schema_error(path, "schema must be an object or boolean"))?;

    const ALLOWED: &[&str] = &[
        "$schema",
        "$id",
        "$anchor",
        "$comment",
        "$defs",
        "definitions",
        "$ref",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "type",
        "const",
        "enum",
        "anyOf",
        "oneOf",
        "allOf",
        "if",
        "then",
        "else",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "prefixItems",
        "minItems",
        "maxItems",
        "pattern",
        "format",
        "minLength",
        "maxLength",
        "minProperties",
        "maxProperties",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "propertyNames",
    ];
    for keyword in object.keys() {
        if !ALLOWED.contains(&keyword.as_str()) {
            return Err(schema_error(
                &format!("{path}/{keyword}"),
                "unsupported JSON Schema assertion",
            ));
        }
    }

    if object.contains_key("$ref") {
        let reference = object["$ref"]
            .as_str()
            .ok_or_else(|| schema_error(&format!("{path}/$ref"), "$ref must be a string"))?;
        if !reference.starts_with('#') || reference.contains('%') {
            return Err(schema_error(
                &format!("{path}/$ref"),
                "only unescaped local JSON Pointer references are supported",
            ));
        }
        for keyword in object.keys() {
            if keyword != "$ref" && !is_schema_annotation(keyword) {
                return Err(schema_error(
                    path,
                    format!("assertion sibling {keyword:?} beside $ref cannot be merged exactly"),
                ));
            }
        }
    }

    if let Some(kind) = object.get("type") {
        let valid = match kind {
            Value::String(value) => is_json_type(value),
            Value::Array(values) => {
                !values.is_empty()
                    && values
                        .iter()
                        .all(|value| value.as_str().is_some_and(is_json_type))
            }
            _ => false,
        };
        if !valid {
            return Err(schema_error(
                &format!("{path}/type"),
                "type must name one or more JSON instance types",
            ));
        }
    }

    if let Some(Value::Array(values)) = object.get("enum") {
        if values.len() > MAX_ENUM_VALUES {
            return Err(schema_error(
                &format!("{path}/enum"),
                format!("{} values exceed {MAX_ENUM_VALUES}", values.len()),
            ));
        }
    } else if object.contains_key("enum") {
        return Err(schema_error(
            &format!("{path}/enum"),
            "enum must be an array",
        ));
    }

    if object.contains_key("pattern") {
        if object.get("pattern").and_then(Value::as_str).is_none() {
            return Err(schema_error(
                &format!("{path}/pattern"),
                "pattern must be a string",
            ));
        }
        if object.contains_key("minLength") || object.contains_key("maxLength") {
            return Err(schema_error(
                path,
                "pattern combined with minLength/maxLength is not exactly representable",
            ));
        }
    }

    if let Some(format) = object.get("format") {
        let format = format
            .as_str()
            .ok_or_else(|| schema_error(&format!("{path}/format"), "format must be a string"))?;
        if !is_supported_string_format(format) {
            return Err(schema_error(
                &format!("{path}/format"),
                format!("unsupported string format {format:?}"),
            ));
        }
        if object.contains_key("pattern")
            || object.contains_key("minLength")
            || object.contains_key("maxLength")
        {
            return Err(schema_error(
                path,
                "format combined with pattern/minLength/maxLength is not exactly representable",
            ));
        }
    }

    for keyword in [
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
    ] {
        if let Some(value) = object.get(keyword) {
            let bound = value.as_u64().ok_or_else(|| {
                schema_error(
                    &format!("{path}/{keyword}"),
                    "bound must be a nonnegative integer",
                )
            })?;
            if bound > 2000 {
                return Err(schema_error(
                    &format!("{path}/{keyword}"),
                    "bound exceeds repetition limit 2000",
                ));
            }
        }
    }

    let has_numeric_bound = ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]
        .iter()
        .any(|keyword| object.contains_key(*keyword));
    if has_numeric_bound {
        if object.get("type").and_then(Value::as_str) != Some("integer") {
            return Err(schema_error(
                path,
                "numeric bounds are currently exact only for integer schemas",
            ));
        }
        integer_range_gbnf(object)?;
    }

    if let Some(properties) = object.get("properties") {
        let properties = properties.as_object().ok_or_else(|| {
            schema_error(
                &format!("{path}/properties"),
                "properties must be an object",
            )
        })?;
        if properties.len() > 32 {
            return Err(schema_error(
                &format!("{path}/properties"),
                format!("{} properties exceed 32", properties.len()),
            ));
        }
        for (name, child) in properties {
            validate_schema_node(child, &format!("{path}/properties/{name}"), depth + 1)?;
        }
    }

    if let Some(property_names) = object.get("propertyNames") {
        let tautology = match property_names {
            Value::Bool(true) => true,
            Value::Object(schema) if schema.is_empty() => true,
            Value::Object(schema) => {
                schema.len() == 1 && schema.get("type").and_then(Value::as_str) == Some("string")
            }
            _ => false,
        };
        if !tautology {
            return Err(schema_error(
                &format!("{path}/propertyNames"),
                "constrained propertyNames is not exactly representable",
            ));
        }
    }

    if let Some(required) = object.get("required") {
        let required = required.as_array().ok_or_else(|| {
            schema_error(&format!("{path}/required"), "required must be an array")
        })?;
        let properties = object
            .get("properties")
            .and_then(Value::as_object)
            .ok_or_else(|| schema_error(path, "required needs an explicit properties object"))?;
        let mut names = HashSet::new();
        for (index, value) in required.iter().enumerate() {
            let name = value.as_str().ok_or_else(|| {
                schema_error(
                    &format!("{path}/required/{index}"),
                    "required entries must be strings",
                )
            })?;
            if !properties.contains_key(name) {
                return Err(schema_error(
                    &format!("{path}/required/{index}"),
                    format!("required property {name:?} is not declared"),
                ));
            }
            if !names.insert(name) {
                return Err(schema_error(
                    &format!("{path}/required/{index}"),
                    format!("duplicate required property {name:?}"),
                ));
            }
        }
    }

    if let Some(additional) = object.get("additionalProperties") {
        if !additional.is_boolean() && !additional.is_object() {
            return Err(schema_error(
                &format!("{path}/additionalProperties"),
                "additionalProperties must be a schema",
            ));
        }
        validate_schema_node(
            additional,
            &format!("{path}/additionalProperties"),
            depth + 1,
        )?;
    }

    for container in ["$defs", "definitions", "properties"] {
        if let Some(values) = object.get(container).and_then(Value::as_object) {
            if container != "properties" {
                for (name, child) in values {
                    validate_schema_node(child, &format!("{path}/{container}/{name}"), depth + 1)?;
                }
            }
        }
    }
    if let Some(items) = object.get("items") {
        validate_schema_node(items, &format!("{path}/items"), depth + 1)?;
    }
    if let Some(prefix) = object.get("prefixItems") {
        let prefix = prefix.as_array().ok_or_else(|| {
            schema_error(
                &format!("{path}/prefixItems"),
                "prefixItems must be an array",
            )
        })?;
        if prefix.len() > 32 {
            return Err(schema_error(
                &format!("{path}/prefixItems"),
                format!("{} entries exceed 32", prefix.len()),
            ));
        }
        for (index, child) in prefix.iter().enumerate() {
            validate_schema_node(child, &format!("{path}/prefixItems/{index}"), depth + 1)?;
        }
    }
    for keyword in ["anyOf", "oneOf", "allOf"] {
        if let Some(branches) = object.get(keyword) {
            let branches = branches.as_array().ok_or_else(|| {
                schema_error(
                    &format!("{path}/{keyword}"),
                    format!("{keyword} must be an array"),
                )
            })?;
            for (index, branch) in branches.iter().enumerate() {
                validate_schema_node(branch, &format!("{path}/{keyword}/{index}"), depth + 1)?;
            }
        }
    }
    for keyword in ["if", "then", "else"] {
        if let Some(branch) = object.get(keyword) {
            validate_schema_node(branch, &format!("{path}/{keyword}"), depth + 1)?;
        }
    }
    Ok(())
}

fn is_json_type(value: &str) -> bool {
    matches!(
        value,
        "null" | "boolean" | "object" | "array" | "number" | "integer" | "string"
    )
}

fn schema_types(schema: &Value) -> Option<HashSet<String>> {
    let object = schema.as_object()?;
    match object.get("type") {
        Some(Value::String(value)) => Some([value.clone()].into_iter().collect()),
        Some(Value::Array(values)) => Some(
            values
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect(),
        ),
        _ => None,
    }
}

fn finite_values(schema: &Value) -> Option<HashSet<String>> {
    let object = schema.as_object()?;
    if let Some(value) = object.get("const") {
        return Some([serde_json::to_string(value).ok()?].into_iter().collect());
    }
    object.get("enum").and_then(Value::as_array).map(|values| {
        values
            .iter()
            .filter_map(|value| serde_json::to_string(value).ok())
            .collect()
    })
}

fn finite_candidates(object: &serde_json::Map<String, Value>) -> Option<Vec<Value>> {
    let mut values = if let Some(value) = object.get("const") {
        vec![value.clone()]
    } else if let Some(Value::Array(values)) = object.get("enum") {
        values.clone()
    } else {
        return None;
    };
    if let Some(Value::Array(enumeration)) = object.get("enum") {
        values.retain(|candidate| enumeration.contains(candidate));
    }
    let mut seen = HashSet::new();
    values.retain(|candidate| {
        serde_json::to_string(candidate)
            .ok()
            .is_some_and(|encoded| seen.insert(encoded))
    });
    Some(values)
}

fn instance_matches_schema(
    instance: &Value,
    schema: &Value,
    root: &Value,
    depth: usize,
) -> Result<bool, SchemaError> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(schema_error(
            "/",
            "finite-value validation exceeds schema depth",
        ));
    }
    if let Some(allowed) = schema.as_bool() {
        return Ok(allowed);
    }
    let object = schema
        .as_object()
        .ok_or_else(|| schema_error("/", "schema must be an object or boolean"))?;
    if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
        let target = if reference == "#" {
            root
        } else {
            root.pointer(reference.trim_start_matches('#'))
                .ok_or_else(|| {
                    schema_error("/$ref", format!("unresolved local reference {reference:?}"))
                })?
        };
        return instance_matches_schema(instance, target, root, depth + 1);
    }
    if object.get("const").is_some_and(|value| value != instance) {
        return Ok(false);
    }
    if object
        .get("enum")
        .and_then(Value::as_array)
        .is_some_and(|values| !values.contains(instance))
    {
        return Ok(false);
    }
    if let Some(branches) = object.get("allOf").and_then(Value::as_array) {
        for branch in branches {
            if !instance_matches_schema(instance, branch, root, depth + 1)? {
                return Ok(false);
            }
        }
    }
    if let Some(branches) = object.get("anyOf").and_then(Value::as_array) {
        let mut matched = false;
        for branch in branches {
            matched |= instance_matches_schema(instance, branch, root, depth + 1)?;
        }
        if !matched {
            return Ok(false);
        }
    }
    if let Some(branches) = object.get("oneOf").and_then(Value::as_array) {
        let mut matches = 0usize;
        for branch in branches {
            matches += usize::from(instance_matches_schema(instance, branch, root, depth + 1)?);
        }
        if matches != 1 {
            return Ok(false);
        }
    }
    if let Some(condition) = object.get("if") {
        let condition_matches = instance_matches_schema(instance, condition, root, depth + 1)?;
        let selected = if condition_matches {
            object.get("then")
        } else {
            object.get("else")
        };
        if let Some(selected) = selected {
            if !instance_matches_schema(instance, selected, root, depth + 1)? {
                return Ok(false);
            }
        }
    }

    if let Some(kind) = object.get("type") {
        let kinds: Vec<&str> = match kind {
            Value::String(kind) => vec![kind],
            Value::Array(kinds) => kinds.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        if !kinds
            .iter()
            .any(|kind| instance_matches_type(instance, kind))
        {
            return Ok(false);
        }
    }

    if let Some(text) = instance.as_str() {
        let length = text.chars().count() as u64;
        if object
            .get("minLength")
            .and_then(Value::as_u64)
            .is_some_and(|min| length < min)
            || object
                .get("maxLength")
                .and_then(Value::as_u64)
                .is_some_and(|max| length > max)
        {
            return Ok(false);
        }
        if let Some(pattern) = object.get("pattern").and_then(Value::as_str) {
            let expression = regex::Regex::new(pattern)
                .map_err(|error| schema_error("/pattern", format!("invalid regex: {error}")))?;
            if !expression.is_match(text) {
                return Ok(false);
            }
        }
        if let Some(format) = object.get("format").and_then(Value::as_str) {
            let matches = match format {
                "json-pointer" => is_json_pointer(text),
                "relative-json-pointer" => is_relative_json_pointer(text),
                _ => regex::Regex::new(
                    string_format_pattern(format)
                        .ok_or_else(|| schema_error("/format", "unsupported string format"))?,
                )
                .map_err(|error| schema_error("/format", format!("invalid format regex: {error}")))?
                .is_match(text),
            };
            if !matches {
                return Ok(false);
            }
        }
    }

    if let Some(number) = instance.as_f64() {
        if object
            .get("minimum")
            .and_then(Value::as_f64)
            .is_some_and(|min| number < min)
            || object
                .get("maximum")
                .and_then(Value::as_f64)
                .is_some_and(|max| number > max)
            || object
                .get("exclusiveMinimum")
                .and_then(Value::as_f64)
                .is_some_and(|min| number <= min)
            || object
                .get("exclusiveMaximum")
                .and_then(Value::as_f64)
                .is_some_and(|max| number >= max)
        {
            return Ok(false);
        }
    }

    if let Some(values) = instance.as_array() {
        if object
            .get("minItems")
            .and_then(Value::as_u64)
            .is_some_and(|min| values.len() < min as usize)
            || object
                .get("maxItems")
                .and_then(Value::as_u64)
                .is_some_and(|max| values.len() > max as usize)
        {
            return Ok(false);
        }
        let prefix = object.get("prefixItems").and_then(Value::as_array);
        for (index, value) in values.iter().enumerate() {
            let item_schema = prefix
                .and_then(|items| items.get(index))
                .or_else(|| object.get("items"));
            if let Some(item_schema) = item_schema {
                if !instance_matches_schema(value, item_schema, root, depth + 1)? {
                    return Ok(false);
                }
            }
        }
    }

    if let Some(values) = instance.as_object() {
        if object
            .get("minProperties")
            .and_then(Value::as_u64)
            .is_some_and(|min| values.len() < min as usize)
            || object
                .get("maxProperties")
                .and_then(Value::as_u64)
                .is_some_and(|max| values.len() > max as usize)
        {
            return Ok(false);
        }
        if let Some(required) = object.get("required").and_then(Value::as_array) {
            if required
                .iter()
                .filter_map(Value::as_str)
                .any(|name| !values.contains_key(name))
            {
                return Ok(false);
            }
        }
        let properties = object.get("properties").and_then(Value::as_object);
        for (name, value) in values {
            if let Some(child) = properties.and_then(|properties| properties.get(name)) {
                if !instance_matches_schema(value, child, root, depth + 1)? {
                    return Ok(false);
                }
            } else if let Some(additional) = object.get("additionalProperties") {
                if !instance_matches_schema(value, additional, root, depth + 1)? {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

fn instance_matches_type(instance: &Value, kind: &str) -> bool {
    match kind {
        "null" => instance.is_null(),
        "boolean" => instance.is_boolean(),
        "object" => instance.is_object(),
        "array" => instance.is_array(),
        "number" => instance.is_number(),
        "integer" => instance
            .as_f64()
            .is_some_and(|number| number.is_finite() && number.fract() == 0.0),
        "string" => instance.is_string(),
        _ => false,
    }
}

fn is_json_pointer(value: &str) -> bool {
    value.is_empty()
        || (value.starts_with('/')
            && value
                .split('/')
                .skip(1)
                .all(|part| !part.contains('~') || valid_pointer_escapes(part)))
}

fn valid_pointer_escapes(value: &str) -> bool {
    let mut chars = value.chars();
    while let Some(character) = chars.next() {
        if character == '~' && !matches!(chars.next(), Some('0' | '1')) {
            return false;
        }
    }
    true
}

fn is_relative_json_pointer(value: &str) -> bool {
    let digits = value.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 || (digits > 1 && value.starts_with('0')) {
        return false;
    }
    let suffix = &value[digits..];
    suffix == "#" || is_json_pointer(suffix)
}

fn types_overlap(left: &HashSet<String>, right: &HashSet<String>) -> bool {
    left.iter().any(|kind| {
        right.contains(kind)
            || (kind == "integer" && right.contains("number"))
            || (kind == "number" && right.contains("integer"))
    })
}

fn schemas_provably_disjoint(left: &Value, right: &Value) -> bool {
    if left == &Value::Bool(false) || right == &Value::Bool(false) {
        return true;
    }
    if let (Some(left_values), Some(right_values)) = (finite_values(left), finite_values(right)) {
        return left_values.is_disjoint(&right_values);
    }
    if let (Some(left_types), Some(right_types)) = (schema_types(left), schema_types(right)) {
        if !types_overlap(&left_types, &right_types) {
            return true;
        }
    }
    let (Some(left_object), Some(right_object)) = (left.as_object(), right.as_object()) else {
        return false;
    };
    let left_required: HashSet<&str> = left_object
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let right_required: HashSet<&str> = right_object
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let left_properties = left_object.get("properties").and_then(Value::as_object);
    let right_properties = right_object.get("properties").and_then(Value::as_object);
    let (Some(left_properties), Some(right_properties)) = (left_properties, right_properties)
    else {
        return false;
    };
    left_properties.iter().any(|(name, left_schema)| {
        left_required.contains(name.as_str())
            && right_required.contains(name.as_str())
            && right_properties
                .get(name)
                .is_some_and(|right_schema| schemas_provably_disjoint(left_schema, right_schema))
    })
}

fn merge_schemas(left: &Value, right: &Value, path: &str) -> Result<Value, SchemaError> {
    match (left, right) {
        (Value::Bool(false), _) | (_, Value::Bool(false)) => return Ok(Value::Bool(false)),
        (Value::Bool(true), value) | (value, Value::Bool(true)) => return Ok(value.clone()),
        _ => {}
    }
    let mut merged = left
        .as_object()
        .cloned()
        .ok_or_else(|| schema_error(path, "allOf branches must be schemas"))?;
    let right = right
        .as_object()
        .ok_or_else(|| schema_error(path, "allOf branches must be schemas"))?;
    for (keyword, value) in right {
        if is_schema_annotation(keyword) {
            merged
                .entry(keyword.clone())
                .or_insert_with(|| value.clone());
            continue;
        }
        match keyword.as_str() {
            "type" => {
                if let Some(existing) = merged.get("type") {
                    let left_types = type_value_set(existing, path)?;
                    let right_types = type_value_set(value, path)?;
                    let mut intersection: Vec<String> =
                        left_types.intersection(&right_types).cloned().collect();
                    if left_types.contains("number") && right_types.contains("integer")
                        || left_types.contains("integer") && right_types.contains("number")
                    {
                        intersection.push("integer".to_string());
                    }
                    intersection.sort();
                    intersection.dedup();
                    if intersection.is_empty() {
                        return Ok(Value::Bool(false));
                    }
                    merged.insert(
                        "type".into(),
                        if intersection.len() == 1 {
                            Value::String(intersection.remove(0))
                        } else {
                            Value::Array(intersection.into_iter().map(Value::String).collect())
                        },
                    );
                } else {
                    merged.insert(keyword.clone(), value.clone());
                }
            }
            "required" => {
                let mut names: Vec<Value> = merged
                    .get("required")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                names.extend(value.as_array().cloned().ok_or_else(|| {
                    schema_error(&format!("{path}/required"), "required must be an array")
                })?);
                names.sort_by_key(|entry| entry.as_str().unwrap_or_default().to_string());
                names.dedup();
                merged.insert("required".into(), Value::Array(names));
            }
            "properties" => {
                let mut properties = merged
                    .get("properties")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                for (name, child) in value.as_object().ok_or_else(|| {
                    schema_error(
                        &format!("{path}/properties"),
                        "properties must be an object",
                    )
                })? {
                    if let Some(existing) = properties.get(name) {
                        properties.insert(
                            name.clone(),
                            merge_schemas(existing, child, &format!("{path}/properties/{name}"))?,
                        );
                    } else {
                        properties.insert(name.clone(), child.clone());
                    }
                }
                merged.insert("properties".into(), Value::Object(properties));
            }
            "minimum" | "exclusiveMinimum" | "minLength" | "minItems" | "minProperties" => {
                let chosen = match merged.get(keyword) {
                    Some(existing) if number_as_f64(existing)? >= number_as_f64(value)? => {
                        existing.clone()
                    }
                    _ => value.clone(),
                };
                merged.insert(keyword.clone(), chosen);
            }
            "maximum" | "exclusiveMaximum" | "maxLength" | "maxItems" | "maxProperties" => {
                let chosen = match merged.get(keyword) {
                    Some(existing) if number_as_f64(existing)? <= number_as_f64(value)? => {
                        existing.clone()
                    }
                    _ => value.clone(),
                };
                merged.insert(keyword.clone(), chosen);
            }
            "additionalProperties" => match merged.get(keyword) {
                None | Some(Value::Bool(true)) => {
                    merged.insert(keyword.clone(), value.clone());
                }
                Some(Value::Bool(false)) => {}
                Some(existing) if existing == value => {}
                Some(_) if value == &Value::Bool(false) => {
                    merged.insert(keyword.clone(), Value::Bool(false));
                }
                Some(_) => {
                    return Err(schema_error(
                        &format!("{path}/{keyword}"),
                        "additionalProperties intersection is not exactly representable",
                    ));
                }
            },
            "const" | "enum" => {
                let current = Value::Object(merged.clone());
                let mut right_finite = serde_json::Map::new();
                right_finite.insert(keyword.clone(), value.clone());
                if let (Some(left_values), Some(right_values)) = (
                    finite_values(&current),
                    finite_values(&Value::Object(right_finite)),
                ) {
                    let values: Vec<Value> = left_values
                        .intersection(&right_values)
                        .filter_map(|encoded| serde_json::from_str(encoded).ok())
                        .collect();
                    if values.is_empty() {
                        return Ok(Value::Bool(false));
                    }
                    merged.remove("const");
                    merged.insert("enum".into(), Value::Array(values));
                } else {
                    merged.insert(keyword.clone(), value.clone());
                }
            }
            _ => match merged.get(keyword) {
                None => {
                    merged.insert(keyword.clone(), value.clone());
                }
                Some(existing) if existing == value => {}
                Some(_) => {
                    return Err(schema_error(
                        &format!("{path}/{keyword}"),
                        "allOf intersection is not exactly representable",
                    ));
                }
            },
        }
    }
    Ok(Value::Object(merged))
}

fn type_value_set(value: &Value, path: &str) -> Result<HashSet<String>, SchemaError> {
    match value {
        Value::String(kind) => Ok([kind.clone()].into_iter().collect()),
        Value::Array(kinds) => kinds
            .iter()
            .map(|kind| {
                kind.as_str()
                    .map(ToOwned::to_owned)
                    .ok_or_else(|| schema_error(path, "type union entries must be strings"))
            })
            .collect(),
        _ => Err(schema_error(path, "type must be a string or array")),
    }
}

fn number_as_f64(value: &Value) -> Result<f64, SchemaError> {
    value
        .as_f64()
        .ok_or_else(|| schema_error("/", "numeric bound must be a finite JSON number"))
}

fn expand_conditional(
    base: &Value,
    conditional: &Value,
    path: &str,
) -> Result<Vec<Value>, SchemaError> {
    let object = conditional
        .as_object()
        .ok_or_else(|| schema_error(&format!("{path}/allOf/0"), "conditional must be an object"))?;
    let predicate = object
        .get("if")
        .and_then(Value::as_object)
        .ok_or_else(|| schema_error(&format!("{path}/allOf/0/if"), "if must be an object"))?;
    let predicate_properties = predicate
        .get("properties")
        .and_then(Value::as_object)
        .ok_or_else(|| schema_error(&format!("{path}/allOf/0/if"), "if needs properties"))?;
    if predicate_properties.len() != 1 {
        return Err(schema_error(
            &format!("{path}/allOf/0/if/properties"),
            "conditional discriminator must contain exactly one property",
        ));
    }
    let (name, condition_schema) = predicate_properties.iter().next().expect("one property");
    let predicate_required = predicate
        .get("required")
        .and_then(Value::as_array)
        .is_some_and(|required| required.iter().any(|entry| entry.as_str() == Some(name)));
    if !predicate_required {
        return Err(schema_error(
            &format!("{path}/allOf/0/if/required"),
            "conditional discriminator must be required",
        ));
    }
    let original = base
        .get("properties")
        .and_then(Value::as_object)
        .and_then(|properties| properties.get(name))
        .ok_or_else(|| {
            schema_error(
                &format!("{path}/properties/{name}"),
                "conditional discriminator must be declared by the base schema",
            )
        })?;

    let mut matched = base.clone();
    set_property_schema(
        &mut matched,
        name,
        merge_schemas(original, condition_schema, path)?,
    )?;
    add_required_property(&mut matched, name)?;
    if let Some(then_schema) = object.get("then") {
        matched = merge_schemas(&matched, then_schema, &format!("{path}/allOf/0/then"))?;
    }

    let complement = complement_discriminator(original, condition_schema, path)?;
    let mut unmatched = base.clone();
    set_property_schema(&mut unmatched, name, complement)?;
    if let Some(else_schema) = object.get("else") {
        unmatched = merge_schemas(&unmatched, else_schema, &format!("{path}/allOf/0/else"))?;
    }
    Ok(vec![matched, unmatched])
}

fn set_property_schema(target: &mut Value, name: &str, schema: Value) -> Result<(), SchemaError> {
    target
        .as_object_mut()
        .and_then(|object| object.get_mut("properties"))
        .and_then(Value::as_object_mut)
        .ok_or_else(|| schema_error("/properties", "base schema needs properties"))?
        .insert(name.to_string(), schema);
    Ok(())
}

fn add_required_property(target: &mut Value, name: &str) -> Result<(), SchemaError> {
    let object = target
        .as_object_mut()
        .ok_or_else(|| schema_error("/", "base schema must be an object"))?;
    let required = object
        .entry("required")
        .or_insert_with(|| Value::Array(Vec::new()))
        .as_array_mut()
        .ok_or_else(|| schema_error("/required", "required must be an array"))?;
    if !required.iter().any(|entry| entry.as_str() == Some(name)) {
        required.push(Value::String(name.to_string()));
    }
    Ok(())
}

fn complement_discriminator(
    original: &Value,
    condition: &Value,
    path: &str,
) -> Result<Value, SchemaError> {
    if condition.get("type").and_then(Value::as_str) == Some("null") {
        let mut object = original
            .as_object()
            .cloned()
            .ok_or_else(|| schema_error(path, "discriminator schema must be an object"))?;
        let mut kinds: Vec<String> = schema_types(original)
            .unwrap_or_else(|| {
                ["boolean", "object", "array", "number", "string"]
                    .into_iter()
                    .map(ToOwned::to_owned)
                    .collect()
            })
            .into_iter()
            .filter(|kind| kind != "null")
            .collect();
        kinds.sort();
        if kinds.is_empty() {
            return Ok(Value::Bool(false));
        }
        object.insert(
            "type".into(),
            if kinds.len() == 1 {
                Value::String(kinds.remove(0))
            } else {
                Value::Array(kinds.into_iter().map(Value::String).collect())
            },
        );
        return Ok(Value::Object(object));
    }
    let Some(original_values) = finite_values(original) else {
        return Err(schema_error(
            path,
            "finite const/enum discriminator needs a finite base domain",
        ));
    };
    let Some(condition_values) = finite_values(condition) else {
        return Err(schema_error(
            path,
            "conditional discriminator must be type:null, const, or enum",
        ));
    };
    let values: Vec<Value> = original_values
        .difference(&condition_values)
        .filter_map(|encoded| serde_json::from_str(encoded).ok())
        .collect();
    if values.is_empty() {
        return Ok(Value::Bool(false));
    }
    let mut object = original.as_object().cloned().unwrap_or_default();
    object.remove("const");
    object.insert("enum".into(), Value::Array(values));
    Ok(Value::Object(object))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::parser::parse;
    use super::super::sampler::GrammarRuntime;
    use super::*;

    fn compile(schema_json: &str) -> String {
        let schema: Value = serde_json::from_str(schema_json).unwrap();
        schema_to_gbnf(&schema).unwrap_or_else(|e| panic!("schema_to_gbnf: {:?}", e))
    }

    fn runtime(schema_json: &str) -> GrammarRuntime {
        let gbnf = compile(schema_json);
        let g = super::super::parser::parse_generated(&gbnf)
            .unwrap_or_else(|e| panic!("parse gbnf:\n{}\nerror: {}", gbnf, e));
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
        assert!(
            rt.accept_bytes(b"{\"age\":30,\"name\":\"Bob\"}"),
            "age-first rejected"
        );
        assert!(rt.is_accepted());
        // name first (non-alphabetical — was broken before iter 75).
        let mut rt2 = runtime(schema);
        assert!(
            rt2.accept_bytes(b"{\"name\":\"Bob\",\"age\":30}"),
            "name-first rejected"
        );
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
        assert!(rt.accept_bytes(b"{\"arguments\":{\"city\":\"NYC\"},\"name\":\"get_weather\"}"));
        assert!(rt.is_accepted());
        // name-first (non-alphabetical — iter 75 fix).
        let mut rt2 = runtime(schema);
        assert!(rt2.accept_bytes(b"{\"name\":\"get_weather\",\"arguments\":{\"city\":\"NYC\"}}"));
        assert!(rt2.is_accepted());
    }

    #[test]
    fn unsupported_type_rejected_at_compile_time() {
        let schema: Value = serde_json::from_str(r#"{"type":"notathing"}"#).unwrap();
        let err = schema_to_gbnf(&schema).unwrap_err();
        assert!(err.to_string().contains("JSON instance types"));
    }

    #[test]
    fn pattern_is_enforced_in_response_schema() {
        let schema = r#"{"type":"string","pattern":"^[a-z]+$"}"#;
        let mut good = runtime(schema);
        assert!(good.accept_bytes(b"\"lowercase\""));
        assert!(good.is_accepted());
        let mut bad = runtime(schema);
        assert!(!bad.accept_bytes(b"\"ABC123\""));
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
        assert!(
            !(ok && rt.is_accepted()),
            "accepted empty object missing required city"
        );
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
        let payload = br#"{"filters":{"max_price":2000,"min_price":500},"query":"laptops"}"#;
        assert!(rt.accept_bytes(payload), "rejected nested (filters-first)");
        assert!(rt.is_accepted());

        // query before filters (non-alphabetical — iter 75 fix).
        let mut rt2 = runtime(schema);
        let payload2 = br#"{"query":"laptops","filters":{"max_price":2000,"min_price":500}}"#;
        assert!(rt2.accept_bytes(payload2), "rejected nested (query-first)");
        assert!(rt2.is_accepted());

        // Critical bug-fix anchor: missing the SECOND required field
        // (query) must be REJECTED. Pre-iter-74 this was falsely accepted.
        let mut rt = runtime(schema);
        let missing = br#"{"filters":{"max_price":2000,"min_price":500}}"#;
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
        assert!(
            !(ok && rt.is_accepted()),
            "accepted object missing required 'unit'"
        );
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
        assert!(rt.accept_bytes(br#"{"tags":["news","tech"],"url":"https://example.com"}"#));
        assert!(rt.is_accepted());

        // url first (non-alphabetical — iter 75).
        let mut rt = runtime(schema);
        assert!(rt.accept_bytes(br#"{"url":"https://example.com","tags":["news","tech"]}"#));
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
        assert!(
            !(ok && rt.is_accepted()),
            "accepted object missing required 'c'"
        );

        // Missing required 'a'.
        let mut rt = runtime(schema);
        let ok = rt.accept_bytes(br#"{"b":2,"c":3}"#);
        assert!(
            !(ok && rt.is_accepted()),
            "accepted object missing required 'a'"
        );
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
        assert!(
            rt.is_accepted(),
            "not accepted after optional-before-required"
        );
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
        let msg = err.to_string();
        assert!(
            msg.contains("33") || msg.contains("max supported"),
            "expected error mentioning property count or 'max supported'; got: {:?}",
            msg
        );
    }

    /// T1.8 large-schema guard: a schema with 4 required + 4 optional = 8
    /// total properties must compile successfully.
    ///
    /// Wave 2.6 W-β2: the required-key cap is 8 (ANY_ORDER_MAX_REQUIRED).
    /// The n_total > 32 guard still caps total properties at 32.  This test
    /// uses a small, fast schema to verify the happy-path for mixed
    /// required/optional, staying well within both caps.
    ///
    /// The original test exercised the now-deleted sequential fallback with
    /// 32 all-required keys.  That path no longer exists; the cap is 8 req.
    #[test]
    fn large_schema_32_properties_compiles_ok() {
        let mut props = serde_json::Map::new();
        let mut required = Vec::new();
        // 4 required keys.
        for i in 0..4usize {
            let key = format!("req{:02}", i);
            props.insert(key.clone(), serde_json::json!({"type": "string"}));
            required.push(serde_json::Value::String(key));
        }
        // 4 optional keys.
        for i in 0..4usize {
            let key = format!("opt{:02}", i);
            props.insert(key.clone(), serde_json::json!({"type": "string"}));
        }
        let schema = serde_json::Value::Object({
            let mut m = serde_json::Map::new();
            m.insert("type".into(), serde_json::json!("object"));
            m.insert("properties".into(), serde_json::Value::Object(props));
            m.insert("required".into(), serde_json::Value::Array(required));
            m
        });

        // Must not error — 4 required (≤ 8 cap) + 4 optional = 8 total (≤ 32 cap).
        let result = schema_to_gbnf(&schema);
        assert!(
            result.is_ok(),
            "4-required + 4-optional schema failed to compile: {:?}",
            result.err()
        );
    }

    // -----------------------------------------------------------------
    // ADR-052 — hard 400 above the 12-key bounded permutation budget.
    //
    // Research grounding: Moshier & Rounds ACL 1987 prove CFG-for-permutations
    // of n required keys is exponential. All production engines (the peer,
    // llguidance, xgrammar, outlines-core) enforce declaration order at large N.
    // Sequential-sorted fallback is a semantic downgrade (Wave-2.5 mantra
    // violation). Replaced with hard SchemaError → HTTP 400.
    // -----------------------------------------------------------------

    /// Boundary at 13 required keys (just over ANY_ORDER_MAX_REQUIRED=12):
    /// must return SchemaError with an operator-actionable message.
    /// HTTP 400 propagation is handled by compile_tool_grammar → ApiError.
    #[test]
    fn thirteen_required_keys_returns_too_many_required_keys() {
        let mut props = serde_json::Map::new();
        let mut required = Vec::new();
        for i in 0..13usize {
            let key = format!("k{}", i);
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
        // W-ζ LOW: assert the typed TooManyRequiredKeys variant is returned.
        match &err {
            SchemaError::TooManyRequiredKeys { count, max, .. } => {
                assert_eq!(*count, 13, "variant must carry count=13");
                assert_eq!(*max, 12_usize, "variant must carry max=12");
            }
            other => panic!("expected TooManyRequiredKeys variant; got {:?}", other),
        }
        // Display must be operator-actionable: count, limit, citation, action.
        let msg = err.to_string();
        assert!(
            msg.contains("13") && msg.contains("12"),
            "expected error mentioning count=13 and limit=12; got: {:?}",
            msg
        );
        assert!(
            msg.contains("Moshier") || msg.contains("ADR-005"),
            "expected operator-actionable citation; got: {:?}",
            msg
        );
        assert!(
            msg.contains("Reduce") || msg.contains("split"),
            "expected actionable instruction; got: {:?}",
            msg
        );
    }

    /// Boundary at exactly 12 required keys (the supported maximum):
    /// must compile successfully with full any-position semantics.
    #[test]
    fn twelve_required_keys_compiles_ok() {
        let mut props = serde_json::Map::new();
        let mut required = Vec::new();
        for i in 0..12usize {
            let key = format!("k{}", i);
            props.insert(key.clone(), serde_json::json!({"type": "integer"}));
            required.push(serde_json::Value::String(key));
        }
        let schema = serde_json::Value::Object({
            let mut m = serde_json::Map::new();
            m.insert("type".into(), serde_json::json!("object"));
            m.insert("properties".into(), serde_json::Value::Object(props));
            m.insert("required".into(), serde_json::Value::Array(required));
            m
        });

        // Must compile without error.
        let result = schema_to_gbnf(&schema);
        assert!(
            result.is_ok(),
            "12 required keys should compile (is the supported max); got: {:?}",
            result.err()
        );

        // Spot-check: any-position must be enforced — reversed order accepted.
        let gbnf = result.unwrap();
        let g = super::super::parser::parse_generated(&gbnf)
            .unwrap_or_else(|e| panic!("parse gbnf: {}", e));
        let rid = g.rule_id("root").unwrap();
        let mut rt = GrammarRuntime::new(g, rid).unwrap();
        // Emit keys in reverse alphabetical order (k7..k0).
        let reversed = br#"{"k11":11,"k10":10,"k9":9,"k8":8,"k7":7,"k6":6,"k5":5,"k4":4,"k3":3,"k2":2,"k1":1,"k0":0}"#;
        assert!(
            rt.accept_bytes(reversed),
            "12-key schema rejected reversed-order input (any-position not enforced)"
        );
        assert!(
            rt.is_accepted(),
            "12-key schema not accepted after reversed input"
        );
    }

    // -----------------------------------------------------------------
    // B4-extras — additionalProperties: multiple trailing extras (Wave 2.6 W-β2)
    //
    // Audit finding: extra wildcard was one-shot in build_optional_chain because
    // _is_wildcard was ignored — the wildcard entry was removed from `remaining`
    // just like a declared optional key.  Fix: when is_wildcard == true, keep
    // the wildcard in remaining so the emitted rule is self-referential (Kleene
    // star via GBNF optional recursion).
    // -----------------------------------------------------------------

    /// When additionalProperties is permissive, a JSON object with multiple
    /// extra keys after the required key(s) must be accepted.
    /// Before W-β2 the wildcard was one-shot; this test fails on the old code.
    #[test]
    fn additional_properties_permissive_accepts_multiple_extras() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }"#;

        // Three extra keys after the required key.
        let mut rt = runtime(schema);
        assert!(
            rt.accept_bytes(br#"{"name":"Alice","x1":"v1","x2":"v2","x3":"v3"}"#),
            "three trailing extra keys were rejected (additionalProperties permissive)"
        );
        assert!(rt.is_accepted());

        // Extra keys before and after the required key.
        let mut rt = runtime(schema);
        assert!(
            rt.accept_bytes(br#"{"before":"b","name":"Alice","after1":"a1","after2":"a2"}"#),
            "extra keys surrounding required key were rejected"
        );
        assert!(rt.is_accepted());

        // Only extra keys in an optional-only object (no required fields).
        let schema_no_req = r#"{
            "type": "object",
            "properties": {
                "opt": {"type": "string"}
            }
        }"#;
        let mut rt = runtime(schema_no_req);
        assert!(
            rt.accept_bytes(br#"{"opt":"v","extra1":"e1","extra2":"e2"}"#),
            "multiple extras in no-required-keys object were rejected"
        );
        assert!(rt.is_accepted());
    }

    // -----------------------------------------------------------------------
    // W-ζ LOW — SchemaError typed variant tests
    //
    // The wave-2.7 audit found that commit 5110dc0 implied a typed
    // SchemaError::TooManyRequiredKeys variant but only had a generic struct.
    // These tests assert:
    //   1. Existing error paths still produce well-formed Display output.
    //   2. The >8-required-keys path emits the new typed variant carrying
    //      fn_name + count (+ max), not the old Generic { message } form.
    // -----------------------------------------------------------------------

    /// W-ζ LOW: existing SchemaError for unsupported type still formats correctly.
    #[test]
    fn schema_error_generic_variant_displays_correctly() {
        let schema: Value = serde_json::from_str(r#"{"type":"notathing"}"#).unwrap();
        let err = schema_to_gbnf(&schema).unwrap_err();
        // Must be Generic variant.
        assert!(
            matches!(&err, SchemaError::Generic { message, .. } if message.contains("JSON instance types")),
            "unsupported-type error must be SchemaError::Generic; got {:?}",
            err
        );
        // Display must include the path and message.
        let s = err.to_string();
        assert!(
            s.contains("json-schema-to-grammar error"),
            "Display must contain prefix: {}",
            s
        );
        assert!(
            s.contains("JSON instance types"),
            "Display must contain message: {}",
            s
        );
    }

    /// W-ζ LOW: >12-required-keys path emits TooManyRequiredKeys variant
    /// carrying fn_name + count.
    #[test]
    fn too_many_required_keys_variant_carries_fn_name_and_count() {
        let mut props = serde_json::Map::new();
        let mut required = Vec::new();
        for i in 0..13usize {
            let k = format!("field{}", i);
            props.insert(k.clone(), serde_json::json!({"type": "string"}));
            required.push(serde_json::Value::String(k));
        }
        let schema = serde_json::json!({
            "type": "object",
            "properties": props,
            "required": required
        });
        let err = schema_to_gbnf(&schema).unwrap_err();
        match &err {
            SchemaError::TooManyRequiredKeys {
                fn_name,
                count,
                max,
            } => {
                // fn_name holds the path (empty = root in schema_to_gbnf context).
                let _ = fn_name; // path is empty string at root; just assert presence
                assert_eq!(*count, 13, "TooManyRequiredKeys must carry count=13");
                assert_eq!(*max, 12, "TooManyRequiredKeys must carry max=12");
            }
            other => panic!("expected SchemaError::TooManyRequiredKeys; got {:?}", other),
        }
        // Display must be actionable.
        let s = err.to_string();
        assert!(s.contains("13"), "Display must mention count: {}", s);
        assert!(s.contains("12"), "Display must mention cap: {}", s);
    }

    fn accepts_fixture(schema: &str, instance: &str) -> bool {
        let schema: Value = serde_json::from_str(schema).expect("schema fixture JSON");
        let instance: Value = serde_json::from_str(instance).expect("instance fixture JSON");
        let bytes = serde_json::to_vec(&instance).expect("serialize instance");
        let grammar = schema_to_gbnf(&schema).expect("compile schema fixture");
        let grammar = super::super::parser::parse_generated(&grammar)
            .unwrap_or_else(|error| panic!("parse generated grammar: {error}\n{grammar}"));
        let root = grammar.rule_id("root").expect("root rule");
        let mut runtime = GrammarRuntime::new(grammar, root).expect("runtime");
        runtime.accept_bytes(&bytes) && runtime.is_accepted()
    }

    #[test]
    fn r2c_stage6_review_lens_fixture_and_mutants() {
        let schema = include_str!(
            "../../../../tests/fixtures/structured_output/r2c/stage6_review_lens.schema.json"
        );
        assert!(accepts_fixture(
            schema,
            include_str!(
                "../../../../tests/fixtures/structured_output/r2c/stage6_review_lens.valid.json"
            )
        ));
        assert!(!accepts_fixture(
            schema,
            include_str!(
                "../../../../tests/fixtures/structured_output/r2c/stage6_review_lens.invalid_disposition.json"
            )
        ));
        assert!(!accepts_fixture(
            schema,
            include_str!(
                "../../../../tests/fixtures/structured_output/r2c/stage6_review_lens.invalid_missing_detail.json"
            )
        ));
    }

    #[test]
    fn r2c_stage9_cwe_fixture_and_mutants() {
        let schema =
            include_str!("../../../../tests/fixtures/structured_output/r2c/stage9_cwe.schema.json");
        for valid in [
            include_str!("../../../../tests/fixtures/structured_output/r2c/stage9_cwe.valid.json"),
            include_str!(
                "../../../../tests/fixtures/structured_output/r2c/stage9_cwe.valid_abstention.json"
            ),
        ] {
            assert!(accepts_fixture(schema, valid));
        }
        for invalid in [
            include_str!(
                "../../../../tests/fixtures/structured_output/r2c/stage9_cwe.invalid_cwe.json"
            ),
            include_str!(
                "../../../../tests/fixtures/structured_output/r2c/stage9_cwe.invalid_abstention.json"
            ),
        ] {
            assert!(!accepts_fixture(schema, invalid));
        }
    }

    #[test]
    fn local_refs_decode_json_pointer_and_reuse_rules() {
        let schema = r##"{
            "$defs":{"a/b":{"type":"string","const":"ok"}},
            "type":"array","items":{"$ref":"#/$defs/a~1b"},"minItems":2,"maxItems":2
        }"##;
        let mut good = runtime(schema);
        assert!(good.accept_bytes(br#"["ok","ok"]"#));
        assert!(good.is_accepted());
        let mut bad = runtime(schema);
        assert!(!bad.accept_bytes(br#"["ok","bad"]"#));
    }

    #[test]
    fn exact_one_of_rejects_overlapping_branches() {
        let schema: Value = serde_json::json!({
            "oneOf": [{"type":"string"}, {"type":"string","minLength":1}]
        });
        let error = schema_to_gbnf(&schema).expect_err("overlap must fail closed");
        assert!(error.to_string().contains("not provably disjoint"));
    }

    #[test]
    fn all_of_intersects_const_and_enum_without_dynamic_key_loss() {
        let schema = serde_json::json!({
            "allOf": [
                {"type":"string", "enum":["allow", "deny"]},
                {"const":"allow"}
            ]
        });
        let grammar = schema_to_gbnf(&schema).expect("intersect allOf");
        let parsed = parse(&grammar).expect("parse allOf grammar");
        let root = parsed.rule_id("root").expect("root");
        let mut allowed = GrammarRuntime::new(parsed.clone(), root).expect("runtime");
        assert!(allowed.accept_bytes(br#""allow""#));
        assert!(allowed.is_accepted());
        let mut denied = GrammarRuntime::new(parsed, root).expect("runtime");
        assert!(!denied.accept_bytes(br#""deny""#) || !denied.is_accepted());
    }

    #[test]
    fn false_schema_compiles_to_empty_language() {
        let grammar = schema_to_gbnf(&Value::Bool(false)).expect("false schema grammar");
        let grammar = parse(&grammar).expect("parse false schema grammar");
        let root = grammar.rule_id("root").expect("root");
        let runtime = GrammarRuntime::new(grammar, root).expect("runtime");
        assert!(!runtime.is_accepted());
        for bytes in [b"null".as_slice(), b"0", br#""x""#, b"{}", b"[]"] {
            let grammar = schema_to_gbnf(&Value::Bool(false)).unwrap();
            let grammar = parse(&grammar).unwrap();
            let root = grammar.rule_id("root").unwrap();
            let mut runtime = GrammarRuntime::new(grammar, root).unwrap();
            assert!(!runtime.accept_bytes(bytes));
        }
    }

    #[test]
    fn unsupported_assertions_fail_closed_with_pointer() {
        for keyword in [
            "multipleOf",
            "uniqueItems",
            "contains",
            "not",
            "patternProperties",
        ] {
            let schema = serde_json::json!({"type":"array", keyword: true});
            let error = schema_to_gbnf(&schema).expect_err(keyword);
            assert!(error.to_string().contains(keyword), "{error}");
        }
    }

    #[test]
    fn bounded_strings_and_arrays_enforce_both_edges() {
        let mut string = runtime(r#"{"type":"string","minLength":2,"maxLength":3}"#);
        assert!(string.accept_bytes(br#""ab""#));
        assert!(string.is_accepted());
        let mut short = runtime(r#"{"type":"string","minLength":2,"maxLength":3}"#);
        assert!(!short.accept_bytes(br#""a""#) || !short.is_accepted());

        let mut array =
            runtime(r#"{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":2}"#);
        assert!(array.accept_bytes(b"[1,2]"));
        assert!(array.is_accepted());
        let mut too_many =
            runtime(r#"{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":2}"#);
        assert!(!too_many.accept_bytes(b"[1,2,3]"));
    }

    #[test]
    fn integer_bounds_enforce_inclusive_and_exclusive_edges() {
        let schema = r#"{"type":"integer","minimum":-12,"exclusiveMaximum":35}"#;
        for accepted in ["-12", "-1", "0", "9", "34"] {
            let mut candidate = runtime(schema);
            assert!(candidate.accept_bytes(accepted.as_bytes()), "{accepted}");
            assert!(candidate.is_accepted(), "{accepted}");
        }
        for rejected in ["-13", "35", "100", "01", "-0"] {
            let mut candidate = runtime(schema);
            assert!(
                !candidate.accept_bytes(rejected.as_bytes()) || !candidate.is_accepted(),
                "{rejected}"
            );
        }
    }

    #[test]
    fn string_format_allowlist_compiles_and_enforces_mutants() {
        let fixtures = [
            ("email", "a.b@example.com", "missing-at.example.com"),
            ("date", "2026-09-03", "2026-19-03"),
            ("time", "23:59:60Z", "25:00:00Z"),
            ("date-time", "2026-09-03T12:30:00Z", "2026-09-03 12:30:00Z"),
            ("duration", "P3DT4H", "three days"),
            ("ipv4", "192.168.1.1", "999.168.1.1"),
            ("ipv6", "::ffff:192.0.2.128", "not:ipv6"),
            ("hostname", "api.example-2.com", "-bad.example"),
            ("uuid", "550e8400-e29b-41d4-a716-446655440000", "550e8400"),
            ("uri", "https://example.com/a?q=1", "http:%GG"),
            ("uri-reference", "../a?q=1", "%GG"),
            ("uri-template", "/users/{id}", "/users/{"),
            ("json-pointer", "/a~1b/0", "/bad~2escape"),
            ("relative-json-pointer", "2/owner", "02/owner"),
        ];
        for (format, valid, invalid) in fixtures {
            let schema = format!(r#"{{"type":"string","format":"{format}"}}"#);
            let mut good = runtime(&schema);
            let valid = serde_json::to_string(valid).unwrap();
            assert!(good.accept_bytes(valid.as_bytes()), "{format}: {valid}");
            assert!(good.is_accepted(), "{format}: {valid}");
            let mut bad = runtime(&schema);
            let invalid = serde_json::to_string(invalid).unwrap();
            assert!(
                !bad.accept_bytes(invalid.as_bytes()) || !bad.is_accepted(),
                "{format}: {invalid}"
            );
        }
        let error = schema_to_gbnf(&serde_json::json!({
            "type":"string", "format":"unknown-format"
        }))
        .expect_err("unknown format must fail closed");
        assert!(error.to_string().contains("/format"));
    }

    #[test]
    fn property_count_bounds_apply_to_declared_and_extra_keys() {
        let closed = r#"{
            "type":"object",
            "properties":{"a":{"type":"integer"},"b":{"type":"integer"},"c":{"type":"integer"}},
            "required":["a"],"additionalProperties":false,
            "minProperties":2,"maxProperties":2
        }"#;
        for accepted in [r#"{"a":1,"b":2}"#, r#"{"c":3,"a":1}"#] {
            let mut candidate = runtime(closed);
            assert!(candidate.accept_bytes(accepted.as_bytes()), "{accepted}");
            assert!(candidate.is_accepted(), "{accepted}");
        }
        for rejected in [r#"{"a":1}"#, r#"{"a":1,"b":2,"c":3}"#] {
            let mut candidate = runtime(closed);
            assert!(!candidate.accept_bytes(rejected.as_bytes()) || !candidate.is_accepted());
        }

        let open = r#"{"type":"object","minProperties":1,"maxProperties":2}"#;
        for accepted in [r#"{"x":1}"#, r#"{"x":1,"y":2}"#] {
            let mut candidate = runtime(open);
            assert!(candidate.accept_bytes(accepted.as_bytes()), "{accepted}");
            assert!(candidate.is_accepted(), "{accepted}");
        }
        for rejected in ["{}", r#"{"x":1,"y":2,"z":3}"#] {
            let mut candidate = runtime(open);
            assert!(!candidate.accept_bytes(rejected.as_bytes()) || !candidate.is_accepted());
        }
    }

    #[test]
    fn composition_siblings_narrow_finite_values() {
        let schema = r#"{
            "type":"string",
            "minLength":2,
            "enum":["x","ok"],
            "anyOf":[{"const":"x"},{"const":"ok"}]
        }"#;
        let mut good = runtime(schema);
        assert!(good.accept_bytes(br#""ok""#) && good.is_accepted());

        let mut short = runtime(schema);
        assert!(
            !short.accept_bytes(br#""x""#) || !short.is_accepted(),
            "enum/anyOf early returns must not discard sibling minLength"
        );

        let one_of = r#"{
            "type":"string",
            "minLength":2,
            "oneOf":[{"const":"x"},{"const":"ok"}]
        }"#;
        let mut one_of_good = runtime(one_of);
        assert!(one_of_good.accept_bytes(br#""ok""#) && one_of_good.is_accepted());
        let mut one_of_short = runtime(one_of);
        assert!(!one_of_short.accept_bytes(br#""x""#) || !one_of_short.is_accepted());

        let impossible_const = r#"{"type":"integer","const":"wrong-type"}"#;
        let mut impossible = runtime(impossible_const);
        assert!(!impossible.accept_bytes(br#""wrong-type""#) || !impossible.is_accepted());
    }

    #[test]
    fn open_object_wildcard_cannot_rematch_a_declared_key() {
        let schema = r#"{
            "type":"object",
            "properties":{"count":{"type":"integer"}}
        }"#;
        let mut declared = runtime(schema);
        assert!(declared.accept_bytes(br#"{"count":1}"#) && declared.is_accepted());

        let mut extra = runtime(schema);
        assert!(extra.accept_bytes(br#"{"note":true}"#) && extra.is_accepted());

        for mutant in [
            br#"{"count":1.5}"#.as_slice(),
            br#"{"\u0063ount":1.5}"#.as_slice(),
        ] {
            let mut runtime = runtime(schema);
            assert!(
                !runtime.accept_bytes(mutant) || !runtime.is_accepted(),
                "open-object wildcard rematched a declared key: {}",
                String::from_utf8_lossy(mutant)
            );
        }
    }

    #[test]
    fn typed_additional_properties_are_enforced_beside_declared_keys() {
        let schema = r#"{
            "type":"object",
            "properties":{"count":{"type":"integer"}},
            "additionalProperties":{"type":"string"}
        }"#;
        let mut good = runtime(schema);
        assert!(good.accept_bytes(br#"{"count":1,"note":"ok"}"#) && good.is_accepted());

        for mutant in [
            br#"{"count":1.5,"note":"ok"}"#.as_slice(),
            br#"{"count":1,"note":2}"#.as_slice(),
        ] {
            let mut runtime = runtime(schema);
            assert!(!runtime.accept_bytes(mutant) || !runtime.is_accepted());
        }
    }

    #[test]
    fn open_object_non_ascii_declared_key_fails_closed() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"café": {"type": "integer"}}
        });
        let error = schema_to_gbnf(&schema).expect_err("must not approximate key exclusion");
        assert!(error.to_string().contains("non-ASCII"));
    }
}
