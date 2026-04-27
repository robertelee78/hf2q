//! GBNF AST → text serializer.
//!
//! Counterpart to `parser.rs` (`parse(text) -> Grammar`):
//! `serialize(grammar) -> text` produces a GBNF source string that the
//! parser re-parses to a semantically equivalent `Grammar` (same rule
//! count, same `GretElement` sequences modulo synthesized rule renames
//! that the parser is free to choose at re-parse time).
//!
//! # Reference
//!
//! Modeled on llama.cpp `llama_grammar_parser::print` +
//! `print_rule` at `/opt/llama.cpp/src/llama-grammar.cpp:296-371` +
//! `:721-736`.  The C++ `print_grammar_char` (`:231-238`) is **lossy** by
//! design ("cop out of encoding UTF-8" — emits `<U+04XX>` literal
//! placeholders that are NOT valid GBNF input).  The Rust port below
//! emits proper escapes so the output round-trips through `parse`.
//!
//! # Why this exists
//!
//! Wave 2.5 audit (cfa-20260427-adr005-wave2.5) caught the
//! `combine_function_grammars` token-scanning rewriter at handlers.rs
//! corrupting GBNF negated character classes such as `[^<\\]` (the
//! Gemma `gemma4-str-char` rule).  The wave-2.6 fix per goalie
//! research §Q4 is "parse-to-AST + serialize-with-renames"; this
//! module is the "serialize" half.  See research-report.md §Q4 for
//! the full design.
//!
//! # Round-trip guarantee
//!
//! `parse(serialize(parse(s))) == parse(s)` (semantic equivalence at
//! the AST level).  Strict byte-identity is NOT guaranteed because:
//!   - Comments are not preserved (llama.cpp's print_rule does not
//!     preserve comments either; the AST does not carry them).
//!   - Whitespace is normalized.
//!   - Rule emission order follows ascending rule-id.

use std::collections::HashMap;
use std::fmt::Write;

use super::parser::{Grammar, GretElement, GretType};

/// Serialize a parsed grammar back to GBNF source text.
///
/// Round-trip property: the returned string re-parses to a `Grammar`
/// whose `rules` are byte-identical to the input grammar's `rules`
/// (same `Vec<Vec<GretElement>>` contents).  Rule names are preserved.
/// Symbol-id assignments may differ if the input had been built by
/// hand (the parser assigns ids in encounter order); for any grammar
/// produced by `parse(...)`, names → ids round-trip exactly.
///
/// Mirrors `llama_grammar_parser::print` at
/// `/opt/llama.cpp/src/llama-grammar.cpp:721-736`.
pub fn serialize(grammar: &Grammar) -> String {
    // Build id → name map (inverse of grammar.symbol_ids).
    let id_to_name: HashMap<u32, String> = grammar
        .symbol_ids
        .iter()
        .map(|(name, id)| (*id, name.clone()))
        .collect();

    let mut out = String::with_capacity(256);
    for (i, rule) in grammar.rules.iter().enumerate() {
        let rule_id = i as u32;
        // Skip empty rule slots (shouldn't happen for a parsed grammar
        // since `parse` validates non-emptiness, but defend against
        // hand-built grammars).
        if rule.is_empty() {
            continue;
        }
        write_rule(&mut out, rule_id, rule, &id_to_name);
    }
    out
}

/// Serialize a single rule.  Mirrors `print_rule` at
/// `/opt/llama.cpp/src/llama-grammar.cpp:296-371`.
///
/// The bracket-close logic (peek at next element to decide whether to
/// emit `]`) is what makes the serializer character-class-aware
/// without scanning text — the AST already encodes the boundary via
/// `GretType::CharAlt` / `CharRngUpper` lookahead.  This is the
/// missing piece in Wave 2.5's token-scanning rewriter.
fn write_rule(
    out: &mut String,
    rule_id: u32,
    rule: &[GretElement],
    id_to_name: &HashMap<u32, String>,
) {
    // Sanity: every rule must terminate with End.  We mirror llama.cpp's
    // `throw runtime_error` by skipping the rule entirely (the parser
    // enforces this invariant on `parse`, so we should never see it).
    if rule.last().map(|e| e.ty) != Some(GretType::End) {
        return;
    }

    // Emit `name ::= `.
    let name = id_to_name
        .get(&rule_id)
        .map(|s| s.as_str())
        .unwrap_or("<anonymous>");
    let _ = write!(out, "{} ::= ", name);

    // Walk elements; for each char-element close the bracket only when
    // the next element is NOT another char-class continuation (CharAlt
    // / CharRngUpper / CharAny).  CharAny here is llama.cpp's exact
    // logic — it's a degenerate "no close" since `.` is its own atom.
    //
    // Empty-alternative disambiguation:
    // An alternative whose source form is empty (zero atoms) is
    // semantically valid — it matches the empty string — but emitting
    // it as bare whitespace breaks round-trip parsing because the
    // parser's `parse_alternates` (parser.rs:267-269 / llama.cpp
    // `:443`) calls `parse_space(.., newline_ok=true)` after `|`,
    // greedily eating the newline and the next rule's name as if it
    // were an alternative continuation.  We emit `""` (which the
    // parser consumes as a zero-element literal — see parser.rs:294
    // where `pos += 1` skips the opening `"`, and the inner loop
    // exits immediately on the closing `"`) to ensure every
    // alternative occupies at least one source token.
    //
    // `alt_is_empty` tracks whether we've emitted any atom since the
    // last Alt (or rule start).  Reset on Alt; flipped false on any
    // atom emission; checked when about to emit `| ` or close out
    // the rule.
    let n = rule.len() - 1; // exclude the trailing End
    let mut i = 0;
    let mut alt_is_empty = true;
    while i < n {
        let elem = rule[i];
        match elem.ty {
            GretType::End => {
                // Cannot happen pre-(n-1) for a valid rule; defend by
                // skipping (mirrors llama.cpp throwing an exception).
                return;
            }
            GretType::Alt => {
                if alt_is_empty {
                    // Pad the just-closed (empty) alternative with
                    // `""` so the parser sees it as a non-empty
                    // literal and doesn't merge it with siblings.
                    let _ = write!(out, "\"\" ");
                }
                let _ = write!(out, "| ");
                alt_is_empty = true;
            }
            GretType::RuleRef => {
                let ref_name = id_to_name
                    .get(&elem.value)
                    .map(|s| s.as_str())
                    .unwrap_or("<unknown>");
                let _ = write!(out, "{} ", ref_name);
                alt_is_empty = false;
            }
            GretType::Char => {
                let _ = write!(out, "[");
                write_char(out, elem.value, /* in_class = */ true);
                alt_is_empty = false;
            }
            GretType::CharNot => {
                let _ = write!(out, "[^");
                write_char(out, elem.value, /* in_class = */ true);
                alt_is_empty = false;
            }
            GretType::CharRngUpper => {
                // The parser enforces that this only follows a char
                // element; mirror llama.cpp's invariant assert.
                // CharRngUpper is a continuation of an existing class,
                // so the alt was already non-empty when we got here.
                let _ = write!(out, "-");
                write_char(out, elem.value, /* in_class = */ true);
            }
            GretType::CharAlt => {
                // Same continuation reasoning as CharRngUpper.
                write_char(out, elem.value, /* in_class = */ true);
            }
            GretType::CharAny => {
                let _ = write!(out, ".");
                alt_is_empty = false;
            }
        }

        // Bracket-close lookahead — exact port of llama.cpp lines
        // 359-368.  Only char-elements (Char / CharNot / CharAlt /
        // CharRngUpper / CharAny) need the trailing `] `.  We close
        // when the NEXT element is NOT one of {CharAlt, CharRngUpper,
        // CharAny} — which means the current char-class run ends here.
        if elem.ty.is_char_element() {
            // Note: rule[i+1] always exists because `i < n` and there's
            // always the End element at index n.
            let next_ty = rule[i + 1].ty;
            let inside_class_continues = matches!(
                next_ty,
                GretType::CharAlt | GretType::CharRngUpper | GretType::CharAny
            );
            if !inside_class_continues {
                // CharAny does not need a closing bracket — it was
                // emitted as `.`.  Suppress the `] ` for that case;
                // the test below short-circuits via the outer match.
                if elem.ty != GretType::CharAny {
                    let _ = write!(out, "] ");
                }
            }
        }

        i += 1;
    }
    // If the trailing alternative is empty (e.g. a recursive
    // subrule whose body is `RuleRef Alt End`), pad with `""` so
    // the parser doesn't bleed the next rule's name into this
    // one's tail.  See the "Empty-alternative disambiguation"
    // comment above for the failure mode this prevents.
    if alt_is_empty {
        let _ = write!(out, "\"\" ");
    }
    out.push('\n');
}

/// Emit a single Unicode scalar `cp` as GBNF source.
///
/// `in_class = true` adds escapes for the four characters that are
/// structural inside a character class (`[`, `]`, `\\`, `-`).  The
/// `-` escape isn't strictly necessary at every position (llama.cpp's
/// parser only treats `-` as range-introducer between two chars), but
/// always escaping it is the conservative choice that keeps the
/// serializer position-agnostic — every char-class element can be
/// emitted independently.
///
/// Outside a class (literal-string context — currently unused since
/// the parser stores literal strings as Char-element sequences inside
/// brackets, never as `"..."`), the same escape table applies minus
/// `-` and `]`.  We default `in_class = true` because the AST
/// representation puts every Char element through a `[...]` bracket
/// (literal strings degenerate to single-char `[c]` brackets).
///
/// All non-ASCII (`> 0x7F`) is emitted as `\uHHHH` (BMP) or
/// `\UHHHHHHHH` (supplementary plane).  This is round-trip safe
/// because the parser at `parse_char` decodes both forms into the
/// same `u32` code point.
fn write_char(out: &mut String, cp: u32, in_class: bool) {
    match cp {
        // Standard escape sequences with single-char shorthands.
        0x09 => out.push_str(r"\t"),
        0x0A => out.push_str(r"\n"),
        0x0D => out.push_str(r"\r"),
        // Backslash itself is always escaped.
        0x5C => out.push_str(r"\\"),
        // Inside a class: `]` ends the class, must be escaped.
        // Always escape `[` too for symmetry — the parser accepts
        // `\[` everywhere.
        0x5B if in_class => out.push_str(r"\["),
        0x5D if in_class => out.push_str(r"\]"),
        // `-` could be parsed as a range introducer; always escape
        // inside a class to make every char-element position-safe.
        0x2D if in_class => out.push_str(r"\x2D"),
        // Quote: `"` ends a quoted literal.  Inside a class it's
        // structurally fine but escape for consistency in case the
        // grammar is later moved to a literal context.
        0x22 => {
            // Escape only when not in class (literal context).  Inside
            // a class, `"` is just a regular char.
            if in_class {
                out.push('"');
            } else {
                out.push_str(r#"\""#);
            }
        }
        // Caret has special meaning ONLY immediately after `[` (the
        // negation marker). The serializer never emits Char as the
        // first element of a class — that role belongs to CharNot.
        // Still, escape `^` as a defensive measure if it appears
        // mid-class (parser accepts unescaped `^` mid-class but emit
        // hex for clarity).
        0x5E if in_class => out.push_str(r"\x5E"),
        // ASCII printable: emit verbatim.
        0x20..=0x7E => out.push(cp as u8 as char),
        // Control chars and other non-printable ASCII.
        0x00..=0x1F | 0x7F => {
            let _ = write!(out, r"\x{:02X}", cp);
        }
        // BMP: `\uHHHH`.
        0x80..=0xFFFF => {
            let _ = write!(out, r"\u{:04X}", cp);
        }
        // Supplementary planes: `\UHHHHHHHH`.
        _ => {
            let _ = write!(out, r"\U{:08X}", cp);
        }
    }
}

// ---------------------------------------------------------------------------
// AST manipulation primitives used by the AST-based combiner.
// ---------------------------------------------------------------------------

/// Apply `f` to every rule name in `grammar`, producing a NEW grammar
/// with renamed `symbol_ids` and references rewritten to match.  The
/// `rules` Vec layout (i.e. rule id → element sequence) is preserved
/// untouched — only the name → id map is rewritten, which makes
/// reference rewriting a no-op (RuleRef stores the rule-id, not the
/// name).
///
/// Pure AST operation: NO string scanning, NO body rewriting.  This
/// is the architectural inversion that fixes the wave-2.5 audit.
pub fn rename_rules<F>(grammar: &Grammar, mut f: F) -> Grammar
where
    F: FnMut(&str) -> String,
{
    let mut new_symbol_ids: HashMap<String, u32> = HashMap::with_capacity(grammar.symbol_ids.len());
    for (name, id) in &grammar.symbol_ids {
        let new_name = f(name);
        new_symbol_ids.insert(new_name, *id);
    }
    Grammar {
        rules: grammar.rules.clone(),
        symbol_ids: new_symbol_ids,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::parser::{parse, GretElement, GretType};

    /// Semantic AST equality: every rule NAME present in `a` must
    /// also be present in `b`, and the element-sequence under each
    /// shared name must match modulo rule-id renumbering.
    ///
    /// Why name-based and not id-based: the parser assigns rule ids
    /// in encounter order (parser.rs:182-190 `get_or_create_symbol`).
    /// `parse(serialize(g))` may visit rule names in a different
    /// order than `g`'s original parse — e.g. if `root` references
    /// `root_2` which references `root_1`, the second parse will
    /// see ids in (root, root_2, root_1) order, while the original
    /// `"a"{0,2}` parse produced (root, root_1, root_2).
    /// Rule bodies stay identical modulo this renumbering.
    ///
    /// This function performs the renumbering check: for every pair
    /// of rule references with matching NAMES on both sides, the
    /// element types and `RuleRef` target NAMES (not raw ids) must
    /// match.
    fn ast_eq(a: &Grammar, b: &Grammar) -> bool {
        if a.rules.len() != b.rules.len() {
            return false;
        }
        if a.symbol_ids.len() != b.symbol_ids.len() {
            return false;
        }
        // Both grammars must define the same set of rule names.
        for name in a.symbol_ids.keys() {
            if !b.symbol_ids.contains_key(name) {
                return false;
            }
        }
        // Build id → name maps for both.
        let a_id_to_name: std::collections::HashMap<u32, &str> = a
            .symbol_ids
            .iter()
            .map(|(n, id)| (*id, n.as_str()))
            .collect();
        let b_id_to_name: std::collections::HashMap<u32, &str> = b
            .symbol_ids
            .iter()
            .map(|(n, id)| (*id, n.as_str()))
            .collect();
        // For each rule by NAME, compare element sequences.
        for (name, &a_id) in &a.symbol_ids {
            let b_id = b.symbol_ids[name];
            let ra = &a.rules[a_id as usize];
            let rb = &b.rules[b_id as usize];
            if ra.len() != rb.len() {
                return false;
            }
            for (ea, eb) in ra.iter().zip(rb.iter()) {
                if ea.ty != eb.ty {
                    return false;
                }
                match ea.ty {
                    GretType::RuleRef => {
                        // Compare via referenced NAMES, not ids.
                        let na = a_id_to_name.get(&ea.value).copied().unwrap_or("?");
                        let nb = b_id_to_name.get(&eb.value).copied().unwrap_or("?");
                        if na != nb {
                            return false;
                        }
                    }
                    _ => {
                        if ea.value != eb.value {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    fn roundtrip(src: &str) -> Grammar {
        let g1 = parse(src).expect("first parse");
        let serialized = serialize(&g1);
        let g2 = parse(&serialized).unwrap_or_else(|e| {
            panic!(
                "round-trip serialize → parse failed: {}\nserialized text was:\n{}",
                e, serialized
            )
        });
        assert!(
            ast_eq(&g1, &g2),
            "AST identity broken on round-trip\noriginal:\n{:?}\nserialized:\n{}\nreparsed:\n{:?}",
            g1, serialized, g2
        );
        g2
    }

    #[test]
    fn round_trip_simple_literal() {
        roundtrip("root ::= \"hello\"\n");
    }

    // -------------------------------------------------------------------
    // Worker-spec round-trip tests (wave2.6 W-γ5a Q4-A acceptance bar).
    // These mirror the names called out in the worker prompt; the
    // shape-equivalent tests above already exercise the same coverage,
    // but the explicit worker-named variants keep the audit trail
    // grep-able from the cfa-20260427 prompt.
    // -------------------------------------------------------------------

    /// Spec test #1: minimal grammar to anchor the scaffold.
    #[test]
    fn parser_round_trip_simple_grammar() {
        roundtrip("root ::= \"hi\"\n");
    }

    /// Spec test #2: the wave-2.5 audit's failing case literally.
    /// Negated char class with backslash → must NOT be corrupted to
    /// `[<\\]` (positive class) by the round-trip.
    #[test]
    fn parser_round_trip_grammar_with_negated_char_class() {
        roundtrip("root ::= [^<\\\\]\n");
    }

    /// Spec test #3: quoted-literal escape coverage. `<|tool_call>`
    /// is the canonical Gemma/Qwen tool-call open token; this is the
    /// shape `combine_function_grammars` will see in production.
    #[test]
    fn parser_round_trip_grammar_with_quoted_literal_escapes() {
        // Embeds the literal `<|tool_call>` as a GBNF quoted string.
        roundtrip("root ::= \"<|tool_call>\"\n");
    }

    /// Spec test #4: alternations + groups, the structural shape from
    /// the worker prompt (`root := A | B (C D)`).  Group expansion
    /// goes through synthesized `_`-named subrules — exercises the
    /// parser/emitter `_`-in-name agreement (parser.rs:600 fix).
    #[test]
    fn parser_round_trip_grammar_with_alternations_and_groups() {
        roundtrip("root ::= a | b ( c d )\na ::= \"a\"\nb ::= \"b\"\nc ::= \"c\"\nd ::= \"d\"\n");
    }

    #[test]
    fn round_trip_alternation() {
        roundtrip("root ::= \"a\" | \"b\" | \"c\"\n");
    }

    #[test]
    fn round_trip_char_class_range() {
        roundtrip("root ::= [a-z]\n");
    }

    #[test]
    fn round_trip_negated_char_class() {
        // The wave-2.5 audit's failing case: negated char class with
        // backslash inside.  Mirrors gemma4-str-char from
        // src/serve/api/registry.rs:1051.
        roundtrip("root ::= [^<\\\\]\n");
    }

    #[test]
    fn round_trip_negated_with_quote_and_backslash() {
        // The canonical JSON `string` rule's interior char.  Same
        // shape as `[^"\\]` in /opt/llama.cpp/grammars/json.gbnf.
        roundtrip("root ::= [^\"\\\\]\n");
    }

    #[test]
    fn round_trip_char_class_multi_alt() {
        roundtrip("root ::= [abc]\n");
    }

    #[test]
    fn round_trip_char_class_range_plus_alt() {
        // [a-zA-Z0-9] — three ranges in one class.
        roundtrip("root ::= [a-zA-Z0-9]\n");
    }

    #[test]
    fn round_trip_quoted_literal_with_escapes() {
        // Backslash + double quote + tab + newline in a literal.
        // Round-trips because every char goes through the AST as
        // a Char element with its decoded code point.
        roundtrip("root ::= \"a\\\\b\\\"c\\nd\\te\"\n");
    }

    #[test]
    fn round_trip_rule_reference() {
        roundtrip("root ::= ws \"x\" ws\nws ::= \" \"?\n");
    }

    #[test]
    fn round_trip_repetition_star() {
        roundtrip("root ::= \"a\"*\n");
    }

    #[test]
    fn round_trip_repetition_plus() {
        roundtrip("root ::= \"a\"+\n");
    }

    #[test]
    fn round_trip_grouping() {
        roundtrip("root ::= ( \"x\" \"y\" ) | \"z\"\n");
    }

    #[test]
    fn round_trip_any_char_dot() {
        roundtrip("root ::= .\n");
    }

    #[test]
    fn round_trip_utf8_literal() {
        // Greek alpha (U+03B1).
        roundtrip("root ::= \"α\"\n");
    }

    #[test]
    fn round_trip_supplementary_plane() {
        // U+1F600 (😀) emits as \U0001F600 — verifies the supplementary
        // plane branch.
        roundtrip("root ::= \"😀\"\n");
    }

    #[test]
    fn round_trip_json_grammar_fixture() {
        // The canonical llama.cpp json grammar.  Stress-tests every
        // GBNF feature we serialize: nested groups, recursion,
        // quoted-literal escapes, negated char class with
        // backslash, ranges, comments (not preserved).
        let src = std::fs::read_to_string("/opt/llama.cpp/grammars/json.gbnf")
            .expect("json.gbnf fixture present");
        roundtrip(&src);
    }

    #[test]
    fn round_trip_arithmetic_grammar_fixture() {
        let src = std::fs::read_to_string("/opt/llama.cpp/grammars/arithmetic.gbnf")
            .expect("arithmetic.gbnf fixture present");
        roundtrip(&src);
    }

    #[test]
    fn round_trip_list_grammar_fixture() {
        let src = std::fs::read_to_string("/opt/llama.cpp/grammars/list.gbnf")
            .expect("list.gbnf fixture present");
        roundtrip(&src);
    }

    #[test]
    fn negated_char_class_semantics_preserved_after_round_trip() {
        // The audit-driving test: build a grammar with `[^<\\]`, run
        // it through the runtime, and confirm that a STRING with `<`
        // is REJECTED (not accepted, as it would be if `[^<\\]` got
        // corrupted to `[<\\]`).
        use super::super::parser::parse;
        use super::super::sampler::GrammarRuntime;

        let src = "root ::= [^<\\\\]+\n";
        let g1 = parse(src).expect("first parse");
        let serialized = serialize(&g1);
        let g2 = parse(&serialized).expect("re-parse");

        // Run the re-parsed grammar through the runtime against `<`.
        let rid = g2.rule_id("root").expect("root rule exists");
        let mut rt = GrammarRuntime::new(g2, rid).expect("runtime init");
        let alive = rt.accept_bytes(b"<");
        assert!(
            !alive,
            "negated char class `[^<\\\\]` was corrupted during round-trip: \
             a `<` byte was ACCEPTED, but the grammar should REJECT it"
        );
    }

    #[test]
    fn rename_rules_preserves_rule_bodies() {
        let src = "root ::= ws \"x\"\nws ::= \" \"?\n";
        let g = parse(src).expect("parse");
        let renamed = rename_rules(&g, |n| format!("fn-7-{}", n));
        // Rule bodies are byte-identical.
        assert_eq!(renamed.rules, g.rules);
        // Names are renamed.
        assert!(renamed.symbol_ids.contains_key("fn-7-root"));
        assert!(renamed.symbol_ids.contains_key("fn-7-ws"));
        // Old names are gone.
        assert!(!renamed.symbol_ids.contains_key("root"));
        assert!(!renamed.symbol_ids.contains_key("ws"));
        // Rule-ids preserved.
        assert_eq!(renamed.rule_id("fn-7-root"), g.rule_id("root"));
        assert_eq!(renamed.rule_id("fn-7-ws"), g.rule_id("ws"));
    }

    #[test]
    fn rename_rules_round_trip_serialize_parse() {
        let src = "root ::= ws \"x\"\nws ::= \" \"?\n";
        let g = parse(src).expect("parse");
        let renamed = rename_rules(&g, |n| format!("fn-3-{}", n));
        let text = serialize(&renamed);
        let reparsed = parse(&text).expect("re-parse renamed");
        // The reparsed grammar must contain the renamed names.
        assert!(reparsed.symbol_ids.contains_key("fn-3-root"));
        assert!(reparsed.symbol_ids.contains_key("fn-3-ws"));
        // And the rule bodies must match what `rename_rules` produced.
        // (Re-parse may reassign rule-ids, so compare via name lookup.)
        for name in renamed.symbol_ids.keys() {
            let rid_a = renamed.rule_id(name).expect("renamed has name");
            let rid_b = reparsed.rule_id(name).expect("reparsed has name");
            // Element sequences must be identical.
            assert_eq!(
                renamed.rules[rid_a as usize], reparsed.rules[rid_b as usize],
                "rule body for {} differs after round trip",
                name
            );
        }
    }

    #[test]
    fn empty_grammar_serializes_to_empty_string() {
        let g = Grammar {
            rules: Vec::new(),
            symbol_ids: HashMap::new(),
        };
        assert_eq!(serialize(&g), "");
    }

    #[test]
    fn write_char_emits_hex_for_control() {
        let mut s = String::new();
        write_char(&mut s, 0x07, true); // BEL
        assert_eq!(s, r"\x07");
    }

    #[test]
    fn write_char_emits_unicode_escape_for_bmp() {
        let mut s = String::new();
        write_char(&mut s, 0x03B1, true); // Greek alpha
        // BMP code point above ASCII → emitted as `\uHHHH` so the
        // serialized output is round-trip safe through `parse_char`'s
        // `\u` branch (parser.rs:698).
        assert_eq!(s, r"\u03B1");
    }

    #[test]
    fn write_char_emits_long_unicode_for_supplementary() {
        let mut s = String::new();
        write_char(&mut s, 0x1F600, true); // 😀
        assert_eq!(s, r"\U0001F600");
    }

    #[test]
    fn write_char_escapes_backslash() {
        let mut s = String::new();
        write_char(&mut s, 0x5C, true);
        assert_eq!(s, r"\\");
    }

    #[test]
    fn write_char_escapes_close_bracket_in_class() {
        let mut s = String::new();
        write_char(&mut s, 0x5D, true);
        assert_eq!(s, r"\]");
    }

    /// Defensive: make sure a synthesized rule (e.g. from `?` expansion)
    /// round-trips. The parser generates rules like `root_1 ::= |  | `
    /// for `"a"?` which contain only Alt+End structures.
    #[test]
    fn round_trip_zero_or_one_repetition() {
        roundtrip("root ::= \"a\"?\n");
    }

    /// Defensive: brace-form repetition expands into multiple
    /// synthesized subrules; round-trip exercises serialization of
    /// chained RuleRef + Alt + End.
    #[test]
    fn round_trip_brace_min_max_repetition() {
        roundtrip("root ::= \"a\"{0,2}\n");
    }

    /// Real-production fixture: the gemma4-str-char rule body literally
    /// from src/serve/api/registry.rs:1051. This is the audit-failing
    /// case lifted into a unit test on the AST serialize.
    #[test]
    fn round_trip_gemma4_str_char_rule() {
        let src = "gemma4-str-char ::= [^<\\\\] | [\\\\] [^\\x00-\\x1F]\n\
                   root ::= gemma4-str-char\n";
        let g1 = parse(src).expect("first parse");
        let serialized = serialize(&g1);
        let g2 = parse(&serialized).expect("re-parse");
        // Semantic equivalence: the gemma4-str-char rule body matches.
        let id1 = g1.rule_id("gemma4-str-char").expect("rule exists");
        let id2 = g2.rule_id("gemma4-str-char").expect("rule exists post-roundtrip");
        assert_eq!(
            g1.rules[id1 as usize], g2.rules[id2 as usize],
            "gemma4-str-char rule body differs after round trip"
        );
    }

    /// Negated-char-class with a trailing range: the parse element
    /// sequence is CharNot(0x00) + CharRngUpper(0x1F).  Verifies the
    /// bracket-close lookahead handles the boundary correctly.
    #[test]
    fn round_trip_negated_class_with_range() {
        roundtrip("root ::= [^\\x00-\\x1F]\n");
    }

    /// Two char classes with a literal between them — exercises the
    /// CharAlt → CharRngUpper transition in adjacent classes.
    #[test]
    fn round_trip_adjacent_char_classes() {
        roundtrip("root ::= [a-z] \"x\" [A-Z]\n");
    }

    /// A char class containing a literal close-bracket via escape:
    /// `[\\]]` would be `[` followed by escaped-`]` followed by `]` —
    /// the escape is consumed in the AST, so on round-trip we must
    /// emit `\]` again.  This is the core escape-discipline test.
    #[test]
    fn round_trip_class_with_escaped_close_bracket() {
        let src = "root ::= [\\]a]\n";
        let g1 = parse(src).expect("first parse");
        let serialized = serialize(&g1);
        let g2 = parse(&serialized).expect("re-parse");
        assert!(
            ast_eq(&g1, &g2),
            "escaped close-bracket in class did not round-trip:\n  serialized: {}",
            serialized
        );
    }

    /// Used as a vector for the unused-import warnings — keeps
    /// `GretElement` / `GretType` imports active in the test module
    /// even when the body uses only one of them.
    #[allow(dead_code)]
    fn _silence_unused_import_warnings() {
        let _ = GretElement::new(GretType::End, 0);
    }
}
