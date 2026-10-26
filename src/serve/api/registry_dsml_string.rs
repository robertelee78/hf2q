//! DSML raw values may contain code angle brackets, but never their close tag.
//! Exact bounded repetition plus a private terminal-scoped runtime delimiter guard.
#[cfg(test)]
use super::gbnf_literal;

pub(super) const CLOSE: &str = "</｜DSML｜parameter>";
/// Character-count composition stays in the established exact repetition
/// compiler. The private runtime guard excludes only COMPLETE raw sentinels
/// consumed by these reserved DATA terminals; structural closing stays legal.
pub(super) fn emit(
    rules: &mut Vec<(String, String)>,
    _prefix: &str,
    min: u64,
    max: Option<u64>,
) -> Result<String, String> {
    validate_bounds(min, max)?;
    use crate::serve::api::grammar::dsml_lexical::{END_BODY, END_NAME, RAW_BODY, RAW_NAME};
    insert(rules, RAW_NAME, RAW_BODY)?;
    insert(rules, END_NAME, END_BODY)?;
    Ok(format!(
        "{} {END_NAME}",
        super::bounded_repeat(RAW_NAME, min, max, rules)
    ))
}
pub(super) fn emit_json(
    rules: &mut Vec<(String, String)>,
    _prefix: &str,
    min: u64,
    max: Option<u64>,
) -> Result<String, String> {
    validate_bounds(min, max)?;
    use crate::serve::api::grammar::dsml_lexical::{JSON_BODY, JSON_NAME};
    insert(rules, JSON_NAME, JSON_BODY)?;
    insert(
        rules,
        "dsml-data-json-char",
        r#"dsml-data-json-raw | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})"#,
    )?;
    Ok(super::bounded_repeat(
        "dsml-data-json-char",
        min,
        max,
        rules,
    ))
}
fn validate_bounds(min: u64, max: Option<u64>) -> Result<(), String> {
    let cap = crate::serve::api::grammar::repetition::MAX_SCHEMA_REPETITION_BOUND;
    if max.is_some_and(|n| n < min) {
        return Err("DeepSeek minLength exceeds maxLength".into());
    }
    if min > cap || max.is_some_and(|n| n > cap) {
        return Err("DeepSeek length exceeds existing schema resource limit".into());
    }
    Ok(())
}
fn insert(rules: &mut Vec<(String, String)>, name: &str, body: &str) -> Result<(), String> {
    if let Some((_, existing)) = rules.iter().find(|(n, _)| n == name) {
        if existing != body {
            return Err("DSML reserved data rule collision".into());
        }
    } else {
        rules.push((name.into(), body.into()));
    }
    Ok(())
}

/// Sufficient lexical proof for an unchanged compiled pattern/format body.
/// If its complete terminal alphabet omits any sentinel scalar, no word in
/// that language can contain the sentinel. We inspect even unreachable rules:
/// over-approximation can refuse a safe language, never bless an unsafe one.
pub(super) fn alphabet_excludes_close(body: &str) -> bool {
    use crate::serve::api::grammar::parser::{parse, GretType};
    use crate::serve::api::grammar::sampler::{match_char, Pos};
    let Ok(grammar) = parse(&format!("root ::= {body}\n")) else {
        return false;
    };
    let marker: Vec<u32> = CLOSE.chars().map(u32::from).collect();
    let mut emitted = vec![false; marker.len()];
    for (rule_id, rule) in grammar.rules.iter().enumerate() {
        for (offset, element) in rule.iter().enumerate() {
            match element.ty {
                GretType::Char | GretType::CharNot | GretType::CharAny => {
                    let pos = Pos::new(rule_id as u32, offset as u32);
                    for (index, &ch) in marker.iter().enumerate() {
                        emitted[index] |= match_char(&grammar, pos, ch).0;
                    }
                }
                GretType::Token
                | GretType::TokenNot
                | GretType::TokenAny
                | GretType::TokenNotSet
                | GretType::TokenSetMember => return false,
                GretType::End
                | GretType::Alt
                | GretType::RuleRef
                | GretType::CharRngUpper
                | GretType::CharAlt => {}
            }
        }
    }
    emitted.contains(&false)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dsml_alphabet_proof_uses_actual_ranges_and_negation() {
        assert!(alphabet_excludes_close("[a-z0-9]+"));
        assert!(alphabet_excludes_close("[^<]*"));
        assert!(!alphabet_excludes_close("[^z]*"));
        assert!(!alphabet_excludes_close("."));
        assert!(!alphabet_excludes_close(&gbnf_literal(CLOSE)));
        assert!(!alphabet_excludes_close(r"[\uFF5C] | [</>a-zA-Z]"));
        assert!(!alphabet_excludes_close("not-a-defined-rule"));
    }
}
