//! DSML raw values may contain code angle brackets, but never their close tag.
//! Prefix-state DFA composed with an optional Unicode-character length counter.
use super::gbnf_literal;

pub(super) const CLOSE: &str = "</｜DSML｜parameter>";
// Same output limits as grammar/parser.rs, not a new schema/admission ceiling.
const MAX_BYTES: usize = 4 * 1024 * 1024;
const MAX_RULES: usize = 262_144;

fn next_prefix(marker: &[char], state: usize, ch: char) -> usize {
    let mut prefix = marker[..state].to_vec();
    prefix.push(ch);
    (0..=marker.len())
        .rev()
        .find(|&n| n <= prefix.len() && prefix[prefix.len() - n..] == marker[..n])
        .unwrap()
}

fn state_name(prefix: &str, count: usize, matched: usize, escape: bool) -> String {
    format!("{prefix}-{count}-{matched}-{}", usize::from(escape))
}

/// No complete delimiter can be consumed as data, even across a backslash pair.
/// Every transition consumes exactly ONE Unicode scalar, preserving length bounds.
/// A trailing backslash remains invalid, as on the previous DSML string surface.
pub(super) fn emit(
    rules: &mut Vec<(String, String)>,
    prefix: &str,
    min: u64,
    max: Option<u64>,
) -> Result<String, String> {
    emit_inner(rules, prefix, min, max, false)
}

pub(super) fn emit_json(
    rules: &mut Vec<(String, String)>,
    prefix: &str,
    min: u64,
    max: Option<u64>,
) -> Result<String, String> {
    emit_inner(rules, prefix, min, max, true)
}

fn emit_inner(
    rules: &mut Vec<(String, String)>,
    prefix: &str,
    min: u64,
    max: Option<u64>,
    json: bool,
) -> Result<String, String> {
    if max.is_some_and(|n| n < min) {
        return Err("DeepSeek minLength exceeds maxLength".into());
    }
    let marker: Vec<_> = CLOSE.chars().collect();
    let limit = usize::try_from(max.unwrap_or(min))
        .map_err(|_| "DeepSeek string length exceeds grammar resource limits")?;
    let states = limit
        .checked_add(1)
        .and_then(|n| n.checked_mul(marker.len()))
        .and_then(|n| n.checked_mul(if json { 1 } else { 2 }))
        .ok_or("DeepSeek string state count overflow")?;
    if states
        .checked_add(rules.len())
        .is_none_or(|n| n > MAX_RULES)
    {
        return Err(
            "DeepSeek delimiter/length grammar exceeds existing rule resource limit".into(),
        );
    }
    let mut alphabet = marker.clone();
    alphabet.push('\\');
    alphabet.sort_unstable();
    alphabet.dedup();
    let excluded = alphabet
        .iter()
        .map(|ch| match ch {
            '\\' => "\\\\".to_string(),
            _ => ch.to_string(),
        })
        .collect::<String>();
    let mut bytes = rules
        .iter()
        .try_fold(0usize, |n, (name, body)| {
            n.checked_add(name.len() + body.len() + 6)
        })
        .ok_or("DeepSeek grammar byte count overflow")?;
    for count in 0..=limit {
        for matched in 0..marker.len() {
            for escape in [false, true].into_iter().filter(|&escape| !json || !escape) {
                let mut alternatives = Vec::new();
                if !escape && count as u64 >= min {
                    alternatives.push("\"\"".to_string());
                }
                if max.is_none_or(|n| (count as u64) < n) {
                    let next_count = (count + 1).min(limit);
                    for &ch in &alphabet {
                        if json && ch == '\\' {
                            continue;
                        }
                        let next = next_prefix(&marker, matched, ch);
                        if next == marker.len() {
                            continue;
                        }
                        let next_escape = !escape && ch == '\\';
                        alternatives.push(format!(
                            "{} {}",
                            gbnf_literal(&ch.to_string()),
                            state_name(prefix, next_count, next, next_escape)
                        ));
                    }
                    // Every other scalar resets the delimiter prefix. Escaped
                    // control characters were forbidden by the old surface.
                    let controls = if json {
                        r#""\x00-\x1F"#
                    } else if escape {
                        r#"\x00-\x1F"#
                    } else {
                        ""
                    };
                    alternatives.push(format!(
                        "[^{excluded}{controls}] {}",
                        state_name(prefix, next_count, 0, false)
                    ));
                    if json {
                        alternatives.push(format!(
                            r#"[\\] (["\\/bfnrt] | "u" [0-9a-fA-F]{{4}}) {}"#,
                            state_name(prefix, next_count, 0, false)
                        ));
                    }
                }
                let body = if alternatives.is_empty() {
                    "[^\\U00000000-\\U0010FFFF]".to_string()
                } else {
                    alternatives.join(" | ")
                };
                let name = state_name(prefix, count, matched, escape);
                bytes = bytes
                    .checked_add(name.len() + body.len() + 6)
                    .ok_or("DeepSeek grammar byte count overflow")?;
                if bytes > MAX_BYTES {
                    return Err(
                        "DeepSeek delimiter/length grammar exceeds existing byte resource limit"
                            .into(),
                    );
                }
                rules.push((name, body));
            }
        }
    }
    Ok(state_name(prefix, 0, 0, false))
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
