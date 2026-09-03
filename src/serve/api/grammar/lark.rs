//! Narrow Lark-to-GBNF conversion matching vLLM's structured-output shim.
//!
//! This is deliberately not a Lark parser. vLLM detects Lark solely by the
//! absence of `::=` in non-comment lines, then rewrites simple `rule: body`
//! productions into EBNF. Keeping that small, source-bound contract prevents
//! accepting a broad Lark dialect that the GBNF runtime cannot enforce.

use std::collections::BTreeSet;

/// Return true when vLLM would send this grammar through its narrow Lark
/// converter rather than directly to XGrammar's EBNF parser.
pub fn is_likely_lark(source: &str) -> bool {
    if source.is_empty() {
        return false;
    }
    source
        .lines()
        .map(clean_line)
        .filter(|line| !line.is_empty())
        .all(|line| !line.contains("::="))
}

/// Convert the subset of Lark accepted by vLLM's compatibility converter.
pub fn to_ebnf(source: &str) -> Result<String, LarkError> {
    if source.trim().is_empty() {
        return Err(LarkError("Grammar string cannot be empty".into()));
    }

    let lines = source.lines().map(clean_line).collect::<Vec<_>>();
    let mut defined = BTreeSet::new();
    let mut first_rule = None;
    for (offset, line) in lines.iter().enumerate() {
        if line.is_empty() || line.starts_with('|') {
            continue;
        }
        if let Some((name, _)) = line.split_once(':') {
            let name = name.trim().trim_matches('?').to_owned();
            if name.is_empty() {
                return Err(line_error(
                    offset,
                    "Invalid rule format. Expected 'rule_name: definition'",
                ));
            }
            if first_rule.is_none() || name == "start" {
                first_rule = Some(name.clone());
            }
            defined.insert(name);
        }
    }
    let Some(first_rule) = first_rule else {
        return Err(LarkError("No valid rules found in grammar".into()));
    };

    let mut referenced = BTreeSet::new();
    let mut output = vec![format!("root ::= {first_rule}")];
    let mut current: Option<String> = None;
    let mut alternatives = Vec::new();

    for (offset, line) in lines.iter().enumerate() {
        if line.is_empty() {
            continue;
        }
        if !line.starts_with('|') && line.contains(':') {
            if let Some(name) = current.take() {
                output.push(format!("{name} ::= {}", alternatives.join(" | ")));
            }
            let (name, definition) = line.split_once(':').expect("checked contains colon");
            let name = name.trim().trim_matches('?').to_owned();
            validate_quotes(definition, &format!("rule '{name}'"), offset)?;
            let definition = definition.replace('\'', "\"");
            referenced.extend(extract_references(&definition));
            current = Some(name);
            alternatives = vec![definition.trim().to_owned()];
        } else if let Some(rest) = line.strip_prefix('|') {
            let Some(name) = current.as_ref() else {
                return Err(line_error(
                    offset,
                    "Alternative '|' without a preceding rule definition",
                ));
            };
            let definition = rest.trim();
            validate_quotes(
                definition,
                &format!("alternative for rule '{name}'"),
                offset,
            )?;
            let definition = definition.replace('\'', "\"");
            referenced.extend(extract_references(&definition));
            alternatives.push(definition);
        }
    }
    if let Some(name) = current {
        output.push(format!("{name} ::= {}", alternatives.join(" | ")));
    }

    let undefined = referenced
        .difference(&defined)
        .filter(|name| name.as_str() != "root")
        .cloned()
        .collect::<Vec<_>>();
    if !undefined.is_empty() {
        return Err(LarkError(format!(
            "Referenced rules are not defined: {}",
            undefined.join(", ")
        )));
    }
    Ok(output.join("\n"))
}

/// Normalize a vLLM `structured_outputs.grammar` payload to GBNF source.
///
/// EBNF is returned unchanged; only grammar text classified as Lark by
/// [`is_likely_lark`] is rewritten. Parsing remains the caller's separate
/// responsibility so request validation can preserve its own error context.
pub fn normalize_for_gbnf(source: &str) -> Result<String, LarkError> {
    if is_likely_lark(source) {
        to_ebnf(source)
    } else {
        Ok(source.to_owned())
    }
}

fn clean_line(line: &str) -> String {
    let hash = line.find('#');
    let slash = line.find("//");
    let end = match (hash, slash) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => line.len(),
    };
    line[..end].trim().to_owned()
}

fn validate_quotes(text: &str, context: &str, offset: usize) -> Result<(), LarkError> {
    if text.matches('\'').count() % 2 != 0 || text.matches('"').count() % 2 != 0 {
        return Err(line_error(
            offset,
            &format!("Mismatched quotes in {context}"),
        ));
    }
    Ok(())
}

fn extract_references(text: &str) -> BTreeSet<String> {
    let mut scrubbed = String::with_capacity(text.len());
    let mut quoted = false;
    for ch in text.chars() {
        if ch == '"' {
            quoted = !quoted;
            scrubbed.push(' ');
        } else if quoted || "+*?()|[]{}".contains(ch) {
            scrubbed.push(' ');
        } else {
            scrubbed.push(ch);
        }
    }
    scrubbed
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .filter(|word| {
            word.chars()
                .next()
                .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        })
        .map(str::to_owned)
        .collect()
}

fn line_error(offset: usize, message: &str) -> LarkError {
    LarkError(format!("Error on line {}: {message}", offset + 1))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LarkError(pub String);

impl std::fmt::Display for LarkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LarkError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_matches_vllm_heuristic() {
        assert!(is_likely_lark("rule: 'abc'"));
        assert!(!is_likely_lark("rule ::= 'abc'"));
        assert!(is_likely_lark("# only a comment\nrule: \"ok\" // tail"));
        assert!(!is_likely_lark(""));
    }

    #[test]
    fn converter_selects_start_and_rewrites_alternatives() {
        assert_eq!(
            to_ebnf("other: 'no'\nstart: 'yes'\n  | other").unwrap(),
            "root ::= start\nother ::= \"no\"\nstart ::= \"yes\" | other"
        );
    }

    #[test]
    fn converter_rejects_unresolved_and_mismatched_quotes() {
        assert!(to_ebnf("start: missing").unwrap_err().0.contains("missing"));
        assert!(to_ebnf("start: 'missing")
            .unwrap_err()
            .0
            .contains("Mismatched quotes"));
    }

    #[test]
    fn normalizer_preserves_ebnf_and_emits_parseable_lark_conversion() {
        assert_eq!(
            normalize_for_gbnf("root ::= \"yes\"").unwrap(),
            "root ::= \"yes\""
        );
        let generated = normalize_for_gbnf("start: 'yes' | 'no'").unwrap();
        crate::serve::api::grammar::parser::parse(&generated).unwrap();
    }
}
