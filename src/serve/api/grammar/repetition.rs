//! Exact GBNF composition of bounded repetitions above the parser's
//! per-operator limit (issue #251).
//!
//! The GBNF parser (`parser.rs`, mirroring the peer) rejects any single
//! `{m,n}` operator whose bounds exceed `MAX_REPETITION_THRESHOLD` (2,000),
//! and any operator for which `n_prev_rules * total_rules >= 2,000`, where
//! `total_rules` is the operator's maximum (or its minimum when unbounded)
//! and `n_prev_rules` counts the rules synthesized by the repeated item
//! (1 for a bare symbol; a group counts every rule generated inside it).
//! JSON Schema length/count bounds such as `maxLength: 4096` therefore cannot
//! be emitted as one operator.
//!
//! [`exact_repetition`] instead emits a language-identical composition in
//! which every operator applies to a bare symbol and stays far below the
//! threshold. With segment size `S` ([`REPETITION_SEGMENT`]) and a shared
//! block rule `B ::= x{S}`:
//!
//! - `x{m}` becomes `B{m / S} x{m mod S}`. Concatenating exact counts of
//!   the same item accepts exactly `m` copies.
//! - `x{0,k}` for `k <= S` is emitted directly. For `k > S` it becomes a
//!   rule `L(k) ::= x{0,S-1} | B L(k-S)`. The first alternative accepts
//!   0..=S-1 copies and the second accepts S..=k copies, so the union is
//!   exactly 0..=k and the alternatives are disjoint by length. The
//!   construction is unambiguous: at most two alternatives share a live
//!   prefix at any time, so runtime stack count stays constant instead of
//!   growing with the number of segments, as a plain concatenation
//!   `x{0,S} x{0,S} ...` would (every split point is a separate parse).
//! - `x{m,n}` is `x{m}` followed by `x{0,n-m}`; `x{m,}` is `x{m}` followed
//!   by `x*`.
//!
//! `B` and the `x{0,S-1}` chain are shared by every level, so a bound `N`
//! costs O(S + N / S) rules and elements rather than O(N); this keeps
//! grammars small enough that requests carrying many bounded tools still fit
//! the combined-grammar byte/rule limits.
//!
//! The composition is placed in a content-addressed named rule and the
//! caller receives only that rule name, so embedding it inside a group that
//! carries its own repetition operator adds exactly one symbol to that
//! group's `n_prev_rules` count. Rule names are derived from the rule body,
//! so two definitions with the same name are always identical.

/// Size of one repetition segment `S`.
///
/// Every emitted operator applies to a bare symbol (`n_prev_rules == 1`) and
/// is at most `max(S, MAX_SCHEMA_REPETITION_BOUND / S)` = 1,024, strictly
/// below the parser's 2,000 threshold (`1 * 2000 >= 2000` is rejected, so
/// 2,000 itself is not usable). 64 roughly minimizes `S + N / S` for the
/// 4,096 bounds seen in real tool schemas.
pub(crate) const REPETITION_SEGMENT: u64 = 64;

/// Largest JSON Schema length/count bound (`minLength`, `maxLength`,
/// `minItems`, `maxItems`, `minProperties`, `maxProperties`) accepted for
/// exact grammar compilation.
///
/// Real tool schemas observed in OpenCode/MCP use 4,096; 65,536 leaves
/// headroom while still rejecting adversarial bounds such as
/// `maxLength: 10^9` with a clear request error. At the ceiling one field
/// costs about 1,200 rules and the runtime peaks at a handful of stacks.
pub(crate) const MAX_SCHEMA_REPETITION_BOUND: u64 = 65_536;

const _: () = assert!(MAX_SCHEMA_REPETITION_BOUND / REPETITION_SEGMENT < 2000);
const _: () = assert!(REPETITION_SEGMENT < 2000);

/// A composed repetition: `expr` is a single bare symbol (a rule name, or
/// the empty literal `""` when the maximum is zero) and `rules` are the
/// `(name, body)` definitions it depends on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Repetition {
    pub expr: String,
    pub rules: Vec<(String, String)>,
}

/// Compose `atom` repeated between `min` and `max` (unbounded when `None`)
/// times, exactly, using only operators the parser accepts.
///
/// `atom` may be any GBNF sequence; a non-identifier atom is first hoisted
/// into its own rule. Callers MUST ensure `max >= min`.
pub(crate) fn exact_repetition(atom: &str, min: u64, max: Option<u64>) -> Repetition {
    debug_assert!(max.is_none_or(|upper| upper >= min));
    let mut rules = Vec::new();
    if max == Some(0) {
        return Repetition {
            expr: "\"\"".to_string(),
            rules,
        };
    }
    let symbol = if is_bare_symbol(atom) {
        atom.to_string()
    } else {
        let name = format!("rep-atom-{:016x}", fnv1a(&[atom]));
        rules.push((name.clone(), atom.to_string()));
        name
    };

    let mut parts = exact_count(&symbol, min, &mut rules);
    match max {
        None => parts.push(format!("{symbol}*")),
        Some(upper) if upper > min => parts.push(optional_up_to(&symbol, upper - min, &mut rules)),
        Some(_) => {}
    }
    let body = parts.join(" ");
    let name = format!("rep-{:016x}", fnv1a(&[&symbol, &body]));
    push_rule(&mut rules, &name, body);
    Repetition { expr: name, rules }
}

/// `symbol` exactly `count` times: whole blocks, then the remainder.
fn exact_count(symbol: &str, count: u64, rules: &mut Vec<(String, String)>) -> Vec<String> {
    let mut parts = Vec::new();
    match count / REPETITION_SEGMENT {
        0 => {}
        1 => parts.push(block_rule(symbol, rules)),
        blocks => parts.push(format!("{}{{{blocks}}}", block_rule(symbol, rules))),
    }
    match count % REPETITION_SEGMENT {
        0 => {}
        1 => parts.push(symbol.to_string()),
        rest => parts.push(format!("{symbol}{{{rest}}}")),
    }
    parts
}

/// Shared rule `B ::= symbol{S}`.
fn block_rule(symbol: &str, rules: &mut Vec<(String, String)>) -> String {
    let body = format!("{symbol}{{{REPETITION_SEGMENT}}}");
    let name = format!("rep-block-{:016x}", fnv1a(&[&body]));
    push_rule(rules, &name, body);
    name
}

/// A single-symbol expression accepting 0..=`count` copies of `symbol`.
fn optional_up_to(symbol: &str, count: u64, rules: &mut Vec<(String, String)>) -> String {
    debug_assert!(count > 0);
    // `levels` rules of the form `L ::= x{0,S-1} | B L'`, built from the
    // innermost (which accepts 1..=S copies directly) outward.
    let levels = (count - 1) / REPETITION_SEGMENT;
    let innermost = count - levels * REPETITION_SEGMENT;
    let mut expression = if innermost == 1 {
        format!("{symbol}?")
    } else {
        format!("{symbol}{{0,{innermost}}}")
    };
    if levels == 0 {
        return expression;
    }
    let below_body = format!("{symbol}{{0,{}}}", REPETITION_SEGMENT - 1);
    let below = format!("rep-below-{:016x}", fnv1a(&[&below_body]));
    push_rule(rules, &below, below_body);
    let block = block_rule(symbol, rules);
    for _ in 0..levels {
        let body = format!("{below} | {block} {expression}");
        let name = format!("rep-opt-{:016x}", fnv1a(&[&body]));
        push_rule(rules, &name, body);
        expression = name;
    }
    expression
}

fn push_rule(rules: &mut Vec<(String, String)>, name: &str, body: String) {
    if !rules.iter().any(|(existing, _)| existing == name) {
        rules.push((name.to_string(), body));
    }
}

fn is_bare_symbol(atom: &str) -> bool {
    !atom.is_empty()
        && atom
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
}

fn fnv1a(parts: &[&str]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for part in parts {
        for byte in part.bytes().chain(std::iter::once(0)) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::super::parser::parse_generated;
    use super::super::sampler::GrammarRuntime;
    use super::*;

    fn grammar(min: u64, max: Option<u64>) -> String {
        let repetition = exact_repetition("x", min, max);
        let mut source = format!("root ::= \"[\" {} \"]\"\nx ::= \"a\"\n", repetition.expr);
        for (name, body) in repetition.rules {
            source.push_str(&format!("{name} ::= {body}\n"));
        }
        source
    }

    /// Returns (accepted, peak active stacks).
    fn accepts(source: &str, count: usize) -> (bool, usize) {
        let parsed = parse_generated(source).unwrap_or_else(|error| panic!("{error}\n{source}"));
        let root = parsed.rule_id("root").unwrap();
        let mut runtime = GrammarRuntime::new(parsed, root).unwrap();
        let mut peak = runtime.stacks.len();
        let mut ok = runtime.accept_bytes(b"[");
        for _ in 0..count {
            if !ok {
                break;
            }
            ok = runtime.accept_bytes(b"a");
            peak = peak.max(runtime.stacks.len());
        }
        (
            ok && runtime.accept_bytes(b"]") && runtime.is_accepted(),
            peak,
        )
    }

    #[test]
    fn segments_stay_within_parser_threshold() {
        for (min, max) in [
            (0, Some(1999)),
            (0, Some(2000)),
            (0, Some(2001)),
            (2001, Some(4096)),
            (4096, None),
            (0, Some(MAX_SCHEMA_REPETITION_BOUND)),
            (
                MAX_SCHEMA_REPETITION_BOUND,
                Some(MAX_SCHEMA_REPETITION_BOUND),
            ),
        ] {
            let source = grammar(min, max);
            parse_generated(&source)
                .unwrap_or_else(|error| panic!("{min}..{max:?}: {error}\n{source}"));
        }
    }

    #[test]
    fn composed_bounds_are_exact() {
        for (min, max) in [
            (0, Some(64)),
            (0, Some(65)),
            (63, Some(129)),
            (128, Some(128)),
            (0, Some(2000)),
            (0, Some(2001)),
            (1999, Some(2001)),
            (2001, Some(2001)),
            (0, Some(4096)),
            (2500, Some(4096)),
        ] {
            let source = grammar(min, max);
            let upper = max.unwrap() as usize;
            let lower = min as usize;
            assert!(accepts(&source, upper).0, "{min}..{max:?} rejects {upper}");
            assert!(accepts(&source, lower).0, "{min}..{max:?} rejects {lower}");
            assert!(
                !accepts(&source, upper + 1).0,
                "{min}..{max:?} accepts {}",
                upper + 1
            );
            if lower > 0 {
                assert!(
                    !accepts(&source, lower - 1).0,
                    "{min}..{max:?} accepts {}",
                    lower - 1
                );
            }
        }
        let unbounded = grammar(2001, None);
        assert!(!accepts(&unbounded, 2000).0);
        assert!(accepts(&unbounded, 2001).0);
        assert!(accepts(&unbounded, 5000).0);
    }

    #[test]
    fn composition_keeps_runtime_stacks_bounded() {
        let source = grammar(0, Some(MAX_SCHEMA_REPETITION_BOUND));
        let full = MAX_SCHEMA_REPETITION_BOUND as usize;
        let (accepted, peak) = accepts(&source, full);
        assert!(accepted);
        assert!(peak <= 8, "unambiguous composition peaked at {peak} stacks");
        assert!(!accepts(&source, full + 1).0);
    }
}
