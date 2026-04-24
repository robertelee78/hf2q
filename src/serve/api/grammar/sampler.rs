//! GBNF runtime sampler — ported from llama.cpp `src/llama-grammar.cpp`
//! functions `llama_grammar_advance_stack` / `llama_grammar_accept` /
//! `llama_grammar_match_char` / `llama_grammar_match_partial_char` /
//! `llama_grammar_reject_candidates_for_stack`.
//!
//! The runtime maintains a set of pushdown stacks over `(rule_id, elem_idx)`
//! positions in the parsed grammar. After each decoded character the stacks
//! are advanced; a token candidate is **accepted** iff at least one stack
//! remains valid after feeding all its characters, and **rejected** if every
//! stack dead-ends.
//!
//! Key port differences vs. the C++ reference:
//!   - llama.cpp uses `const llama_grammar_element *` pointers to identify a
//!     position in the flat `rules[i].data()` buffer. Rust borrow semantics
//!     would make that unwieldy; we use `Pos(rule_id, elem_idx)` instead —
//!     O(1) lookup into `rules[rule_id][elem_idx]`, stable across clones.
//!   - `accept_char` returns a fresh `Stacks` set (immutable-style) rather
//!     than mutating in place. The caller assigns it back to the live state.
//!   - `PartialUtf8` uses `Option<NonZeroU8>` for the byte count instead of
//!     `int n_remain = -1` sentinel. Invalid state is explicitly `Err(...)`.
//!
//! This file is **pure compute** — no axum / mlx-native dependencies. It
//! runs entirely on CPU and can be exercised without a loaded model, which
//! is why the full JSON-acceptance test suite below has no fixture cost.

use super::parser::{GretElement, GretType, Grammar};

// ---------------------------------------------------------------------------
// Position + Stack types
// ---------------------------------------------------------------------------

/// Stable position inside the flat `rules[rule_id][elem_idx]` buffer. Used
/// in place of llama.cpp's raw `const llama_grammar_element *` pointers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pos {
    pub rule_id: u32,
    pub elem_idx: u32,
}

impl Pos {
    pub fn new(rule_id: u32, elem_idx: u32) -> Self {
        Self { rule_id, elem_idx }
    }
    /// Advance to the next element in the same rule. Caller ensures the
    /// next element exists.
    pub fn advance(self) -> Self {
        Self {
            rule_id: self.rule_id,
            elem_idx: self.elem_idx + 1,
        }
    }
}

/// A grammar pushdown stack — a list of `Pos` entries. The TOP of the stack
/// is `.last()` (matches llama.cpp's `stack.back()`).
pub type Stack = Vec<Pos>;
pub type Stacks = Vec<Stack>;

// ---------------------------------------------------------------------------
// Partial UTF-8 accumulator
// ---------------------------------------------------------------------------

/// Accumulator for multi-byte UTF-8 sequences that straddle token boundaries.
/// After accepting a token whose last bytes form a partial UTF-8 sequence,
/// `partial` carries the decoded bits and the remaining byte count into the
/// next token.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PartialUtf8 {
    pub value: u32,
    /// Bytes remaining to complete the code point. `0` = no partial state
    /// (clean). Negative values are represented here as `n_remain = 0` +
    /// the `invalid` flag (llama.cpp uses `-1`; we use a sentinel).
    pub n_remain: i8,
}

// ---------------------------------------------------------------------------
// End-of-sequence test
// ---------------------------------------------------------------------------

fn is_end_of_sequence(e: &GretElement) -> bool {
    matches!(e.ty, GretType::End | GretType::Alt)
}

/// Look up `pos` in the grammar's flat rule buffer. Returns `None` if the
/// position is out of bounds — that shouldn't happen for a valid grammar but
/// the sampler defends against it so a corrupt state can't panic the process.
fn at<'a>(grammar: &'a Grammar, pos: Pos) -> Option<&'a GretElement> {
    grammar
        .rules
        .get(pos.rule_id as usize)
        .and_then(|r| r.get(pos.elem_idx as usize))
}

// ---------------------------------------------------------------------------
// match_char — does `chr` satisfy the char-range element at `pos`?
// ---------------------------------------------------------------------------

/// Returns `(matched, after_range_pos)`. `after_range_pos` points to the
/// element immediately following the char-range group (regardless of match).
///
/// Mirrors `llama_grammar_match_char` at llama-grammar.cpp:758.
pub fn match_char(grammar: &Grammar, mut pos: Pos, chr: u32) -> (bool, Pos) {
    let rule = match grammar.rules.get(pos.rule_id as usize) {
        Some(r) => r.as_slice(),
        None => return (false, pos),
    };
    let head = rule[pos.elem_idx as usize];
    let is_positive_char = matches!(head.ty, GretType::Char | GretType::CharAny);
    debug_assert!(is_positive_char || head.ty == GretType::CharNot);

    let mut found = false;
    loop {
        let cur = rule[pos.elem_idx as usize];
        let next = rule.get(pos.elem_idx as usize + 1);
        if next.map(|e| e.ty) == Some(GretType::CharRngUpper) {
            // inclusive range, e.g. [a-z]
            found = found || (cur.value <= chr && chr <= next.unwrap().value);
            pos.elem_idx += 2;
        } else if cur.ty == GretType::CharAny {
            found = true;
            pos.elem_idx += 1;
        } else {
            // exact char match, e.g. [a] or "a"
            found = found || cur.value == chr;
            pos.elem_idx += 1;
        }
        if rule.get(pos.elem_idx as usize).map(|e| e.ty) != Some(GretType::CharAlt) {
            break;
        }
    }
    (found == is_positive_char, pos)
}

/// Returns `true` iff some continuation of the given partial UTF-8 sequence
/// could satisfy the char range at `pos`. Mirrors
/// `llama_grammar_match_partial_char`.
pub fn match_partial_char(grammar: &Grammar, mut pos: Pos, partial: PartialUtf8) -> bool {
    let rule = match grammar.rules.get(pos.rule_id as usize) {
        Some(r) => r.as_slice(),
        None => return false,
    };
    let head = rule[pos.elem_idx as usize];
    let is_positive_char = matches!(head.ty, GretType::Char | GretType::CharAny);

    let n_remain = partial.n_remain;
    if n_remain < 0 {
        return false;
    }
    if n_remain == 1 && partial.value < 2 {
        return false;
    }

    let shift = (n_remain as u32) * 6;
    let mut low = partial.value << shift;
    let mask = if shift == 0 { 0 } else { (1u32 << shift) - 1 };
    let high = low | mask;
    if low == 0 {
        if n_remain == 2 {
            low = 1 << 11;
        } else if n_remain == 3 {
            low = 1 << 16;
        }
    }

    loop {
        let cur = rule[pos.elem_idx as usize];
        let next = rule.get(pos.elem_idx as usize + 1);
        if next.map(|e| e.ty) == Some(GretType::CharRngUpper) {
            let end = next.unwrap().value;
            if cur.value <= high && low <= end {
                return is_positive_char;
            }
            pos.elem_idx += 2;
        } else if cur.ty == GretType::CharAny {
            return true;
        } else {
            if low <= cur.value && cur.value <= high {
                return is_positive_char;
            }
            pos.elem_idx += 1;
        }
        if rule.get(pos.elem_idx as usize).map(|e| e.ty) != Some(GretType::CharAlt) {
            break;
        }
    }
    !is_positive_char
}

// ---------------------------------------------------------------------------
// advance_stack — expand a stack until every entry is a char-class element
// ---------------------------------------------------------------------------

/// Transforms one stack into the set of stacks that all end at a terminal
/// (char-class) element. Handles `RuleRef` expansion (with alternatives) and
/// skips `End`/`Alt` elements at the stack top.
///
/// Mirrors `llama_grammar_advance_stack` at llama-grammar.cpp:853.
pub fn advance_stack(grammar: &Grammar, stack: Stack, new_stacks: &mut Stacks) {
    let mut todo: Vec<Stack> = Vec::new();
    todo.push(stack);
    // `seen` dedups across our BFS frontier.
    let mut seen: std::collections::HashSet<Stack> =
        std::collections::HashSet::new();

    while let Some(curr_stack) = todo.pop() {
        if seen.contains(&curr_stack) {
            continue;
        }
        seen.insert(curr_stack.clone());

        if curr_stack.is_empty() {
            if !new_stacks.contains(&curr_stack) {
                new_stacks.push(curr_stack);
            }
            continue;
        }

        let top = *curr_stack.last().unwrap();
        let elem = match at(grammar, top) {
            Some(e) => *e,
            None => continue,
        };

        match elem.ty {
            GretType::RuleRef => {
                let rule_id = elem.value;
                let target_rule = match grammar.rules.get(rule_id as usize) {
                    Some(r) => r,
                    None => continue,
                };

                let mut subpos = Pos::new(rule_id, 0);
                loop {
                    // `next_stack` = curr_stack without its top.
                    let mut next_stack: Stack = curr_stack[..curr_stack.len() - 1].to_vec();
                    // If this ref is followed by another element, push that.
                    let follow = top.advance();
                    if let Some(nxt) = at(grammar, follow) {
                        if !is_end_of_sequence(nxt) {
                            next_stack.push(follow);
                        }
                    }
                    // If the target rule's alternate has content, push its start.
                    if let Some(first) = target_rule.get(subpos.elem_idx as usize) {
                        if !is_end_of_sequence(first) {
                            next_stack.push(subpos);
                        }
                    }
                    todo.push(next_stack);

                    // Scan to end-of-sequence within the target rule.
                    loop {
                        let e = match target_rule.get(subpos.elem_idx as usize) {
                            Some(e) => *e,
                            None => {
                                break;
                            }
                        };
                        if is_end_of_sequence(&e) {
                            break;
                        }
                        subpos.elem_idx += 1;
                    }
                    // If stopped on Alt, continue to next alternative; else done.
                    let stop = target_rule.get(subpos.elem_idx as usize).map(|e| e.ty);
                    if stop == Some(GretType::Alt) {
                        subpos.elem_idx += 1;
                    } else {
                        break;
                    }
                }
            }
            GretType::Char | GretType::CharNot | GretType::CharAny => {
                if !new_stacks.contains(&curr_stack) {
                    new_stacks.push(curr_stack);
                }
            }
            _ => {
                // End / Alt / CharAlt / CharRngUpper — stack should never
                // rest on those. Silently drop in the Rust port (C++ aborts).
                tracing::debug!(
                    "grammar::advance_stack: dropping stack with top element type {:?}",
                    elem.ty
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// accept_chr — consume a single character from a stack
// ---------------------------------------------------------------------------

/// Feeds one character to one stack, producing zero or more successor stacks.
/// Mirrors `llama_grammar_accept_chr` at llama-grammar.cpp:1016.
fn accept_chr_into(
    grammar: &Grammar,
    stack: &Stack,
    chr: u32,
    new_stacks: &mut Stacks,
) {
    if stack.is_empty() {
        return;
    }
    let top = *stack.last().unwrap();
    let elem = match at(grammar, top) {
        Some(e) => *e,
        None => return,
    };
    // Tokens aren't handled in this port (see mod-level docs).
    if matches!(elem.ty, GretType::End | GretType::Alt) {
        return;
    }

    let (matched, after) = match_char(grammar, top, chr);
    if matched {
        let mut new_stack: Stack = stack[..stack.len() - 1].to_vec();
        // If the next element isn't End/Alt, push it.
        if let Some(nxt) = at(grammar, after) {
            if !is_end_of_sequence(nxt) {
                new_stack.push(after);
            }
        }
        advance_stack(grammar, new_stack, new_stacks);
    }
}

/// Accept one character against the current stack set, returning the new set.
/// Mirrors `llama_grammar_accept`.
pub fn accept_char(grammar: &Grammar, stacks: &Stacks, chr: u32) -> Stacks {
    let mut new_stacks: Stacks = Vec::with_capacity(stacks.len());
    for stack in stacks {
        accept_chr_into(grammar, stack, chr, &mut new_stacks);
    }
    new_stacks
}

// ---------------------------------------------------------------------------
// Grammar runtime state
// ---------------------------------------------------------------------------

/// Runtime grammar state — wraps the parsed grammar with the current stack
/// set and partial-UTF-8 accumulator. Cheap to clone (the inner grammar is
/// shared via `Arc` by callers; stacks copy explicitly).
#[derive(Debug, Clone)]
pub struct GrammarRuntime {
    pub grammar: Grammar,
    pub stacks: Stacks,
    pub partial_utf8: PartialUtf8,
}

impl GrammarRuntime {
    /// Initialize runtime from a parsed grammar and a start rule.
    /// Mirrors the `llama_grammar_init_impl` behavior of seeding `stacks`
    /// from the start rule's alternatives.
    pub fn new(grammar: Grammar, start_rule_id: u32) -> Option<Self> {
        let start_rule = grammar.rules.get(start_rule_id as usize)?;
        if start_rule.is_empty() {
            return None;
        }
        let mut stacks: Stacks = Vec::new();
        let mut subpos = Pos::new(start_rule_id, 0);
        loop {
            let mut stack: Stack = Vec::new();
            // If the alternative is non-empty, push its start.
            if let Some(first) = start_rule.get(subpos.elem_idx as usize) {
                if !is_end_of_sequence(first) {
                    stack.push(subpos);
                }
            }
            let mut advanced: Stacks = Vec::new();
            advance_stack(&grammar, stack, &mut advanced);
            for s in advanced {
                if !stacks.contains(&s) {
                    stacks.push(s);
                }
            }
            // Scan to end-of-sequence.
            loop {
                let e = match start_rule.get(subpos.elem_idx as usize) {
                    Some(e) => *e,
                    None => break,
                };
                if is_end_of_sequence(&e) {
                    break;
                }
                subpos.elem_idx += 1;
            }
            let stop = start_rule.get(subpos.elem_idx as usize).map(|e| e.ty);
            if stop == Some(GretType::Alt) {
                subpos.elem_idx += 1;
            } else {
                break;
            }
        }
        Some(Self {
            grammar,
            stacks,
            partial_utf8: PartialUtf8::default(),
        })
    }

    /// Feed one Unicode code point. Returns `true` if any stacks remain
    /// after the feed (i.e. the grammar still has a valid continuation).
    pub fn accept_char(&mut self, chr: u32) -> bool {
        self.stacks = accept_char(&self.grammar, &self.stacks, chr);
        !self.stacks.is_empty()
    }

    /// Feed a byte string (e.g. a decoded token text). Partial UTF-8 bytes
    /// at the tail are carried in `self.partial_utf8`. Returns `true` if the
    /// grammar still has a valid continuation after consuming all the bytes.
    pub fn accept_bytes(&mut self, bytes: &[u8]) -> bool {
        let mut i = 0;

        // If there's an unfinished UTF-8 code point from the previous call,
        // try to complete it first. Only runs when `partial.n_remain > 0`.
        if self.partial_utf8.n_remain > 0 {
            let mut partial = self.partial_utf8;
            while i < bytes.len() && partial.n_remain > 0 {
                let b = bytes[i];
                if (b & 0xC0) != 0x80 {
                    self.partial_utf8 = PartialUtf8 { value: 0, n_remain: -1 };
                    self.stacks.clear();
                    return false;
                }
                partial.value = (partial.value << 6) | (b & 0x3F) as u32;
                partial.n_remain -= 1;
                i += 1;
            }
            if partial.n_remain > 0 {
                // Still incomplete at end-of-buffer — stash for next call.
                self.partial_utf8 = partial;
                return !self.stacks.is_empty();
            }
            // Completed code point.
            self.partial_utf8 = PartialUtf8::default();
            if !self.accept_char(partial.value) {
                return false;
            }
        }

        // Now consume full code points from `bytes[i..]`.
        while i < bytes.len() {
            let first = bytes[i];
            let (needed, mut val) = if first & 0x80 == 0 {
                (0usize, first as u32)
            } else if first & 0xE0 == 0xC0 {
                (1, (first & 0x1F) as u32)
            } else if first & 0xF0 == 0xE0 {
                (2, (first & 0x0F) as u32)
            } else if first & 0xF8 == 0xF0 {
                (3, (first & 0x07) as u32)
            } else {
                self.stacks.clear();
                return false;
            };
            i += 1;
            let mut remain = needed as i8;
            while remain > 0 && i < bytes.len() {
                let b = bytes[i];
                if (b & 0xC0) != 0x80 {
                    self.stacks.clear();
                    return false;
                }
                val = (val << 6) | (b & 0x3F) as u32;
                i += 1;
                remain -= 1;
            }
            if remain > 0 {
                self.partial_utf8 = PartialUtf8 { value: val, n_remain: remain };
                return !self.stacks.is_empty();
            }
            if !self.accept_char(val) {
                return false;
            }
        }
        self.partial_utf8 = PartialUtf8::default();
        !self.stacks.is_empty()
    }

    /// Returns `true` if the grammar is in an accepting state — i.e. any
    /// stack is empty (meaning the root rule has been fully matched).
    pub fn is_accepted(&self) -> bool {
        self.stacks.iter().any(|s| s.is_empty())
    }

    /// Is the grammar dead? (no stacks remain; no continuation can satisfy it).
    pub fn is_dead(&self) -> bool {
        self.stacks.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::parser::parse;

    fn runtime_from(src: &str, start: &str) -> GrammarRuntime {
        let g = parse(src).expect("parse");
        let rid = g.rule_id(start).expect("start rule exists");
        GrammarRuntime::new(g, rid).expect("start rule nonempty")
    }

    #[test]
    fn accept_exact_literal_sequence() {
        let mut rt = runtime_from("root ::= \"abc\"\n", "root");
        assert!(!rt.is_dead());
        assert!(rt.accept_char('a' as u32));
        assert!(rt.accept_char('b' as u32));
        assert!(rt.accept_char('c' as u32));
        assert!(rt.is_accepted(), "grammar should be in accepting state");
    }

    #[test]
    fn reject_wrong_literal() {
        let mut rt = runtime_from("root ::= \"abc\"\n", "root");
        assert!(!rt.accept_char('X' as u32));
        assert!(rt.is_dead());
    }

    #[test]
    fn char_class_range_accepts_in_range() {
        let mut rt = runtime_from("root ::= [a-z]\n", "root");
        assert!(rt.accept_char('m' as u32));
        assert!(rt.is_accepted());
    }

    #[test]
    fn char_class_range_rejects_out_of_range() {
        let mut rt = runtime_from("root ::= [a-z]\n", "root");
        assert!(!rt.accept_char('A' as u32));
        assert!(rt.is_dead());
    }

    #[test]
    fn negated_char_class() {
        let mut rt = runtime_from("root ::= [^abc]\n", "root");
        assert!(rt.accept_char('z' as u32));
        assert!(rt.is_accepted());

        let mut rt2 = runtime_from("root ::= [^abc]\n", "root");
        assert!(!rt2.accept_char('a' as u32));
    }

    #[test]
    fn any_char_dot_accepts_anything() {
        let mut rt = runtime_from("root ::= .\n", "root");
        assert!(rt.accept_char('Q' as u32));
        assert!(rt.is_accepted());
    }

    #[test]
    fn alternation_either_path_accepted() {
        let mut rt = runtime_from("root ::= \"yes\" | \"no\"\n", "root");
        assert!(rt.accept_char('n' as u32));
        assert!(rt.accept_char('o' as u32));
        assert!(rt.is_accepted());
    }

    #[test]
    fn alternation_wrong_prefix_rejected() {
        let mut rt = runtime_from("root ::= \"yes\" | \"no\"\n", "root");
        assert!(!rt.accept_char('q' as u32));
    }

    #[test]
    fn rule_reference_chain() {
        let mut rt = runtime_from(
            "root ::= ws \"hi\"\nws ::= \" \"\n",
            "root",
        );
        assert!(rt.accept_char(' ' as u32));
        assert!(rt.accept_char('h' as u32));
        assert!(rt.accept_char('i' as u32));
        assert!(rt.is_accepted());
    }

    #[test]
    fn kleene_star_zero_occurrences() {
        let rt = runtime_from("root ::= \"a\"*\n", "root");
        // zero occurrences — already accepted.
        assert!(rt.is_accepted());
    }

    #[test]
    fn kleene_star_many_occurrences() {
        let mut rt = runtime_from("root ::= \"a\"*\n", "root");
        for _ in 0..10 {
            assert!(rt.accept_char('a' as u32));
        }
        assert!(rt.is_accepted());
    }

    #[test]
    fn plus_requires_at_least_one() {
        let rt_zero = runtime_from("root ::= \"a\"+\n", "root");
        // Before accepting any char, should not yet be accepted.
        assert!(!rt_zero.is_accepted());

        let mut rt = runtime_from("root ::= \"a\"+\n", "root");
        assert!(rt.accept_char('a' as u32));
        assert!(rt.is_accepted());
        assert!(rt.accept_char('a' as u32));
        assert!(rt.is_accepted());
    }

    #[test]
    fn optional_question_mark_both_paths() {
        let rt_empty = runtime_from("root ::= \"x\"?\n", "root");
        assert!(rt_empty.is_accepted());

        let mut rt_one = runtime_from("root ::= \"x\"?\n", "root");
        assert!(rt_one.accept_char('x' as u32));
        assert!(rt_one.is_accepted());

        let mut rt_two = runtime_from("root ::= \"x\"?\n", "root");
        rt_two.accept_char('x' as u32);
        assert!(!rt_two.accept_char('x' as u32));
    }

    #[test]
    fn brace_exact_count_rejects_over() {
        let mut rt = runtime_from("root ::= \"a\"{3}\n", "root");
        assert!(rt.accept_char('a' as u32));
        assert!(!rt.is_accepted());
        assert!(rt.accept_char('a' as u32));
        assert!(!rt.is_accepted());
        assert!(rt.accept_char('a' as u32));
        assert!(rt.is_accepted());
        assert!(!rt.accept_char('a' as u32));
    }

    #[test]
    fn brace_range_accepts_within() {
        for count in 0..=3 {
            let mut rt = runtime_from("root ::= \"a\"{0,3}\n", "root");
            for _ in 0..count {
                assert!(rt.accept_char('a' as u32), "count={}", count);
            }
            assert!(rt.is_accepted(), "count={}", count);
        }
        // 4 'a's — extra char should be rejected.
        let mut rt = runtime_from("root ::= \"a\"{0,3}\n", "root");
        for _ in 0..3 {
            rt.accept_char('a' as u32);
        }
        assert!(!rt.accept_char('a' as u32));
    }

    #[test]
    fn accept_bytes_utf8() {
        let mut rt = runtime_from("root ::= \"α\"\n", "root");
        // Greek alpha UTF-8 is CE B1.
        assert!(rt.accept_bytes("α".as_bytes()));
        assert!(rt.is_accepted());
    }

    #[test]
    fn accept_bytes_incremental_utf8() {
        let mut rt = runtime_from("root ::= \"α\"\n", "root");
        // Feed the first byte — should accumulate as partial.
        assert!(rt.accept_bytes(&[0xCE]));
        assert!(!rt.is_accepted(), "not yet accepted after 1 byte");
        // Feed the rest.
        assert!(rt.accept_bytes(&[0xB1]));
        assert!(rt.is_accepted());
    }

    #[test]
    fn json_grammar_value_rule_accepts_scalars_and_arrays() {
        // llama.cpp's `json.gbnf` has `root ::= object` (root accepts ONLY
        // a top-level object), but `value ::= object | array | string |
        // number | ("true" | "false" | "null") ws` accepts everything.
        // Verify each alternative against the `value` rule.
        let src = std::fs::read_to_string("/opt/llama.cpp/grammars/json.gbnf")
            .expect("json.gbnf fixture");
        for input in [
            "null",
            "true",
            "false",
            "123",
            "-4.2",
            "\"hello\"",
            "[]",
            "[1,2,3]",
            "{}",
            "{\"k\":\"v\"}",
            "{\"a\":1,\"b\":[true,false]}",
        ] {
            let g = parse(&src).expect("parse");
            let rid = g.rule_id("value").unwrap();
            let mut rt = GrammarRuntime::new(g, rid).unwrap();
            assert!(
                rt.accept_bytes(input.as_bytes()),
                "json grammar (value rule) should accept {:?}", input
            );
            assert!(
                rt.is_accepted(),
                "json grammar (value rule) should ACCEPT {:?}", input
            );
        }
    }

    #[test]
    fn json_grammar_root_rule_requires_object() {
        // `root ::= object` — bare scalars rejected, objects accepted.
        let src = std::fs::read_to_string("/opt/llama.cpp/grammars/json.gbnf")
            .expect("json.gbnf fixture");
        for good_object in ["{}", "{\"k\":\"v\"}", "{\"a\":1,\"b\":[true,false]}"] {
            let g = parse(&src).expect("parse");
            let rid = g.rule_id("root").unwrap();
            let mut rt = GrammarRuntime::new(g, rid).unwrap();
            assert!(rt.accept_bytes(good_object.as_bytes()));
            assert!(rt.is_accepted(), "root must accept {:?}", good_object);
        }
        for bad_scalar in ["null", "42", "\"hello\""] {
            let g = parse(&src).expect("parse");
            let rid = g.rule_id("root").unwrap();
            let mut rt = GrammarRuntime::new(g, rid).unwrap();
            let ok = rt.accept_bytes(bad_scalar.as_bytes());
            assert!(
                !(ok && rt.is_accepted()),
                "root MUST reject bare scalar {:?}",
                bad_scalar
            );
        }
    }

    #[test]
    fn json_grammar_rejects_malformed() {
        let src = std::fs::read_to_string("/opt/llama.cpp/grammars/json.gbnf")
            .expect("json.gbnf fixture");
        for input in [
            "nul",        // truncated
            "tru e",      // space in literal
            "[1,",        // truncated array
            "{\"k\":}",   // missing value
            "\"unterminated",
        ] {
            let g = parse(&src).expect("parse");
            let rid = g.rule_id("root").unwrap();
            let mut rt = GrammarRuntime::new(g, rid).unwrap();
            let ok = rt.accept_bytes(input.as_bytes());
            // Either bytes were rejected mid-stream OR the grammar isn't in
            // an accepting state at end-of-input.
            let final_accepted = ok && rt.is_accepted();
            assert!(
                !final_accepted,
                "json grammar MUST reject {:?}, but it accepted",
                input
            );
        }
    }

    #[test]
    fn json_grammar_rejects_trailing_garbage_after_object() {
        // `root ::= object` — after a valid object, only trailing ws is
        // allowed. Alphabetic garbage is rejected.
        let src = std::fs::read_to_string("/opt/llama.cpp/grammars/json.gbnf")
            .expect("json.gbnf fixture");
        let g = parse(&src).expect("parse");
        let rid = g.rule_id("root").unwrap();
        let mut rt = GrammarRuntime::new(g, rid).unwrap();
        assert!(rt.accept_bytes(b"{}"));
        let still_alive = rt.accept_bytes(b"x");
        assert!(
            !still_alive,
            "json root rule must reject 'x' after a complete '{{}}'"
        );
    }

    #[test]
    fn dead_grammar_stays_dead() {
        let mut rt = runtime_from("root ::= \"a\"\n", "root");
        assert!(!rt.accept_char('x' as u32));
        assert!(rt.is_dead());
        assert!(!rt.accept_char('a' as u32));
        assert!(rt.is_dead());
    }

    #[test]
    fn partial_utf8_initial_state_is_clean() {
        let rt = runtime_from("root ::= \"a\"\n", "root");
        assert_eq!(rt.partial_utf8, PartialUtf8::default());
    }
}
