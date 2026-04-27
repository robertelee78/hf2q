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
///
/// # Trigger gate (Wave 2.6 W-α5 Q2)
///
/// The optional `awaiting_trigger` flag implements the **lazy grammar /
/// trigger-activated FSM** pattern from llama.cpp PR #9639 (canonical
/// implementation at `/opt/llama.cpp/src/llama-grammar.cpp:1287-1439`):
///
/// * `apply` (mask) — `mask_invalid_tokens` short-circuits to 0 masked
///   when `is_awaiting_trigger()` is true.  All preamble tokens stay
///   live; the model can emit any text up to the open marker.
/// * `accept` (advance) — `accept_bytes` is a no-op returning `true`
///   (alive) while awaiting trigger.  No stacks advance, no UTF-8
///   accumulator state changes.
/// * `is_dead`     — returns `false` while awaiting trigger.  A
///   suspended runtime never dies.
/// * `is_accepted` — returns `false` while awaiting trigger.  A
///   suspended runtime never reaches the accepting state.
///
/// All three short-circuits gate on the SAME boolean — no split-state
/// window where mask says "off" but advance says "on".  This is the
/// architectural property the wave-2.5 audit caught when the gate lived
/// in a separate `Arc<AtomicBool>` outside the runtime
/// (cfa-20260427-adr005-wave2.5/codex-review-last.txt divergence A1,
/// citing `engine.rs:1401, 1489, 1554, 2041, 2145, 2195`).
///
/// The flag is set EXPLICITLY by the caller via
/// [`GrammarRuntime::set_awaiting_trigger`] at construction time, and
/// is flipped to false by [`GrammarRuntime::trigger`] when the engine's
/// `ToolCallSplitter` sees the per-model open marker (e.g. Gemma 4
/// `call:`, Qwen 3.5 `<function=`).  llama.cpp does NOT reset the flag
/// per call — multi-tool support comes from the chat-template-rendered
/// grammar accepting `(call)+` directly.  See PR #9639 / Hermes 2 Pro
/// template in `/opt/llama.cpp/docs/function-calling.md`.
///
/// Default for new runtimes is `awaiting_trigger == false` so existing
/// callers that don't opt in get the eager-enforcement behavior
/// unchanged (the wave-2.4 baseline).
#[derive(Debug, Clone)]
pub struct GrammarRuntime {
    pub grammar: Grammar,
    pub stacks: Stacks,
    pub partial_utf8: PartialUtf8,
    /// Trigger gate.  When `true`, `accept_bytes`/`is_dead`/`is_accepted`
    /// all short-circuit so the runtime is functionally suspended.  Flipped
    /// false either explicitly by [`GrammarRuntime::trigger`] (e.g. when
    /// the engine's tool-call splitter sees the open marker) or implicitly
    /// at construction (the default for `GrammarKind::ResponseFormat`
    /// runtimes).
    awaiting_trigger: bool,
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
            awaiting_trigger: false,
        })
    }

    // -----------------------------------------------------------------
    // Wave 2.6 W-α5 Q2 — trigger gate (lazy-grammar pattern)
    // -----------------------------------------------------------------

    /// Read the trigger gate.
    ///
    /// When `true`, the runtime is functionally suspended: `accept_bytes`
    /// is a no-op, `is_dead`/`is_accepted` both return `false`, and
    /// `mask::mask_invalid_tokens` short-circuits to 0 masked tokens.
    pub fn is_awaiting_trigger(&self) -> bool {
        self.awaiting_trigger
    }

    /// Explicitly set the trigger gate.  Engine callers wire this to
    /// `true` for `GrammarKind::ToolCallBody` runtimes at construction;
    /// `GrammarKind::ResponseFormat` runtimes leave the default
    /// (`false`) so enforcement is eager.
    ///
    /// Mirrors llama.cpp's `lazy` parameter to `llama_grammar_init_impl`
    /// at `/opt/llama.cpp/src/llama-grammar.cpp:1287-1298`:
    ///
    /// ```c++
    /// llama_grammar_init_impl(..., bool lazy, ...)
    /// {
    ///     ...
    ///     grammar->awaiting_trigger = lazy;
    /// }
    /// ```
    pub fn set_awaiting_trigger(&mut self, value: bool) {
        self.awaiting_trigger = value;
    }

    /// Flip the trigger gate to `false`.  Called by the engine when the
    /// `ToolCallSplitter` reports a `ToolCallOpen` event.  After this,
    /// every subsequent `accept_bytes` / mask call enforces the grammar
    /// normally.
    ///
    /// llama.cpp does NOT reset this flag back to `true` on the close
    /// marker — multi-tool support comes from the chat-template-rendered
    /// grammar shape accepting `(call)+` directly (Hermes 2 Pro).  See
    /// research-report.md Q2 anti-finding + `/opt/llama.cpp/docs/function-calling.md`.
    pub fn trigger(&mut self) {
        self.awaiting_trigger = false;
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
    ///
    /// Wave 2.6 W-α5 Q2: when [`is_awaiting_trigger`] is `true`, this is
    /// a no-op returning `true` (alive).  No stacks advance, no UTF-8
    /// accumulator state changes.  The runtime is suspended until the
    /// engine calls [`trigger`] (typically in the `ToolCallOpen`
    /// handler).  Mirrors llama.cpp `llama_grammar_accept_impl` at
    /// `/opt/llama.cpp/src/llama-grammar.cpp:1382-1439`.
    pub fn accept_bytes(&mut self, bytes: &[u8]) -> bool {
        if self.awaiting_trigger {
            // Self-gate: no advance while suspended.  Return `true`
            // (alive) so callers that test the return value treat
            // the runtime as still-viable.  This is the apply+accept
            // single-boolean atomicity invariant from research-report.md
            // Q2.
            return true;
        }
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
    ///
    /// Wave 2.6 W-α5 Q2: a suspended runtime
    /// (`is_awaiting_trigger() == true`) is NEVER in an accepting state.
    /// Returning `false` here lets the engine call this method
    /// unconditionally on every step (no separate gate around the call
    /// site needed) — the trigger gate IS the gate.
    pub fn is_accepted(&self) -> bool {
        if self.awaiting_trigger {
            return false;
        }
        self.stacks.iter().any(|s| s.is_empty())
    }

    /// Is the grammar dead? (no stacks remain; no continuation can satisfy it).
    ///
    /// Wave 2.6 W-α5 Q2: a suspended runtime
    /// (`is_awaiting_trigger() == true`) is NEVER dead.  Returning
    /// `false` here lets the engine call this method unconditionally on
    /// every step — the wave-2.5 audit divergence at engine.rs:1554 +
    /// :2195 ("dead-check global") is no longer a divergence because
    /// the runtime self-gates instead of relying on a sibling
    /// `Arc<AtomicBool>`.
    pub fn is_dead(&self) -> bool {
        if self.awaiting_trigger {
            return false;
        }
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

    // -----------------------------------------------------------------
    // Wave 2.6 W-α5 Q2 — trigger-gate (lazy grammar) contract tests
    // -----------------------------------------------------------------
    //
    // These exercise the runtime self-gate that replaces the wave-2.5
    // `Arc<AtomicBool>` sibling in engine.rs.  Each test is a direct
    // expression of an audit divergence (cf. cfa-20260427-adr005-wave2.5
    // /codex-review-last.txt) — together they prove the new contract
    // makes the divergence structurally impossible.

    /// Audit fix: `accept_bytes` is a no-op while
    /// `is_awaiting_trigger()` is true.  This is the wave-2.5 audit
    /// divergence "advance the grammar runtime unconditionally outside
    /// the gated region" (engine.rs:1401, 1489, 2041, 2145) — moving
    /// the gate INSIDE the runtime makes the unconditional advance
    /// safe.
    #[test]
    fn runtime_accept_noops_when_awaiting_trigger() {
        // Grammar that requires literal "abc"; if accept_bytes were
        // NOT a no-op while suspended, feeding "xyz" would die.
        let mut rt = runtime_from("root ::= \"abc\"\n", "root");
        rt.set_awaiting_trigger(true);
        // Snapshot pre-state for byte-exact equality check.
        let pre_stacks = rt.stacks.clone();
        let pre_partial = rt.partial_utf8;

        // Feed garbage bytes.  Must NOT advance, must NOT clear stacks.
        let alive = rt.accept_bytes(b"xyz");
        assert!(
            alive,
            "suspended runtime must report alive=true (return value semantics)"
        );
        assert_eq!(
            rt.stacks, pre_stacks,
            "suspended runtime stacks MUST NOT change on accept_bytes"
        );
        assert_eq!(
            rt.partial_utf8, pre_partial,
            "suspended runtime partial_utf8 MUST NOT change on accept_bytes"
        );
        assert!(rt.is_awaiting_trigger(), "trigger gate stays armed");

        // Now flip the gate; the original grammar should still be intact.
        rt.trigger();
        assert!(!rt.is_awaiting_trigger());
        assert!(rt.accept_bytes(b"abc"), "post-trigger grammar accepts literal");
        assert!(rt.is_accepted(), "literal fully matched");
    }

    /// Audit fix: `is_dead()` returns false while suspended.  This
    /// kills the wave-2.5 audit divergence "src/serve/api/engine.rs:1554
    /// and 2195 also terminate on dead grammar regardless of gate" —
    /// the engine can call `is_dead()` unconditionally because the
    /// runtime self-gates.
    #[test]
    fn runtime_is_dead_returns_false_while_awaiting_trigger() {
        let mut rt = runtime_from("root ::= \"abc\"\n", "root");
        rt.set_awaiting_trigger(true);

        // Even if the underlying stacks were artificially cleared, the
        // gate masks death.  Most natural pre-trigger state is the
        // construction default (stacks non-empty), which would also
        // report not-dead.  Force the worst case explicitly:
        rt.stacks.clear();
        assert!(
            !rt.is_dead(),
            "suspended runtime MUST NOT report dead even when stacks empty"
        );
        assert!(
            !rt.is_accepted(),
            "suspended runtime MUST NOT report accepted even when stacks empty"
        );

        // Triggering reveals the underlying state honestly.
        rt.trigger();
        assert!(
            rt.is_dead(),
            "post-trigger runtime with empty stacks IS dead"
        );
    }

    /// Audit fix: `is_accepted()` returns false while suspended.
    /// Symmetric to the dead-check fix; ensures the early-termination
    /// branch in the streaming decode loop (engine.rs:2199-2208) is
    /// safe to call unconditionally.
    #[test]
    fn runtime_is_accepted_returns_false_while_awaiting_trigger() {
        // `"a"*` is in an accepting state immediately (zero
        // occurrences satisfies the kleene star).
        let mut rt = runtime_from("root ::= \"a\"*\n", "root");
        assert!(rt.is_accepted(), "kleene star is accepted at zero occurrences");

        rt.set_awaiting_trigger(true);
        assert!(
            !rt.is_accepted(),
            "suspended runtime MUST NOT report accepted even when underlying \
             grammar IS in an accepting state"
        );

        rt.trigger();
        assert!(rt.is_accepted(), "post-trigger reveals the actual state");
    }

    /// Default for new runtimes is awaiting_trigger=false (eager
    /// enforcement).  This is the contract that makes
    /// `GrammarKind::ResponseFormat` safe — its runtime never waits.
    /// Without this default, every existing caller of
    /// `GrammarRuntime::new` would silently enter the lazy-grammar
    /// state and break response_format=json_schema enforcement.
    #[test]
    fn runtime_default_is_eager_not_awaiting_trigger() {
        let rt = runtime_from("root ::= \"abc\"\n", "root");
        assert!(
            !rt.is_awaiting_trigger(),
            "default runtime MUST NOT await trigger (eager enforcement is the safe \
             default; ResponseFormat-kind grammars rely on this)"
        );
    }

    /// `set_awaiting_trigger(true)` then `trigger()` is the explicit
    /// path used by the engine for `GrammarKind::ToolCallBody`
    /// runtimes.  Verifies the round-trip restores normal enforcement.
    #[test]
    fn runtime_set_then_trigger_restores_eager_enforcement() {
        let mut rt = runtime_from("root ::= \"abc\"\n", "root");
        rt.set_awaiting_trigger(true);
        assert!(rt.is_awaiting_trigger());

        // Pre-trigger garbage MUST be ignored.
        let _ = rt.accept_bytes(b"PREAMBLE-junk-blah");
        assert!(rt.is_awaiting_trigger());

        // Trigger.
        rt.trigger();
        assert!(!rt.is_awaiting_trigger());

        // Post-trigger: the grammar is intact; the literal matches.
        assert!(rt.accept_bytes(b"abc"));
        assert!(rt.is_accepted());
    }

    /// Multi-tool-call regression guard.  llama.cpp does NOT reset
    /// awaiting_trigger on the close marker — multi-call support comes
    /// from the chat-template-rendered grammar accepting `(call)+`
    /// directly.  This test verifies a `(call)+`-shaped grammar
    /// continues to accept after a complete first call without any
    /// runtime reset.
    ///
    /// See research-report.md Q2 anti-finding: "No production engine
    /// implements parallel-tool-call as a multi-runtime architecture."
    #[test]
    fn multi_tool_call_grammar_continues_across_close_marker() {
        // Two complete calls back-to-back, single grammar.  Models
        // `<call>X</call><call>Y</call>` collapsed to two literals.
        let src = "root ::= call+\ncall ::= \"<call>\" [a-z]+ \"</call>\"\n";
        let mut rt = runtime_from(src, "root");
        // Engine-equivalent: trigger fires once when the splitter sees
        // the first `<call>`; that flip is permanent.
        rt.set_awaiting_trigger(true);
        rt.trigger();

        // First complete call.
        assert!(rt.accept_bytes(b"<call>foo</call>"));
        assert!(
            rt.is_accepted(),
            "after first complete call, kleene + is in an accepting state"
        );

        // Second complete call WITHOUT any reset — proving the runtime
        // carries state through naturally.  This is the canonical
        // pattern from llama.cpp + Hermes 2 Pro template
        // (/opt/llama.cpp/docs/function-calling.md).
        assert!(
            rt.accept_bytes(b"<call>bar</call>"),
            "(call)+ grammar MUST accept a second complete call without runtime reset"
        );
        assert!(rt.is_accepted(), "still accepting after the second call");
    }
}
