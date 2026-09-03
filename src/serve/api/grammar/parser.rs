//! GBNF parser.
//!
//! Input: a `.gbnf` grammar text (e.g. `json.gbnf`).
//! Output: an encoded rule set of `Vec<Vec<GretElement>>` where each rule is
//! a list of elements terminated by `End`, with `Alt` separating
//! alternatives. This binary encoding matches the peer's exactly so the
//! two runtimes can share fixtures.
//!
//! Implementation notes:
//!   - Operates on `&[u8]` via index arithmetic rather than `&str`.
//!   - UTF-8 code points are decoded to `u32` scalars (not Rust `char`); the
//!     parser stores grammar chars as `u32` throughout.
//!   - Repetition expansion (`S{m,n}`, `S*`, `S+`, `S?`) rewrites to
//!     synthesized sub-rules identically to the peer, so a `json.gbnf`
//!     parsed here produces a byte-identical rule set.
//!
//! Errors are recoverable via `Result<Grammar, ParseError>`. `parser::parse`
//! returns the grammar on success and a `ParseError` pointing at the
//! offending byte offset on failure.

use std::collections::HashMap;

/// Grammar-element type. Wire-compatible with the peer's `llama_gretype`:
/// the u8 discriminants match so fixtures cross-compare cleanly.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GretType {
    End = 0,
    Alt = 1,
    RuleRef = 2,
    Char = 3,
    CharNot = 4,
    CharRngUpper = 5,
    CharAlt = 6,
    CharAny = 7,
    Token = 8,
    TokenNot = 9,
}

impl GretType {
    pub fn is_char_element(self) -> bool {
        matches!(
            self,
            GretType::Char
                | GretType::CharNot
                | GretType::CharAlt
                | GretType::CharRngUpper
                | GretType::CharAny
        )
    }
}

/// A single grammar element: a type + a 32-bit value interpreted per type.
/// For `Char*` types `value` is a Unicode code point, for `RuleRef` it is a
/// rule id, and for `Token` / `TokenNot` it is a tokenizer token id. For
/// `End` / `Alt` / `CharAny` it is unused (stored as 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GretElement {
    pub ty: GretType,
    pub value: u32,
}

impl GretElement {
    pub fn new(ty: GretType, value: u32) -> Self {
        Self { ty, value }
    }
}

/// The full parsed grammar: a list of rules (each rule is a Vec of elements
/// ending in `End`, with `Alt` separating alternatives) plus a name → rule-id
/// map for lookup + printing.
#[derive(Debug, Clone, PartialEq)]
pub struct Grammar {
    pub rules: Vec<Vec<GretElement>>,
    pub symbol_ids: HashMap<String, u32>,
}

impl Grammar {
    pub fn rule(&self, id: u32) -> Option<&[GretElement]> {
        self.rules.get(id as usize).map(|v| v.as_slice())
    }

    /// Return the rule id for `name`, or `None` if not defined.
    pub fn rule_id(&self, name: &str) -> Option<u32> {
        self.symbol_ids.get(name).copied()
    }

    pub fn rule_name(&self, id: u32) -> Option<&str> {
        self.symbol_ids
            .iter()
            .find(|(_, v)| **v == id)
            .map(|(k, _)| k.as_str())
    }
}

/// Parse error with byte offset into the original grammar string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub offset: usize,
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "grammar parse error at byte {}: {}",
            self.offset, self.message
        )
    }
}
impl std::error::Error for ParseError {}

// ---------------------------------------------------------------------------
// Max repetition threshold. Prevents grammars like `a{999999}` from
// exploding.
// ---------------------------------------------------------------------------
const MAX_REPETITION_THRESHOLD: u64 = 2000;

// ---------------------------------------------------------------------------
// Top-level parser
// ---------------------------------------------------------------------------

/// Parse a GBNF grammar string. Returns `Err(ParseError)` on malformed input.
pub fn parse(src: &str) -> Result<Grammar, ParseError> {
    parse_impl(src, None)
}

/// Parse GBNF with a model-bound textual token resolver.
///
/// Numeric `<[id]>` forms bypass the resolver. For textual forms, `resolver`
/// receives the complete bracketed lexeme (for example `<think>`) and must
/// return exactly one tokenizer token id or a diagnostic explaining why the
/// lexeme is not one token. This mirrors llama.cpp's `add_special=false,
/// parse_special=true` single-token requirement without coupling the grammar
/// module to one tokenizer implementation.
pub fn parse_with_token_resolver<F>(src: &str, mut resolver: F) -> Result<Grammar, ParseError>
where
    F: FnMut(&str) -> Result<u32, String>,
{
    parse_impl(src, Some(&mut resolver))
}

/// Resolve token terminals against a concrete Hugging Face tokenizer.
/// Added/special textual tokens remain parseable while BOS/EOS processors are
/// disabled, a textual lexeme that produces zero or multiple ids is rejected,
/// and every resulting numeric token id must exist in this vocabulary.
pub fn parse_with_tokenizer(
    src: &str,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<Grammar, ParseError> {
    let grammar = parse_with_token_resolver(src, |lexeme| {
        let encoding = tokenizer
            .encode(lexeme, false)
            .map_err(|error| error.to_string())?;
        match encoding.get_ids() {
            [token_id] => Ok(*token_id),
            ids => Err(format!(
                "must resolve to exactly one token id, resolved {}",
                ids.len()
            )),
        }
    })?;

    if let Some(invalid) = grammar.rules.iter().flatten().find(|element| {
        matches!(element.ty, GretType::Token | GretType::TokenNot)
            && tokenizer.id_to_token(element.value).is_none()
    }) {
        return Err(ParseError {
            offset: 0,
            message: format!(
                "token id {} is not present in the bound tokenizer vocabulary",
                invalid.value
            ),
        });
    }

    Ok(grammar)
}

fn parse_impl(
    src: &str,
    token_resolver: Option<&mut dyn FnMut(&str) -> Result<u32, String>>,
) -> Result<Grammar, ParseError> {
    let bytes = src.as_bytes();
    let mut state = ParserState {
        bytes,
        token_resolver,
        symbol_ids: HashMap::new(),
        rules: Vec::new(),
    };
    let mut pos = parse_space(bytes, 0, true);
    while pos < bytes.len() && bytes[pos] != 0 {
        pos = state.parse_rule(pos)?;
    }
    // Validate: every referenced rule must be defined (non-empty).
    for (rule_idx, rule) in state.rules.iter().enumerate() {
        if rule.is_empty() {
            // Find name for this rule id:
            let name = state
                .symbol_ids
                .iter()
                .find(|(_, v)| **v as usize == rule_idx)
                .map(|(k, _)| k.as_str())
                .unwrap_or("<anonymous>");
            return Err(ParseError {
                offset: 0,
                message: format!("undefined rule identifier '{}'", name),
            });
        }
        for elem in rule {
            if elem.ty == GretType::RuleRef {
                let rid = elem.value as usize;
                if rid >= state.rules.len() || state.rules[rid].is_empty() {
                    let name = state
                        .symbol_ids
                        .iter()
                        .find(|(_, v)| **v == elem.value)
                        .map(|(k, _)| k.as_str())
                        .unwrap_or("<unknown>");
                    return Err(ParseError {
                        offset: 0,
                        message: format!("undefined rule identifier '{}'", name),
                    });
                }
            }
        }
    }
    validate_no_left_recursion(&state.rules)?;
    Ok(Grammar {
        rules: state.rules,
        symbol_ids: state.symbol_ids,
    })
}

/// Reject grammars whose left-corner graph contains a cycle.
///
/// A reference is a left corner when it can be reached before consuming a
/// terminal. Nullable references therefore expose the next reference in the
/// same alternative. The runtime expands left corners before it consumes a
/// character; a cycle here would otherwise grow its work stack without a
/// progress bound.
fn validate_no_left_recursion(rules: &[Vec<GretElement>]) -> Result<(), ParseError> {
    let mut nullable = vec![false; rules.len()];
    loop {
        let mut changed = false;
        for (rule_id, rule) in rules.iter().enumerate() {
            if !nullable[rule_id] && has_nullable_alternative(rule, &nullable) {
                nullable[rule_id] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut left_corners = vec![Vec::new(); rules.len()];
    for (rule_id, rule) in rules.iter().enumerate() {
        let mut before_terminal = true;
        for element in rule {
            match element.ty {
                GretType::Alt => before_terminal = true,
                GretType::End => break,
                GretType::RuleRef if before_terminal => {
                    let target = element.value as usize;
                    if !left_corners[rule_id].contains(&target) {
                        left_corners[rule_id].push(target);
                    }
                    before_terminal = nullable[target];
                }
                _ => before_terminal = false,
            }
        }
    }

    // Iterative DFS avoids moving the unbounded recursion risk from the
    // grammar runtime into validation itself.
    let mut state = vec![0u8; rules.len()]; // 0 = unseen, 1 = active, 2 = done
    for start in 0..rules.len() {
        if state[start] != 0 {
            continue;
        }
        state[start] = 1;
        let mut work = vec![(start, 0usize)];
        while let Some((rule_id, next_edge)) = work.last_mut() {
            if *next_edge == left_corners[*rule_id].len() {
                state[*rule_id] = 2;
                work.pop();
                continue;
            }
            let target = left_corners[*rule_id][*next_edge];
            *next_edge += 1;
            match state[target] {
                0 => {
                    state[target] = 1;
                    work.push((target, 0));
                }
                1 => {
                    return Err(ParseError {
                        offset: 0,
                        message: "unsupported grammar: left recursion detected".into(),
                    });
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn has_nullable_alternative(rule: &[GretElement], nullable: &[bool]) -> bool {
    let mut alternative_is_nullable = true;
    for element in rule {
        match element.ty {
            GretType::Alt | GretType::End => {
                if alternative_is_nullable {
                    return true;
                }
                if element.ty == GretType::End {
                    return false;
                }
                alternative_is_nullable = true;
            }
            GretType::RuleRef if alternative_is_nullable => {
                alternative_is_nullable = nullable[element.value as usize];
            }
            _ => alternative_is_nullable = false,
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Parser state & internal methods
// ---------------------------------------------------------------------------

struct ParserState<'a, 'r> {
    bytes: &'a [u8],
    token_resolver: Option<&'r mut dyn FnMut(&str) -> Result<u32, String>>,
    symbol_ids: HashMap<String, u32>,
    rules: Vec<Vec<GretElement>>,
}

impl<'a, 'r> ParserState<'a, 'r> {
    fn get_or_create_symbol(&mut self, name: String) -> u32 {
        let next_id = self.symbol_ids.len() as u32;
        let entry = self.symbol_ids.entry(name).or_insert(next_id);
        let id = *entry;
        // Ensure rules vec is long enough to index this id.
        while self.rules.len() <= id as usize {
            self.rules.push(Vec::new());
        }
        id
    }

    fn generate_symbol(&mut self, base: &str) -> u32 {
        // Allocate a synthesized symbol named `{base}_{next}`.
        let next = self.symbol_ids.len() as u32;
        let name = format!("{}_{}", base, next);
        self.get_or_create_symbol(name)
    }

    fn add_rule(&mut self, rule_id: u32, rule: Vec<GretElement>) {
        while self.rules.len() <= rule_id as usize {
            self.rules.push(Vec::new());
        }
        self.rules[rule_id as usize] = rule;
    }

    // --- parse_rule ---

    fn parse_rule(&mut self, src: usize) -> Result<usize, ParseError> {
        let name_start = src;
        let name_end = parse_name(self.bytes, src)?;
        let after_name = parse_space(self.bytes, name_end, false);

        let name: String = std::str::from_utf8(&self.bytes[name_start..name_end])
            .map_err(|_| ParseError {
                offset: name_start,
                message: "rule name not valid UTF-8".into(),
            })?
            .to_string();
        let rule_id = self.get_or_create_symbol(name.clone());

        if after_name + 3 > self.bytes.len() || &self.bytes[after_name..after_name + 3] != b"::=" {
            return Err(ParseError {
                offset: after_name,
                message: "expecting '::='".into(),
            });
        }
        let mut pos = parse_space(self.bytes, after_name + 3, true);

        pos = self.parse_alternates(pos, &name, rule_id, false)?;

        // Optional trailing newline.
        if pos < self.bytes.len() {
            if self.bytes[pos] == b'\r' {
                pos += if pos + 1 < self.bytes.len() && self.bytes[pos + 1] == b'\n' {
                    2
                } else {
                    1
                };
            } else if self.bytes[pos] == b'\n' {
                pos += 1;
            } else if self.bytes[pos] != 0 {
                return Err(ParseError {
                    offset: pos,
                    message: "expecting newline or end".into(),
                });
            }
        }

        Ok(parse_space(self.bytes, pos, true))
    }

    // --- parse_alternates ---

    fn parse_alternates(
        &mut self,
        src: usize,
        rule_name: &str,
        rule_id: u32,
        is_nested: bool,
    ) -> Result<usize, ParseError> {
        let mut rule: Vec<GretElement> = Vec::new();
        let mut pos = self.parse_sequence(src, rule_name, &mut rule, is_nested)?;
        while pos < self.bytes.len() && self.bytes[pos] == b'|' {
            rule.push(GretElement::new(GretType::Alt, 0));
            pos = parse_space(self.bytes, pos + 1, true);
            pos = self.parse_sequence(pos, rule_name, &mut rule, is_nested)?;
        }
        rule.push(GretElement::new(GretType::End, 0));
        self.add_rule(rule_id, rule);
        Ok(pos)
    }

    // --- parse_sequence ---

    fn parse_sequence(
        &mut self,
        mut pos: usize,
        rule_name: &str,
        rule: &mut Vec<GretElement>,
        is_nested: bool,
    ) -> Result<usize, ParseError> {
        let mut last_sym_start: usize = rule.len();
        let mut n_prev_rules: u64 = 1;

        loop {
            if pos >= self.bytes.len() || self.bytes[pos] == 0 {
                break;
            }
            let c = self.bytes[pos];
            if c == b'"' {
                // Literal string: emit one `Char` element per code point.
                pos += 1;
                last_sym_start = rule.len();
                n_prev_rules = 1;
                while pos < self.bytes.len() && self.bytes[pos] != b'"' {
                    if self.bytes[pos] == 0 {
                        return Err(ParseError {
                            offset: pos,
                            message: "unexpected end of input in literal".into(),
                        });
                    }
                    let (cp, next) = parse_char(self.bytes, pos)?;
                    rule.push(GretElement::new(GretType::Char, cp));
                    pos = next;
                }
                if pos >= self.bytes.len() {
                    return Err(ParseError {
                        offset: pos,
                        message: "unterminated literal".into(),
                    });
                }
                pos = parse_space(self.bytes, pos + 1, is_nested);
            } else if c == b'[' {
                // Character class.
                pos += 1;
                let start_type = if pos < self.bytes.len() && self.bytes[pos] == b'^' {
                    pos += 1;
                    GretType::CharNot
                } else {
                    GretType::Char
                };
                last_sym_start = rule.len();
                n_prev_rules = 1;
                while pos < self.bytes.len() && self.bytes[pos] != b']' {
                    if self.bytes[pos] == 0 {
                        return Err(ParseError {
                            offset: pos,
                            message: "unexpected end of input in char class".into(),
                        });
                    }
                    let (cp, next) = parse_char(self.bytes, pos)?;
                    let ty = if last_sym_start < rule.len() {
                        GretType::CharAlt
                    } else {
                        start_type
                    };
                    rule.push(GretElement::new(ty, cp));
                    pos = next;
                    // `-` introducing a range (but not the closing bracket).
                    if pos + 1 < self.bytes.len()
                        && self.bytes[pos] == b'-'
                        && self.bytes[pos + 1] != b']'
                    {
                        if self.bytes[pos + 1] == 0 {
                            return Err(ParseError {
                                offset: pos + 1,
                                message: "unexpected end of input in range".into(),
                            });
                        }
                        let (endcp, nend) = parse_char(self.bytes, pos + 1)?;
                        rule.push(GretElement::new(GretType::CharRngUpper, endcp));
                        pos = nend;
                    }
                }
                if pos >= self.bytes.len() {
                    return Err(ParseError {
                        offset: pos,
                        message: "unterminated char class".into(),
                    });
                }
                pos = parse_space(self.bytes, pos + 1, is_nested);
            } else if c == b'<' || c == b'!' {
                // Token terminal. Numeric token ids do not require a
                // tokenizer/vocabulary binding and are therefore accepted by
                // the standalone parser exactly as llama.cpp accepts them.
                let ty = if c == b'!' {
                    pos += 1;
                    GretType::TokenNot
                } else {
                    GretType::Token
                };
                let (token_id, token_end) = self.parse_token(pos)?;
                last_sym_start = rule.len();
                n_prev_rules = 1;
                rule.push(GretElement::new(ty, token_id));
                pos = parse_space(self.bytes, token_end, is_nested);
            } else if is_word_char(c) {
                // Rule reference.
                let name_end = parse_name(self.bytes, pos)?;
                let name: String = std::str::from_utf8(&self.bytes[pos..name_end])
                    .map_err(|_| ParseError {
                        offset: pos,
                        message: "rule reference not valid UTF-8".into(),
                    })?
                    .to_string();
                let ref_id = self.get_or_create_symbol(name);
                pos = parse_space(self.bytes, name_end, is_nested);
                last_sym_start = rule.len();
                n_prev_rules = 1;
                rule.push(GretElement::new(GretType::RuleRef, ref_id));
            } else if c == b'(' {
                // Grouping: synthesize a sub-rule.
                pos = parse_space(self.bytes, pos + 1, true);
                let n_rules_before = self.symbol_ids.len() as u32;
                let sub_rule_id = self.generate_symbol(rule_name);
                pos = self.parse_alternates(pos, rule_name, sub_rule_id, true)?;
                n_prev_rules =
                    std::cmp::max(1, self.symbol_ids.len() as u32 - n_rules_before) as u64;
                last_sym_start = rule.len();
                rule.push(GretElement::new(GretType::RuleRef, sub_rule_id));
                if pos >= self.bytes.len() || self.bytes[pos] != b')' {
                    return Err(ParseError {
                        offset: pos,
                        message: "expecting ')'".into(),
                    });
                }
                pos = parse_space(self.bytes, pos + 1, is_nested);
            } else if c == b'.' {
                // Any-char.
                last_sym_start = rule.len();
                n_prev_rules = 1;
                rule.push(GretElement::new(GretType::CharAny, 0));
                pos = parse_space(self.bytes, pos + 1, is_nested);
            } else if c == b'*' {
                pos = parse_space(self.bytes, pos + 1, is_nested);
                self.handle_repetitions(
                    rule,
                    rule_name,
                    &mut last_sym_start,
                    &mut n_prev_rules,
                    0,
                    u64::MAX,
                    pos,
                )?;
            } else if c == b'+' {
                pos = parse_space(self.bytes, pos + 1, is_nested);
                self.handle_repetitions(
                    rule,
                    rule_name,
                    &mut last_sym_start,
                    &mut n_prev_rules,
                    1,
                    u64::MAX,
                    pos,
                )?;
            } else if c == b'?' {
                pos = parse_space(self.bytes, pos + 1, is_nested);
                self.handle_repetitions(
                    rule,
                    rule_name,
                    &mut last_sym_start,
                    &mut n_prev_rules,
                    0,
                    1,
                    pos,
                )?;
            } else if c == b'{' {
                pos = parse_space(self.bytes, pos + 1, is_nested);
                if pos >= self.bytes.len() || !is_digit(self.bytes[pos]) {
                    return Err(ParseError {
                        offset: pos,
                        message: "expecting an int in {...}".into(),
                    });
                }
                let int_end = parse_int(self.bytes, pos)?;
                let min_times: u64 = std::str::from_utf8(&self.bytes[pos..int_end])
                    .unwrap_or("0")
                    .parse()
                    .map_err(|_| ParseError {
                        offset: pos,
                        message: "invalid int".into(),
                    })?;
                pos = parse_space(self.bytes, int_end, is_nested);

                let max_times: u64;
                if pos < self.bytes.len() && self.bytes[pos] == b'}' {
                    max_times = min_times;
                    pos = parse_space(self.bytes, pos + 1, is_nested);
                } else if pos < self.bytes.len() && self.bytes[pos] == b',' {
                    pos = parse_space(self.bytes, pos + 1, is_nested);
                    if pos < self.bytes.len() && is_digit(self.bytes[pos]) {
                        let int_end = parse_int(self.bytes, pos)?;
                        max_times = std::str::from_utf8(&self.bytes[pos..int_end])
                            .unwrap_or("0")
                            .parse()
                            .map_err(|_| ParseError {
                                offset: pos,
                                message: "invalid int".into(),
                            })?;
                        pos = parse_space(self.bytes, int_end, is_nested);
                    } else {
                        max_times = u64::MAX;
                    }
                    if pos >= self.bytes.len() || self.bytes[pos] != b'}' {
                        return Err(ParseError {
                            offset: pos,
                            message: "expecting '}' in {n,m}".into(),
                        });
                    }
                    pos = parse_space(self.bytes, pos + 1, is_nested);
                } else {
                    return Err(ParseError {
                        offset: pos,
                        message: "expecting ',' or '}' in repetition".into(),
                    });
                }

                let has_max = max_times != u64::MAX;
                if min_times > MAX_REPETITION_THRESHOLD
                    || (has_max && max_times > MAX_REPETITION_THRESHOLD)
                {
                    return Err(ParseError {
                        offset: pos,
                        message: "number of repetitions exceeds sane defaults".into(),
                    });
                }
                self.handle_repetitions(
                    rule,
                    rule_name,
                    &mut last_sym_start,
                    &mut n_prev_rules,
                    min_times,
                    max_times,
                    pos,
                )?;
            } else {
                break;
            }
        }
        Ok(pos)
    }

    fn parse_token(&mut self, start: usize) -> Result<(u32, usize), ParseError> {
        if self.bytes.get(start) != Some(&b'<') {
            return Err(ParseError {
                offset: start,
                message: "expecting '<' for token terminal".into(),
            });
        }
        if self.bytes.get(start + 1) == Some(&b'[') {
            return parse_explicit_token_id(self.bytes, start);
        }

        let close = self.bytes[start + 1..]
            .iter()
            .position(|byte| *byte == b'>')
            .map(|relative| start + 1 + relative)
            .ok_or_else(|| ParseError {
                offset: self.bytes.len(),
                message: "expecting '>' after textual token".into(),
            })?;
        let lexeme = std::str::from_utf8(&self.bytes[start..=close])
            .map_err(|_| ParseError {
                offset: start,
                message: "textual token terminal is not valid UTF-8".into(),
            })?
            .to_string();
        let Some(resolver) = self.token_resolver.as_mut() else {
            return Err(ParseError {
                offset: start,
                message: "textual token terminal requires a tokenizer resolver".into(),
            });
        };
        let token_id = resolver(&lexeme).map_err(|message| ParseError {
            offset: start,
            message: format!("invalid textual token {lexeme:?}: {message}"),
        })?;
        Ok((token_id, close + 1))
    }

    // --- repetition expansion ---

    /// Rewrite the tail of `rule` (from `*last_sym_start`) as `min_times`
    /// copies followed by a chain of synthesized sub-rules for the `max -
    /// min` optional copies.
    #[allow(clippy::too_many_arguments)]
    fn handle_repetitions(
        &mut self,
        rule: &mut Vec<GretElement>,
        rule_name: &str,
        last_sym_start: &mut usize,
        n_prev_rules: &mut u64,
        min_times: u64,
        max_times: u64,
        pos: usize,
    ) -> Result<(), ParseError> {
        let no_max = max_times == u64::MAX;
        if *last_sym_start == rule.len() {
            return Err(ParseError {
                offset: pos,
                message: "expecting preceding item to */+/?/{".into(),
            });
        }
        let prev_rule: Vec<GretElement> = rule[*last_sym_start..].to_vec();
        let total_rules: u64 = if !no_max && max_times > 0 {
            max_times
        } else if min_times > 0 {
            min_times
        } else {
            1
        };
        if *n_prev_rules * total_rules >= MAX_REPETITION_THRESHOLD {
            return Err(ParseError {
                offset: pos,
                message: "n_prev_rules * total_rules exceeds MAX_REPETITION_THRESHOLD".into(),
            });
        }

        if min_times == 0 {
            rule.truncate(*last_sym_start);
        } else {
            for _ in 1..min_times {
                rule.extend_from_slice(&prev_rule);
            }
        }

        let mut last_rec_rule_id: u32 = 0;
        let n_opt: u64 = if no_max { 1 } else { max_times - min_times };

        let mut rec_rule: Vec<GretElement> = prev_rule.clone();
        for i in 0..n_opt {
            rec_rule.truncate(prev_rule.len());
            let rec_rule_id = self.generate_symbol(rule_name);
            if i > 0 || no_max {
                rec_rule.push(GretElement::new(
                    GretType::RuleRef,
                    if no_max {
                        rec_rule_id
                    } else {
                        last_rec_rule_id
                    },
                ));
            }
            rec_rule.push(GretElement::new(GretType::Alt, 0));
            rec_rule.push(GretElement::new(GretType::End, 0));
            self.add_rule(rec_rule_id, rec_rule.clone());
            last_rec_rule_id = rec_rule_id;
        }
        if n_opt > 0 {
            rule.push(GretElement::new(GretType::RuleRef, last_rec_rule_id));
        }
        *n_prev_rules *= total_rules;
        if *n_prev_rules < 1 {
            *n_prev_rules = 1;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Low-level parse helpers
// ---------------------------------------------------------------------------

fn is_digit(c: u8) -> bool {
    (b'0'..=b'9').contains(&c)
}

fn is_word_char(c: u8) -> bool {
    // The peer's `is_word_char` does NOT include `_`, but its symbol-id
    // generator synthesizes rule names like `root_1` for
    // `?`/`*`/`+`/`{n,m}`/`(...)` expansions.  The peer gets away with the
    // inconsistency because it never round-trips parsed grammars through
    // its own parser — `print` is a debug-only sink there.
    //
    // hf2q's `combine_function_grammars` (handlers.rs) NEEDS the round-trip:
    // parse user grammars → rename via `serialize::rename_rules` → serialize
    // → re-parse the combined output.  Without `_` in the word set, every
    // synthesized subrule name fails to re-parse.  We diverge from the peer
    // by one byte to keep the parser self-consistent with its own emitter.
    // Wave 2.6 W-γ5a (Q4-A) — see /opt/hf2q/src/serve/api/grammar/serialize.rs.
    (b'a'..=b'z').contains(&c)
        || (b'A'..=b'Z').contains(&c)
        || c == b'-'
        || c == b'_'
        || is_digit(c)
}

/// Skip whitespace / comments starting at `pos`. `#` to end-of-line is a
/// comment. `\r\n` only when `newline_ok`.
fn parse_space(bytes: &[u8], mut pos: usize, newline_ok: bool) -> usize {
    while pos < bytes.len() {
        let c = bytes[pos];
        if c == b' ' || c == b'\t' || c == b'#' || (newline_ok && (c == b'\r' || c == b'\n')) {
            if c == b'#' {
                while pos < bytes.len() && bytes[pos] != b'\r' && bytes[pos] != b'\n' {
                    pos += 1;
                }
            } else {
                pos += 1;
            }
        } else {
            break;
        }
    }
    pos
}

fn parse_name(bytes: &[u8], start: usize) -> Result<usize, ParseError> {
    let mut pos = start;
    while pos < bytes.len() && is_word_char(bytes[pos]) {
        pos += 1;
    }
    if pos == start {
        return Err(ParseError {
            offset: start,
            message: "expecting name".into(),
        });
    }
    Ok(pos)
}

fn parse_int(bytes: &[u8], start: usize) -> Result<usize, ParseError> {
    let mut pos = start;
    while pos < bytes.len() && is_digit(bytes[pos]) {
        pos += 1;
    }
    if pos == start {
        return Err(ParseError {
            offset: start,
            message: "expecting integer".into(),
        });
    }
    Ok(pos)
}

/// Parse llama.cpp's vocabulary-independent numeric token form `<[id]>`.
/// Textual `<token>` forms require a tokenizer binding and are intentionally
/// rejected by this standalone parser instead of being guessed from text.
fn parse_explicit_token_id(bytes: &[u8], start: usize) -> Result<(u32, usize), ParseError> {
    if bytes.get(start) != Some(&b'<') {
        return Err(ParseError {
            offset: start,
            message: "expecting '<' for token terminal".into(),
        });
    }
    let id_start = start + 2;
    if bytes.get(start + 1) != Some(&b'[') {
        return Err(ParseError {
            offset: start + 1,
            message: "textual token terminal requires a tokenizer resolver; expecting '[id]'"
                .into(),
        });
    }
    let id_end = parse_int(bytes, id_start)?;
    let id = std::str::from_utf8(&bytes[id_start..id_end])
        .ok()
        .and_then(|text| text.parse::<u32>().ok())
        .ok_or_else(|| ParseError {
            offset: id_start,
            message: "token id is outside the u32 range".into(),
        })?;
    if bytes.get(id_end) != Some(&b']') {
        return Err(ParseError {
            offset: id_end,
            message: "expecting ']' after token id".into(),
        });
    }
    if bytes.get(id_end + 1) != Some(&b'>') {
        return Err(ParseError {
            offset: id_end + 1,
            message: "expecting '>' after token id".into(),
        });
    }
    Ok((id, id_end + 2))
}

/// Parse `N` hex digits starting at `pos`. Returns the decoded value.
fn parse_hex(bytes: &[u8], pos: usize, n: usize) -> Result<(u32, usize), ParseError> {
    let mut value: u32 = 0;
    let mut i = 0;
    while i < n && pos + i < bytes.len() {
        let c = bytes[pos + i];
        value <<= 4;
        if (b'a'..=b'f').contains(&c) {
            value += (c - b'a' + 10) as u32;
        } else if (b'A'..=b'F').contains(&c) {
            value += (c - b'A' + 10) as u32;
        } else if is_digit(c) {
            value += (c - b'0') as u32;
        } else {
            break;
        }
        i += 1;
    }
    if i != n {
        return Err(ParseError {
            offset: pos,
            message: format!("expecting {} hex chars", n),
        });
    }
    Ok((value, pos + n))
}

/// Parse a single character — literal, backslash-escape, or UTF-8 code point.
fn parse_char(bytes: &[u8], pos: usize) -> Result<(u32, usize), ParseError> {
    if pos >= bytes.len() {
        return Err(ParseError {
            offset: pos,
            message: "unexpected end of input".into(),
        });
    }
    if bytes[pos] == b'\\' {
        if pos + 1 >= bytes.len() {
            return Err(ParseError {
                offset: pos,
                message: "unexpected end of input after '\\\\'".into(),
            });
        }
        let c = bytes[pos + 1];
        return match c {
            b'x' => parse_hex(bytes, pos + 2, 2),
            b'u' => parse_hex(bytes, pos + 2, 4),
            b'U' => parse_hex(bytes, pos + 2, 8),
            b't' => Ok(('\t' as u32, pos + 2)),
            b'r' => Ok(('\r' as u32, pos + 2)),
            b'n' => Ok(('\n' as u32, pos + 2)),
            b'\\' | b'"' | b'[' | b']' | b'-' => Ok((c as u32, pos + 2)),
            _ => Err(ParseError {
                offset: pos,
                message: format!("unknown escape '\\{}'", c as char),
            }),
        };
    }
    decode_utf8_one(bytes, pos)
}

/// Decode one UTF-8 code point starting at `pos`. Returns `(code_point,
/// next_pos)`. On invalid/truncated sequences returns an error.
fn decode_utf8_one(bytes: &[u8], pos: usize) -> Result<(u32, usize), ParseError> {
    if pos >= bytes.len() {
        return Err(ParseError {
            offset: pos,
            message: "unexpected end of input".into(),
        });
    }
    let first = bytes[pos];
    let (mut value, n) = if first & 0x80 == 0 {
        (first as u32, 1)
    } else if first & 0xE0 == 0xC0 {
        ((first & 0x1F) as u32, 2)
    } else if first & 0xF0 == 0xE0 {
        ((first & 0x0F) as u32, 3)
    } else if first & 0xF8 == 0xF0 {
        ((first & 0x07) as u32, 4)
    } else {
        return Err(ParseError {
            offset: pos,
            message: "invalid UTF-8 leading byte".into(),
        });
    };
    if pos + n > bytes.len() {
        return Err(ParseError {
            offset: pos,
            message: "truncated UTF-8 sequence".into(),
        });
    }
    for i in 1..n {
        let b = bytes[pos + i];
        if b & 0xC0 != 0x80 {
            return Err(ParseError {
                offset: pos + i,
                message: "invalid UTF-8 continuation byte".into(),
            });
        }
        value = (value << 6) | ((b & 0x3F) as u32);
    }
    Ok((value, pos + n))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(src: &str) -> Grammar {
        match parse(src) {
            Ok(g) => g,
            Err(e) => panic!("unexpected parse error: {} on grammar:\n{}", e, src),
        }
    }

    #[test]
    fn empty_grammar_parses() {
        let g = parse_ok("");
        assert!(g.rules.is_empty());
    }

    #[test]
    fn single_rule_literal() {
        let g = parse_ok("root ::= \"hello\"\n");
        assert_eq!(g.rules.len(), 1);
        let r = g.rules[0].clone();
        // h e l l o + End
        assert_eq!(r.len(), 6);
        assert_eq!(
            r[0..5],
            vec![
                GretElement::new(GretType::Char, 'h' as u32),
                GretElement::new(GretType::Char, 'e' as u32),
                GretElement::new(GretType::Char, 'l' as u32),
                GretElement::new(GretType::Char, 'l' as u32),
                GretElement::new(GretType::Char, 'o' as u32),
            ][..]
        );
        assert_eq!(r[5], GretElement::new(GretType::End, 0));
    }

    #[test]
    fn explicit_token_terminals_match_llama_cpp_encoding() {
        let g = parse_ok("root ::= <[1000]> !<[1001]> <[1001]>\n");
        assert_eq!(
            g.rules[0],
            vec![
                GretElement::new(GretType::Token, 1000),
                GretElement::new(GretType::TokenNot, 1001),
                GretElement::new(GretType::Token, 1001),
                GretElement::new(GretType::End, 0),
            ]
        );
        assert_eq!(GretType::Token as u8, 8);
        assert_eq!(GretType::TokenNot as u8, 9);
    }

    #[test]
    fn explicit_token_terminal_rejects_malformed_ids() {
        for src in [
            "root ::= <[]>\n",
            "root ::= <[x]>\n",
            "root ::= <[10>\n",
            "root ::= <[10]\n",
            "root ::= ! [10]\n",
        ] {
            assert!(parse(src).is_err(), "malformed token parsed: {src:?}");
        }
    }

    #[test]
    fn textual_token_terminals_use_model_bound_resolver() {
        let mut seen = Vec::new();
        let grammar = parse_with_token_resolver(
            "root ::= <think> !</think>\n",
            |lexeme| -> Result<u32, String> {
                seen.push(lexeme.to_string());
                match lexeme {
                    "<think>" => Ok(10),
                    "</think>" => Ok(11),
                    _ => Err(format!("not one tokenizer token: {lexeme}")),
                }
            },
        )
        .expect("resolved textual tokens");

        assert_eq!(seen, vec!["<think>", "</think>"]);
        assert_eq!(
            grammar.rules[0],
            vec![
                GretElement::new(GretType::Token, 10),
                GretElement::new(GretType::TokenNot, 11),
                GretElement::new(GretType::End, 0),
            ]
        );
    }

    #[test]
    fn textual_token_terminal_fails_closed_without_or_from_resolver() {
        let missing = parse("root ::= <think>\n").expect_err("resolver is required");
        assert!(missing.message.contains("resolver"));

        let rejected = parse_with_token_resolver("root ::= <two-tokens>\n", |_lexeme| {
            Err::<u32, _>("must resolve to exactly one token".to_string())
        })
        .expect_err("resolver rejection must surface");
        assert!(rejected.message.contains("exactly one token"));
    }

    #[test]
    fn textual_tokenizer_binding_resolves_registered_special_tokens() {
        use tokenizers::{models::bpe::BPE, AddedToken, Tokenizer};

        let mut tokenizer = Tokenizer::new(BPE::default());
        tokenizer.add_special_tokens(&[
            AddedToken::from("<think>".to_string(), true),
            AddedToken::from("</think>".to_string(), true),
        ]);
        let think = tokenizer.token_to_id("<think>").expect("think id");
        let end_think = tokenizer.token_to_id("</think>").expect("end id");

        let grammar = parse_with_tokenizer("root ::= <think> (!</think>)* </think>\n", &tokenizer)
            .expect("model-bound token parse");
        assert_eq!(
            grammar.rules[0][0],
            GretElement::new(GretType::Token, think)
        );
        assert!(grammar
            .rules
            .iter()
            .flatten()
            .any(|element| { *element == GretElement::new(GretType::TokenNot, end_think) }));
        assert!(grammar
            .rules
            .iter()
            .flatten()
            .any(|element| *element == GretElement::new(GretType::Token, end_think)));

        let numeric = parse_with_tokenizer(
            &format!("root ::= <[{think}]> !<[{end_think}]>\n"),
            &tokenizer,
        )
        .expect("numeric ids present in the model vocabulary");
        assert_eq!(numeric.rules[0][0].value, think);

        let invalid = parse_with_tokenizer("root ::= <[4294967295]>\n", &tokenizer)
            .expect_err("unknown numeric token id must fail before decode");
        assert!(invalid.message.contains("not present"));
    }

    #[test]
    fn alternates_produce_alt_element() {
        let g = parse_ok("root ::= \"a\" | \"b\"\n");
        let r = &g.rules[0];
        // a Alt b End
        assert_eq!(r.len(), 4);
        assert_eq!(r[0], GretElement::new(GretType::Char, 'a' as u32));
        assert_eq!(r[1], GretElement::new(GretType::Alt, 0));
        assert_eq!(r[2], GretElement::new(GretType::Char, 'b' as u32));
        assert_eq!(r[3], GretElement::new(GretType::End, 0));
    }

    #[test]
    fn char_class_with_range() {
        // [a-z] → Char('a'), CharRngUpper('z'), End
        let g = parse_ok("root ::= [a-z]\n");
        let r = &g.rules[0];
        assert_eq!(r[0], GretElement::new(GretType::Char, 'a' as u32));
        assert_eq!(r[1], GretElement::new(GretType::CharRngUpper, 'z' as u32));
        assert_eq!(r[2], GretElement::new(GretType::End, 0));
    }

    #[test]
    fn char_class_negated_with_range() {
        let g = parse_ok("root ::= [^A-Z]\n");
        let r = &g.rules[0];
        assert_eq!(r[0], GretElement::new(GretType::CharNot, 'A' as u32));
        assert_eq!(r[1], GretElement::new(GretType::CharRngUpper, 'Z' as u32));
    }

    #[test]
    fn char_class_with_multiple_alt_chars() {
        // [abc] → Char('a'), CharAlt('b'), CharAlt('c'), End
        let g = parse_ok("root ::= [abc]\n");
        let r = &g.rules[0];
        assert_eq!(r[0], GretElement::new(GretType::Char, 'a' as u32));
        assert_eq!(r[1], GretElement::new(GretType::CharAlt, 'b' as u32));
        assert_eq!(r[2], GretElement::new(GretType::CharAlt, 'c' as u32));
    }

    #[test]
    fn any_char_dot() {
        let g = parse_ok("root ::= .\n");
        let r = &g.rules[0];
        assert_eq!(r[0], GretElement::new(GretType::CharAny, 0));
    }

    #[test]
    fn rule_reference() {
        let g = parse_ok("root ::= ws\nws ::= \" \"\n");
        assert_eq!(g.rule_id("root"), Some(0));
        assert_eq!(g.rule_id("ws"), Some(1));
        assert_eq!(g.rules[0][0], GretElement::new(GretType::RuleRef, 1));
    }

    #[test]
    fn undefined_rule_reference_errors() {
        // `foo` is referenced but never defined — parse rejects it.
        let err = parse("root ::= foo\n").unwrap_err();
        assert!(err.message.contains("undefined"));
    }

    #[test]
    fn grouping_creates_subrule() {
        // (x y) | z — the grouping creates a synthesized subrule.
        let g = parse_ok("root ::= (\"x\" \"y\") | \"z\"\nws ::= \" \"\n");
        // First rule references the synthesized one.
        // root alternatives: (RuleRef → sub), Alt, 'z', End
        let r = &g.rules[0];
        assert_eq!(r[0].ty, GretType::RuleRef);
        assert_eq!(r[1], GretElement::new(GretType::Alt, 0));
        assert_eq!(r[2], GretElement::new(GretType::Char, 'z' as u32));
    }

    #[test]
    fn zero_or_one_repetition_question() {
        // x? — zero-or-one. Expands to `x?_0 := x | ; root := x?_0`.
        let g = parse_ok("root ::= \"a\"?\n");
        // root should end up with just a RuleRef to the synthesized subrule.
        let r = &g.rules[0];
        assert_eq!(r[0].ty, GretType::RuleRef);
    }

    #[test]
    fn one_or_more_plus_expansion() {
        // x+ → x x*  → rewrites via handle_repetitions.
        let g = parse_ok("root ::= \"a\"+\n");
        // root: CHAR('a'), RuleRef(sub), End
        let r = &g.rules[0];
        assert_eq!(r[0], GretElement::new(GretType::Char, 'a' as u32));
        assert_eq!(r[1].ty, GretType::RuleRef);
        assert_eq!(r[2], GretElement::new(GretType::End, 0));
    }

    #[test]
    fn brace_min_only_repetition() {
        // x{3} — exactly 3 copies of x and no synthesized rule.
        let g = parse_ok("root ::= \"a\"{3}\n");
        let r = &g.rules[0];
        assert_eq!(r[0], GretElement::new(GretType::Char, 'a' as u32));
        assert_eq!(r[1], GretElement::new(GretType::Char, 'a' as u32));
        assert_eq!(r[2], GretElement::new(GretType::Char, 'a' as u32));
        assert_eq!(r[3], GretElement::new(GretType::End, 0));
    }

    #[test]
    fn brace_min_max_repetition() {
        // x{0,2} — expands to subrules.
        let g = parse_ok("root ::= \"a\"{0,2}\n");
        let r = &g.rules[0];
        // root just references the chain of synthesized subrules.
        assert_eq!(r[0].ty, GretType::RuleRef);
    }

    #[test]
    fn brace_min_comma_no_max_repetition() {
        // x{2,} — 2 copies + a tail-recursive subrule for the rest.
        let g = parse_ok("root ::= \"a\"{2,}\n");
        let r = &g.rules[0];
        assert_eq!(r[0], GretElement::new(GretType::Char, 'a' as u32));
        assert_eq!(r[1], GretElement::new(GretType::Char, 'a' as u32));
        assert_eq!(r[2].ty, GretType::RuleRef);
    }

    #[test]
    fn escape_sequences() {
        // \n \t \\ \" \x41 B
        let g = parse_ok("root ::= \"\\n\\t\\\\\\\"\\x41\\u0042\"\n");
        let r = &g.rules[0];
        assert_eq!(r[0], GretElement::new(GretType::Char, '\n' as u32));
        assert_eq!(r[1], GretElement::new(GretType::Char, '\t' as u32));
        assert_eq!(r[2], GretElement::new(GretType::Char, '\\' as u32));
        assert_eq!(r[3], GretElement::new(GretType::Char, '"' as u32));
        assert_eq!(r[4], GretElement::new(GretType::Char, 0x41));
        assert_eq!(r[5], GretElement::new(GretType::Char, 0x42));
    }

    #[test]
    fn comments_ignored() {
        let g = parse_ok("# comment line\nroot ::= \"x\" # trailing\n");
        assert_eq!(g.rules[0][0], GretElement::new(GretType::Char, 'x' as u32));
    }

    #[test]
    fn multiline_rule_via_grouping() {
        // The peer's multi-line rules only work inside groupings (parse_space
        // inside `(...)` is called with newline_ok=true via the is_nested
        // flag). A raw top-level sequence split across newlines is NOT
        // supported — the first newline terminates the outer parse_sequence.
        // This mirrors json.gbnf's `object ::= "{" ws ( ... ) "}" ws` shape.
        let g = parse_ok("root ::= ( \n  \"x\"\n  \"y\"\n )\n");
        let r = &g.rules[0];
        // Outer rule references the synthesized subrule for the grouping.
        assert_eq!(r[0].ty, GretType::RuleRef);
        // The inner subrule contains both literals.
        let sub_id = r[0].value as usize;
        let sub = &g.rules[sub_id];
        assert_eq!(sub[0], GretElement::new(GretType::Char, 'x' as u32));
        assert_eq!(sub[1], GretElement::new(GretType::Char, 'y' as u32));
    }

    #[test]
    fn all_vendored_llama_cpp_grammars_parse() {
        for (name, src) in super::super::test_fixtures::LLAMA_CPP_GRAMMARS {
            let g = parse_ok(src);
            assert!(!g.rules.is_empty(), "{name} produced no rules");
            assert!(g.rule_id("root").is_some(), "{name} has no root rule");
        }
    }

    #[test]
    fn escaped_dash_is_a_literal_character_in_a_class() {
        let g = parse_ok(r#"root ::= [a\-z]"#);
        let rule = &g.rules[0];
        assert_eq!(rule[0], GretElement::new(GretType::Char, 'a' as u32));
        assert_eq!(rule[1], GretElement::new(GretType::CharAlt, '-' as u32));
        assert_eq!(rule[2], GretElement::new(GretType::CharAlt, 'z' as u32));
        assert_eq!(rule[3], GretElement::new(GretType::End, 0));
    }

    #[test]
    fn rejects_direct_left_recursion() {
        let err = parse("root ::= \"a\" | root \"a\"\n").unwrap_err();
        assert!(err.message.contains("left recursion"));
    }

    #[test]
    fn rejects_indirect_left_recursion() {
        let err = parse(
            "root ::= asdf\n\
             asdf ::= \"a\" | foo \"b\"\n\
             foo ::= \"c\" | asdf \"d\"\n",
        )
        .unwrap_err();
        assert!(err.message.contains("left recursion"));
    }

    #[test]
    fn rejects_left_recursion_after_nullable_prefix() {
        let err = parse(
            "root ::= maybe loop | \"x\"\n\
             maybe ::= | \"m\"\n\
             loop ::= root \"y\"\n",
        )
        .unwrap_err();
        assert!(err.message.contains("left recursion"));
    }

    #[test]
    fn accepts_right_recursion_and_terminal_guarded_cycles() {
        parse_ok(
            "root ::= \"a\" root | guarded\n\
             guarded ::= \"g\" tail\n\
             tail ::= \"t\" guarded |\n",
        );
    }

    #[test]
    fn missing_colon_colon_equals_errors() {
        let err = parse("root  \"x\"\n").unwrap_err();
        assert!(err.message.contains("::=") || err.message.contains("'::='"));
    }

    #[test]
    fn unterminated_literal_errors() {
        let err = parse("root ::= \"abc\n").unwrap_err();
        assert!(
            err.message.to_lowercase().contains("unterminated")
                || err.message.to_lowercase().contains("unexpected end")
        );
    }

    #[test]
    fn invalid_escape_errors() {
        let err = parse("root ::= \"\\q\"\n").unwrap_err();
        assert!(err.message.contains("unknown escape"));
    }

    #[test]
    fn utf8_literal_decoded() {
        // Greek alpha (U+03B1) is `\xCE\xB1` in UTF-8.
        let g = parse_ok("root ::= \"α\"\n");
        let r = &g.rules[0];
        assert_eq!(r[0], GretElement::new(GretType::Char, 0x3B1));
    }

    #[test]
    fn rule_with_empty_body_accepted_as_always_match_empty() {
        // The peer accepts empty rule bodies — they match the empty string.
        // This documents parity with it.
        let g = parse_ok("root ::= \n");
        assert_eq!(g.rules.len(), 1);
        assert_eq!(g.rules[0], vec![GretElement::new(GretType::End, 0)]);
    }
}
