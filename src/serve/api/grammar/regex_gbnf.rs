//! Regex → GBNF compiler for the JSON Schema `pattern` keyword
//! (ADR-005 iter-231c).
//!
//! Frontier constraint engines (llguidance, xgrammar) compile regex
//! constraints into the grammar; the peer's json-schema-to-grammar
//! defers `pattern`. Real-world tool schemas (e.g. MCP servers such as
//! ruvnet-brain's `argv` items: `^[a-z][a-z0-9-]*$`) DO use `pattern`,
//! so this module closes the gap with a real compiler instead of a
//! bypass.
//!
//! # Supported subset (everything else is an honest `Err`)
//!
//!   * literal characters (UTF-8), `.`
//!   * char classes `[...]` with ranges + negation; `\d \w \s` expanded
//!     INSIDE classes
//!   * escapes outside classes: `\d \D \w \W \s \S`, control escapes
//!     (`\n \r \t \f \v`), punctuation literals, `\xHH`, `\uHHHH`
//!   * groups `(...)` and `(?:...)` (capturing == non-capturing for
//!     acceptance)
//!   * alternation `|`
//!   * quantifiers `* + ? {n} {n,m} {n,}` (bounded by the GBNF parser's
//!     MAX_REPETITION_THRESHOLD = 2000)
//!   * anchors `^` / `$` — JSON Schema `pattern` is UNANCHORED
//!     ("contains" semantics); anchors are stripped and the body is
//!     wrapped in the surface wildcard per (has-`^`, has-`$`).
//!
//! # Honest `Err` (non-regular or ambiguous; never silently downgraded)
//!
//! backreferences `\1`, zero-width assertions `\b \B \A \z \Z`,
//! look-around `(?= (?! (?<= (?<!`, atomic groups `(?>`, unicode
//! property classes `\p{...}`, complement class-escapes (`\D \W \S`)
//! INSIDE classes, repetition bounds > 2000.
//!
//! # Enforcement level (documented)
//!
//! The pattern is enforced on the RAW string text the model emits —
//! i.e. between the JSON quotes (Qwen nested), the `<|"|>` markers
//! (Gemma), or the XML tags (Qwen top-level raw). JSON escape sequences
//! the model could theoretically emit (`\u0061` for `a`) do NOT satisfy
//! pattern classes; models emit plain text for the ASCII-identifier
//! patterns used in practice. Negated classes and `.` are additionally
//! intersected with the surface's structurally-forbidden bytes (JSON
//! `"` `\` and the close-tag first byte `<`) so a pattern can never
//! swallow the string terminator — same conservative contract as the
//! scalar string rules.

use std::fmt;

/// String surface the compiled pattern is embedded into — determines
/// which bytes are structurally forbidden inside wildcard expansions
/// (negated classes, `.`, contains-wrappers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    /// Ordinary JSON string content. Quotes, backslashes, and raw controls
    /// are structural; `<` is ordinary data (unlike tool-marker surfaces).
    JsonString,
    /// Qwen nested JSON string: content between `"..."` — `"` and `\`
    /// (JSON syntax) and `<` (the `</parameter>` first byte) are
    /// forbidden; raw control bytes are invalid JSON.
    QwenJsonString,
    /// Qwen top-level raw string: content between XML tags — `<`
    /// forbidden (close-tag first byte).
    QwenRawString,
    /// DeepSeek DSML raw string. The runtime keeps the close-tag parse alive
    /// while `<` is ambiguous, so source text may contain angle brackets.
    DeepSeekRawString,
    /// Gemma marker string: content between `<|"|>...<|"|>` — `<`
    /// forbidden (marker first byte).
    GemmaMarkerString,
}

/// Structured error from the regex compiler. Mapped to
/// `EmitterError::UnsupportedSchemaFeature` at the registry call sites.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegexError(pub String);

impl fmt::Display for RegexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// GBNF parser repetition cap (parser.rs MAX_REPETITION_THRESHOLD).
const MAX_REPEAT: u64 = 2000;

/// Compile a JSON Schema `pattern` regex into a GBNF expression for the
/// string CONTENT (no surrounding quotes/markers — the caller wraps).
pub fn regex_to_gbnf_body(pattern: &str, surface: Surface) -> Result<String, RegexError> {
    let chars: Vec<char> = pattern.chars().collect();
    let anchored_start = chars.first() == Some(&'^');
    let anchored_end = chars.last() == Some(&'$') && chars.len() > (anchored_start as usize);
    let start = if anchored_start { 1 } else { 0 };
    let end = if anchored_end {
        chars.len() - 1
    } else {
        chars.len()
    };
    let mut p = Parser {
        chars: &chars[start..end],
        pos: 0,
        surface,
    };
    let body = p.parse_alternation()?;
    if p.pos != p.chars.len() {
        return Err(RegexError(format!(
            "unexpected trailing input at offset {}",
            p.pos
        )));
    }
    // Anchoring: JSON Schema pattern is "contains" unless anchored.
    let dot = dot_class(surface);
    match (anchored_start, anchored_end) {
        (true, true) => Ok(body),
        (true, false) => Ok(format!("{} {}*", body, dot)),
        (false, true) => Ok(format!("{}* {}", dot, body)),
        (false, false) => Ok(format!("{}* {} {}*", dot, body, dot)),
    }
}

/// The `.` expansion for a surface — regex `.` is "any char except
/// linefeed", intersected with the surface's structural forbiddens.
fn dot_class(surface: Surface) -> &'static str {
    match surface {
        Surface::JsonString => r#"[^"\\\n\x00-\x1F]"#,
        Surface::QwenJsonString => r#"[^"\\<\n\x00-\x1F]"#,
        Surface::QwenRawString => "[^<\n]",
        Surface::DeepSeekRawString => "[^\n]",
        Surface::GemmaMarkerString => "[^<\n]",
    }
}

struct Parser<'a> {
    chars: &'a [char],
    pos: usize,
    surface: Surface,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }
    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }
    fn expect(&mut self, c: char) -> Result<(), RegexError> {
        match self.bump() {
            Some(x) if x == c => Ok(()),
            _ => Err(RegexError(format!(
                "expected '{}' at offset {}",
                c, self.pos
            ))),
        }
    }

    /// alternation := concat ( '|' concat )*
    fn parse_alternation(&mut self) -> Result<String, RegexError> {
        let mut alts = vec![self.parse_concatenation()?];
        while self.peek() == Some('|') {
            self.pos += 1;
            alts.push(self.parse_concatenation()?);
        }
        if alts.len() == 1 {
            Ok(alts.pop().unwrap())
        } else {
            Ok(format!("( {} )", alts.join(" | ")))
        }
    }

    /// concatenation := repeat*
    fn parse_concatenation(&mut self) -> Result<String, RegexError> {
        let mut parts: Vec<String> = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            parts.push(self.parse_repeat()?);
        }
        if parts.is_empty() {
            // Empty alternative (e.g. `(a|)`) — matches empty string.
            return Ok(r#"""""#.to_string());
        }
        Ok(parts.join(" "))
    }

    /// repeat := atom quantifier*
    fn parse_repeat(&mut self) -> Result<String, RegexError> {
        let mut atom = self.parse_atom()?;
        loop {
            match self.peek() {
                Some('*') => {
                    self.pos += 1;
                    atom = format!("{}*", atom);
                }
                Some('+') => {
                    self.pos += 1;
                    atom = format!("{}+", atom);
                }
                Some('?') => {
                    self.pos += 1;
                    atom = format!("{}?", atom);
                }
                Some('{') => {
                    self.pos += 1;
                    let min = self.parse_usize()?;
                    let max = if self.peek() == Some(',') {
                        self.pos += 1;
                        if self.peek() == Some('}') {
                            None
                        } else {
                            Some(self.parse_usize()?)
                        }
                    } else {
                        Some(min)
                    };
                    self.expect('}')?;
                    for b in [Some(min), max].into_iter().flatten() {
                        if b > MAX_REPEAT {
                            return Err(RegexError(format!(
                                "repetition bound {} exceeds {}",
                                b, MAX_REPEAT
                            )));
                        }
                    }
                    if let Some(m) = max {
                        if m < min {
                            return Err(RegexError(format!(
                                "repetition {{{},{}}} has max < min",
                                min, m
                            )));
                        }
                    }
                    atom = match max {
                        Some(m) => format!("{}{{{},{}}}", atom, min, m),
                        None => format!("{}{{{},}}", atom, min),
                    };
                }
                _ => break,
            }
        }
        Ok(atom)
    }

    fn parse_usize(&mut self) -> Result<u64, RegexError> {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.pos += 1;
        }
        if start == self.pos {
            return Err(RegexError(format!(
                "expected integer at offset {}",
                self.pos
            )));
        }
        self.chars[start..self.pos]
            .iter()
            .collect::<String>()
            .parse()
            .map_err(|_| RegexError(format!("invalid integer at offset {}", start)))
    }

    /// atom := literal | escape | class | group | '.'
    fn parse_atom(&mut self) -> Result<String, RegexError> {
        match self.peek() {
            Some('(') => {
                self.pos += 1;
                // Group prefixes: only `(?:` (non-capturing) is
                // accepted; capturing `(` is handled identically.
                if self.peek() == Some('?') {
                    self.pos += 1;
                    match self.peek() {
                        Some(':') => {
                            self.pos += 1;
                        }
                        Some('=') | Some('!') => {
                            return Err(RegexError("lookahead is not regular".into()))
                        }
                        Some('<') => return Err(RegexError("lookbehind is not regular".into())),
                        Some('>') => return Err(RegexError("atomic groups unsupported".into())),
                        _ => {
                            return Err(RegexError(format!(
                                "unsupported group prefix '(?{}'",
                                self.peek().unwrap_or(' ')
                            )))
                        }
                    }
                }
                let body = self.parse_alternation()?;
                self.expect(')')?;
                Ok(format!("( {} )", body))
            }
            Some('[') => self.parse_class(),
            Some('.') => {
                self.pos += 1;
                Ok(dot_class(self.surface).to_string())
            }
            Some('\\') => self.parse_escape(false),
            Some(c) => {
                self.pos += 1;
                Ok(gbnf_literal_char(c))
            }
            None => Err(RegexError("unexpected end of pattern".into())),
        }
    }

    /// Parse an escape sequence (the leading `\` is current).
    /// `in_class` selects class-context rules (literal emission without
    /// GBNF-element wrapping).
    fn parse_escape(&mut self, in_class: bool) -> Result<String, RegexError> {
        self.expect('\\')?;
        let c = self
            .bump()
            .ok_or_else(|| RegexError("dangling escape at end".into()))?;
        match c {
            'd' => Ok("[0-9]".to_string()),
            'w' => Ok("[a-zA-Z0-9_]".to_string()),
            's' => Ok("[ \\t\\n\\r]".to_string()),
            'D' => Ok(self.negated(&[('0', Some('9'))], in_class)),
            'W' => Ok(self.negated_w(in_class)),
            'S' => Ok(self.negated_s(in_class)),
            'n' => Ok(self.lit_or_class_char('\n', in_class)),
            'r' => Ok(self.lit_or_class_char('\r', in_class)),
            't' => Ok(self.lit_or_class_char('\t', in_class)),
            'f' => Ok(self.lit_or_class_char('\x0C', in_class)),
            'v' => Ok(self.lit_or_class_char('\x0B', in_class)),
            'x' => {
                let h = self.take_hex(2)?;
                Ok(self.lit_or_class_char(h, in_class))
            }
            'u' => {
                let h = self.take_hex(4)?;
                Ok(self.lit_or_class_char(h, in_class))
            }
            '1'..='9' => Err(RegexError(format!("backreference \\{} is not regular", c))),
            'b' | 'B' | 'A' | 'z' | 'Z' => Err(RegexError(format!(
                "zero-width assertion \\{} unsupported",
                c
            ))),
            'p' | 'P' => Err(RegexError(
                "unicode property classes \\p{...} unsupported".into(),
            )),
            // Any other escaped char is its literal self (regex lenient
            // escape rule: `\/` = `/`, `\-` = `-`, `\.` = `.`, …).
            other => Ok(self.lit_or_class_char(other, in_class)),
        }
    }

    fn take_hex(&mut self, n: usize) -> Result<char, RegexError> {
        let mut v: u32 = 0;
        for _ in 0..n {
            let c = self
                .bump()
                .ok_or_else(|| RegexError("truncated hex escape".into()))?;
            let d = c
                .to_digit(16)
                .ok_or_else(|| RegexError(format!("bad hex digit '{}'", c)))?;
            v = v * 16 + d;
        }
        char::from_u32(v).ok_or_else(|| RegexError(format!("bad code point U+{:X}", v)))
    }

    /// Literal char emission: outside a class produce a quoted GBNF
    /// literal; inside a class produce the raw class item.
    fn lit_or_class_char(&self, c: char, in_class: bool) -> String {
        if in_class {
            class_item_char(c)
        } else {
            gbnf_literal_char(c)
        }
    }

    /// Negated digit class `\D` (also used for `\W`/`\S` helpers).
    fn negated(&self, ranges: &[(char, Option<char>)], _in_class: bool) -> String {
        let mut items = String::new();
        for (lo, hi) in ranges {
            items.push_str(&class_item_char(*lo));
            if let Some(hi) = hi {
                items.push('-');
                items.push_str(&class_item_char(*hi));
            }
        }
        self.negated_class(&items)
    }

    fn negated_w(&self, _in_class: bool) -> String {
        self.negated_class("a-zA-Z0-9_")
    }

    fn negated_s(&self, _in_class: bool) -> String {
        self.negated_class(" \\t\\n\\r")
    }

    /// Wrap class items in a NEGATED GBNF class, adding the surface's
    /// structurally-forbidden bytes to the negation set so the wildcard
    /// can never swallow the string terminator.
    fn negated_class(&self, items: &str) -> String {
        match self.surface {
            Surface::JsonString => format!("[^{}\"\\\\\\x00-\\x1F]", items),
            Surface::QwenJsonString => format!("[^{}\"\\\\<\\x00-\\x1F]", items),
            Surface::QwenRawString | Surface::GemmaMarkerString => {
                format!("[^{}<]", items)
            }
            Surface::DeepSeekRawString => format!("[^{}]", items),
        }
    }

    /// class := '[' '^'? class_item* ']'
    fn parse_class(&mut self) -> Result<String, RegexError> {
        self.expect('[')?;
        let negated = if self.peek() == Some('^') {
            self.pos += 1;
            true
        } else {
            false
        };
        let mut items = String::new();
        let mut first = true;
        loop {
            let c = self
                .peek()
                .ok_or_else(|| RegexError("unterminated char class".into()))?;
            if c == ']' && !first {
                self.pos += 1;
                break;
            }
            first = false;
            // One class item: literal, escape, or expansion.  Track
            // whether the item is a SINGLE char — only single chars may
            // start a range (`\d-x` must NOT become `0-9-x`).
            let (lo, lo_single): (String, bool) = if c == '\\' {
                // \d \w \s expand INSIDE classes; \D \W \S are honest
                // errors here (complement semantics inside a class are
                // ambiguous to intersect with surface forbiddens).
                match self.chars.get(self.pos + 1) {
                    Some('d') => {
                        self.pos += 2;
                        ("0-9".to_string(), false)
                    }
                    Some('w') => {
                        self.pos += 2;
                        ("a-zA-Z0-9_".to_string(), false)
                    }
                    Some('s') => {
                        self.pos += 2;
                        (" \\t\\n\\r".to_string(), false)
                    }
                    Some('D') | Some('W') | Some('S') => {
                        return Err(RegexError(
                            "complement class-escapes (\\D \\W \\S) inside classes unsupported"
                                .into(),
                        ))
                    }
                    _ => (self.parse_escape(true)?, true),
                }
            } else {
                self.pos += 1;
                (class_item_char(c), true)
            };
            items.push_str(&lo);
            // Range: `a-z` — `-` is literal at class start/end, and may
            // only follow a single-char item.
            if lo_single
                && self.peek() == Some('-')
                && self.chars.get(self.pos + 1).copied().unwrap_or(']') != ']'
            {
                self.pos += 1; // consume '-'
                let hc = self
                    .bump()
                    .ok_or_else(|| RegexError("unterminated class range".into()))?;
                let hi = if hc == '\\' {
                    self.parse_escape(true)?
                } else {
                    class_item_char(hc)
                };
                items.push('-');
                items.push_str(&hi);
            }
        }
        if items.is_empty() && !negated {
            return Err(RegexError("empty char class".into()));
        }
        if negated {
            Ok(self.negated_class(&items))
        } else {
            Ok(format!("[{}]", items))
        }
    }
}

/// GBNF literal for a single char — `"c"` with `"` and `\` escaped.
fn gbnf_literal_char(c: char) -> String {
    match c {
        '"' => r#"\""#.to_string(),
        '\\' => r#"\\"#.to_string(),
        '\n' => r#"\n"#.to_string(),
        '\r' => r#"\r"#.to_string(),
        '\t' => r#"\t"#.to_string(),
        other => format!("\"{}\"", other),
    }
}

/// Raw char inside a GBNF class — `]` and `\` escaped; controls as
/// escapes.
fn class_item_char(c: char) -> String {
    match c {
        ']' => r"\]".to_string(),
        '\\' => r"\\".to_string(),
        '\n' => r"\n".to_string(),
        '\r' => r"\r".to_string(),
        '\t' => r"\t".to_string(),
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn compile(pattern: &str) -> String {
        regex_to_gbnf_body(pattern, Surface::QwenJsonString).expect("compile")
    }

    #[test]
    fn anchored_identifier_pattern_compiles() {
        // The ruvnet-brain `argv` items pattern — the iter-231c driver.
        let body = compile("^[a-z][a-z0-9-]*$");
        assert_eq!(body, r#"[a-z] [a-z0-9-]*"#);
    }

    #[test]
    fn unanchored_pattern_gets_contains_wrappers() {
        let body = compile("foo");
        assert!(
            body.starts_with(r#"[^"\\<\n\x00-\x1F]* "#),
            "prefix wrapper: {}",
            body
        );
        assert!(
            body.ends_with(r#" [^"\\<\n\x00-\x1F]*"#),
            "suffix wrapper: {}",
            body
        );
        assert!(body.contains(r#""f" "o" "o""#));
    }

    #[test]
    fn quantifiers_compile_verbatim() {
        assert_eq!(compile(r"^\d{4}$"), r#"[0-9]{4,4}"#);
        assert_eq!(compile(r"^\d{2,4}$"), r#"[0-9]{2,4}"#);
        assert_eq!(compile(r"^\d{2,}$"), r#"[0-9]{2,}"#);
        assert_eq!(compile(r"^a+$"), r#""a"+"#);
        assert_eq!(compile(r"^ab?$"), r#""a" "b"?"#);
    }

    #[test]
    fn alternation_and_groups_compile() {
        // Groups containing alternation are double-parenthesized (inner
        // from parse_alternation, outer from the group) — semantically
        // identical in GBNF, kept for emitter simplicity.
        assert_eq!(
            compile("^(get|post)$"),
            r#"( ( "g" "e" "t" | "p" "o" "s" "t" ) )"#
        );
        assert_eq!(compile("^(?:ab|cd)x$"), r#"( ( "a" "b" | "c" "d" ) ) "x""#);
    }

    #[test]
    fn escapes_compile() {
        assert_eq!(compile(r"^\w+$"), r#"[a-zA-Z0-9_]+"#);
        assert_eq!(compile(r"^\s$"), r#"[ \t\n\r]"#);
        assert_eq!(compile(r"^\$$"), r#""$""#);
    }

    #[test]
    fn negated_class_adds_surface_forbiddens() {
        // `[^0-9]` must ALSO exclude `"` `\` `<` + controls on the JSON surface.
        assert_eq!(compile(r"^[^0-9]$"), r#"[^0-9"\\<\x00-\x1F]"#);
    }

    #[test]
    fn dot_expansion_excludes_quote_backslash_lt() {
        assert_eq!(compile("^.$"), dot_class(Surface::QwenJsonString));
        assert_eq!(
            regex_to_gbnf_body("^.$", Surface::GemmaMarkerString).unwrap(),
            "[^<\n]"
        );
        assert_eq!(
            regex_to_gbnf_body("^.$", Surface::DeepSeekRawString).unwrap(),
            "[^\n]"
        );
    }

    #[test]
    fn nonregular_features_error() {
        assert!(regex_to_gbnf_body(r"^(a)\1$", Surface::QwenJsonString).is_err());
        assert!(regex_to_gbnf_body(r"^a(?=b)$", Surface::QwenJsonString).is_err());
        assert!(regex_to_gbnf_body(r"^\bword$", Surface::QwenJsonString).is_err());
        assert!(regex_to_gbnf_body(r"^\p{L}+$", Surface::QwenJsonString).is_err());
    }

    #[test]
    fn oversize_repetition_errors() {
        assert!(regex_to_gbnf_body(r"^a{2001}$", Surface::QwenJsonString).is_err());
        assert!(regex_to_gbnf_body(r"^a{2000}$", Surface::QwenJsonString).is_ok());
    }

    /// End-to-end sanity: the compiled body must PARSE as GBNF.
    #[test]
    fn compiled_bodies_parse_as_gbnf() {
        for pat in [
            r"^[a-z][a-z0-9-]*$",
            r"^\d{4}-\d{2}-\d{2}$",
            r"^(get|post|put|delete)$",
            r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
            r"^[\w.+-]+@[\w-]+\.[\w.]+$",
            r"semver",
            r"^\S+$",
        ] {
            let body = compile(pat);
            let gbnf = format!("root ::= {}\n", body);
            crate::serve::api::grammar::parser::parse(&gbnf)
                .unwrap_or_else(|e| panic!("pattern {:?} → invalid GBNF {:?}: {}", pat, body, e));
        }
    }
}
