//! Private HF-generated DSML terminal semantics, not a general GBNF extension.
//! The compact scalar-length grammar remains authoritative. Only data-terminal
//! branches may not consume the final scalar of the raw parameter close marker.
use super::parser::{parse, Grammar, GretType};
use super::sampler::GrammarCandidate;
use super::sampler::Pos;
const CLOSE: &[u32] = &[
    '<' as u32,
    '/' as u32,
    '｜' as u32,
    'D' as u32,
    'S' as u32,
    'M' as u32,
    'L' as u32,
    '｜' as u32,
    'p' as u32,
    'a' as u32,
    'r' as u32,
    'a' as u32,
    'm' as u32,
    'e' as u32,
    't' as u32,
    'e' as u32,
    'r' as u32,
    '>' as u32,
];
pub(crate) const RAW_NAME: &str = "hf2q_dsml_data_raw";
pub(crate) const JSON_NAME: &str = "hf2q_dsml_json_raw";
pub(crate) const END_NAME: &str = "hf2q_dsml_raw_end";
pub(crate) const END_BODY: &str = "\"\"";
pub(crate) const RAW_BODY: &str = ".";
pub(crate) const JSON_BODY: &str = r#"[^"\\\x00-\x1F]"#;
#[derive(Debug)]
pub(super) struct Guard {
    rules: Vec<u8>,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub(super) struct Scan {
    prefix: u8,
    escaped: bool,
}
impl Scan {
    pub(super) fn advance(self, ch: u32) -> Self {
        Self {
            prefix: Guard::advance(self.prefix, ch).0,
            escaped: ch == u32::from('\\') && !self.escaped,
        }
    }
    pub(super) fn through(self, chars: &[u32]) -> Self {
        chars.iter().fold(self, |s, &ch| s.advance(ch))
    }
}
fn kind(name: &str) -> Result<Option<&'static str>, ()> {
    let mut local = name;
    loop {
        if local == END_NAME {
            return Ok(Some(END_BODY));
        }
        if local == RAW_NAME {
            return Ok(Some(RAW_BODY));
        }
        if local == JSON_NAME {
            return Ok(Some(JSON_BODY));
        }
        let rest = local
            .strip_prefix("fn-")
            .or_else(|| local.strip_prefix("variant-"));
        let Some(rest) = rest else {
            break;
        };
        let Some((index, tail)) = rest.split_once('-') else {
            break;
        };
        if index.is_empty() || !index.bytes().all(|b| b.is_ascii_digit()) {
            break;
        }
        local = tail;
    }
    // Private semantics must never silently vanish through an unknown rename.
    if name.ends_with("-hf2q_dsml_raw_end")
        || name.ends_with("-hf2q_dsml_data_raw")
        || name.ends_with("-hf2q_dsml_json_raw")
    {
        Err(())
    } else {
        Ok(None)
    }
}

impl Guard {
    /// Reserved names require their exact terminal-only body. A generated-name
    /// collision is a refused runtime, never a differently guarded language.
    pub(super) fn from_grammar(grammar: &Grammar) -> Result<Option<Self>, ()> {
        let mut rules = Vec::new();
        for (name, &id) in &grammar.symbol_ids {
            if let Some(body) = kind(name)? {
                let expected = parse(&format!("root ::= {body}\n")).map_err(|_| ())?;
                let root = expected.rule_id("root").ok_or(())?;
                if grammar.rules.get(id as usize) != expected.rules.get(root as usize) {
                    return Err(());
                }
                if rules.is_empty() {
                    rules.resize(grammar.rules.len(), 0);
                }
                rules[id as usize] = if body == RAW_BODY {
                    1
                } else if body == JSON_BODY {
                    2
                } else {
                    3
                };
            }
        }
        // The private generated DSML capability is character-only. Refuse
        // mixed token terminals, whose atomic transition is not this lexical
        // surface; ordinary token grammars without the guard are unchanged.
        if !rules.is_empty()
            && grammar.rules.iter().flatten().any(|e| {
                matches!(
                    e.ty,
                    GretType::Token
                        | GretType::TokenNot
                        | GretType::TokenAny
                        | GretType::TokenNotSet
                        | GretType::TokenSetMember
                )
            })
        {
            return Err(());
        }
        Ok((!rules.is_empty()).then_some(Self { rules }))
    }
    pub(super) fn allows(&self, _grammar: &Grammar, pos: Pos, ch: u32, scan: Scan) -> bool {
        let kind = self.rules.get(pos.rule_id as usize).copied().unwrap_or(0);
        if kind != 0 && Self::advance(scan.prefix, ch).1 {
            return false;
        }
        if kind == 1 && scan.escaped && ch < 0x20 {
            return false;
        }
        true
    }
    /// The epsilon predicate belongs only to its generic RAW value branch.
    /// Literal enum/const branches never traverse it, even in mixed grammars.
    pub(super) fn allows_rule(&self, rule: u32, scan: Scan) -> bool {
        self.rules.get(rule as usize).copied() != Some(3) || !scan.escaped
    }
    /// Marker has a unique '<', so mismatch can only restart at that scalar.
    pub(super) fn advance(prefix: u8, ch: u32) -> (u8, bool) {
        let n = prefix as usize;
        if CLOSE.get(n) == Some(&ch) {
            let next = n + 1;
            if next == CLOSE.len() {
                (0, true)
            } else {
                (next as u8, false)
            }
        } else {
            (u8::from(ch == CLOSE[0]), false)
        }
    }
    /// One O(total candidate scalars) pass, no per-branch escape counters.
    pub(super) fn scans(
        initial: Scan,
        candidates: &[GrammarCandidate],
        chars: &[u32],
    ) -> Vec<Scan> {
        let mut result = vec![Scan::default(); chars.len()];
        for c in candidates {
            let mut state = initial;
            for i in c.cursor..c.end {
                result[i] = state;
                state = state.advance(chars[i]);
            }
        }
        result
    }
}
