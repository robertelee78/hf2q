//! GBNF (GGML Grammar BNF) grammar stack for grammar-constrained decoding.
//!
//! Preserves the exact grammar-element encoding so a peer `.gbnf` file
//! produces the same rule set under hf2q's parser.
//!
//! Decision #6 (ADR-005 Phase 2 refinement, 2026-04-23): grammar-constrained
//! decoding obviates post-hoc tool-call parsing. `response_format:
//! {type: "json_object"}` and `{type: "json_schema", ...}` ride this same
//! infrastructure via json-schema → GBNF translation.
//!
//! Numeric token terminals (`<[id]>` / `!<[id]>`) are supported without a
//! vocabulary binding. Textual `<token>` forms use
//! [`parse_with_token_resolver`] so tokenizer-specific resolution remains an
//! explicit model-bound operation. hf2q-local structural extensions use
//! `<[*]>` for any non-EOG token and sorted, deduplicated `!<[id,...]>` for a
//! bounded exclusion set; neither form expands across the model vocabulary.

pub mod json_schema;
pub mod lark;
pub mod mask;
pub mod parser;
pub mod regex_gbnf;
pub mod request;
pub mod sampler;
pub mod serialize;
pub mod structural_tag;

#[allow(unused_imports)]
pub use parser::{
    parse, parse_with_token_resolver, parse_with_tokenizer, Grammar, GretElement, GretType,
    ParseError,
};
#[allow(unused_imports)]
pub use sampler::{
    GrammarRuntime, LazyGrammarConfig, PartialUtf8, Pos, Stack, Stacks,
    MAX_LAZY_TRIGGER_BUFFER_BYTES,
};
#[allow(unused_imports)]
pub use serialize::{rename_rules, serialize};

#[cfg(test)]
pub(crate) mod test_fixtures {
    pub(crate) const PEER_GRAMMARS: &[(&str, &str)] = &[
        (
            "arithmetic.gbnf",
            include_str!("../../../../scripts/fixtures/grammars/peer/arithmetic.gbnf"),
        ),
        (
            "c.gbnf",
            include_str!("../../../../scripts/fixtures/grammars/peer/c.gbnf"),
        ),
        (
            "chess.gbnf",
            include_str!("../../../../scripts/fixtures/grammars/peer/chess.gbnf"),
        ),
        (
            "english.gbnf",
            include_str!("../../../../scripts/fixtures/grammars/peer/english.gbnf"),
        ),
        (
            "japanese.gbnf",
            include_str!("../../../../scripts/fixtures/grammars/peer/japanese.gbnf"),
        ),
        (
            "json.gbnf",
            include_str!("../../../../scripts/fixtures/grammars/peer/json.gbnf"),
        ),
        (
            "json_arr.gbnf",
            include_str!("../../../../scripts/fixtures/grammars/peer/json_arr.gbnf"),
        ),
        (
            "list.gbnf",
            include_str!("../../../../scripts/fixtures/grammars/peer/list.gbnf"),
        ),
    ];

    pub(crate) fn peer_grammar(name: &str) -> &'static str {
        PEER_GRAMMARS
            .iter()
            .find_map(|(fixture_name, source)| (*fixture_name == name).then_some(*source))
            .unwrap_or_else(|| panic!("unknown peer grammar fixture: {name}"))
    }
}
