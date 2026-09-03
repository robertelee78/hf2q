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
pub use sampler::{GrammarRuntime, PartialUtf8, Pos, Stack, Stacks};
#[allow(unused_imports)]
pub use serialize::{rename_rules, serialize};

#[cfg(test)]
pub(crate) mod test_fixtures {
    pub(crate) const LLAMA_CPP_GRAMMARS: &[(&str, &str)] = &[
        (
            "arithmetic.gbnf",
            include_str!("../../../../tests/fixtures/grammars/llama_cpp/arithmetic.gbnf"),
        ),
        (
            "c.gbnf",
            include_str!("../../../../tests/fixtures/grammars/llama_cpp/c.gbnf"),
        ),
        (
            "chess.gbnf",
            include_str!("../../../../tests/fixtures/grammars/llama_cpp/chess.gbnf"),
        ),
        (
            "english.gbnf",
            include_str!("../../../../tests/fixtures/grammars/llama_cpp/english.gbnf"),
        ),
        (
            "japanese.gbnf",
            include_str!("../../../../tests/fixtures/grammars/llama_cpp/japanese.gbnf"),
        ),
        (
            "json.gbnf",
            include_str!("../../../../tests/fixtures/grammars/llama_cpp/json.gbnf"),
        ),
        (
            "json_arr.gbnf",
            include_str!("../../../../tests/fixtures/grammars/llama_cpp/json_arr.gbnf"),
        ),
        (
            "list.gbnf",
            include_str!("../../../../tests/fixtures/grammars/llama_cpp/list.gbnf"),
        ),
    ];

    pub(crate) fn llama_cpp_grammar(name: &str) -> &'static str {
        LLAMA_CPP_GRAMMARS
            .iter()
            .find_map(|(fixture_name, source)| (*fixture_name == name).then_some(*source))
            .unwrap_or_else(|| panic!("unknown llama.cpp grammar fixture: {name}"))
    }
}
