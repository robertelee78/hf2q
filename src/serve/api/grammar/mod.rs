//! GBNF (GGML Grammar BNF) grammar stack for grammar-constrained decoding.
//!
//! Ported from llama.cpp's C++ implementation to pure Rust, preserving the
//! exact grammar-element encoding so a llama.cpp `.gbnf` file produces the
//! same rule set under hf2q's parser.
//!
//! Reference files (llama.cpp @ /opt/llama.cpp, as of 2026-04-23):
//!   - `src/llama-grammar.h` — element types, parser + runtime struct shapes.
//!   - `src/llama-grammar.cpp` — parser implementation (~750 LOC) + runtime
//!     sampler (~750 LOC). This iter ports the **parser** only; the sampler
//!     (advance_stack, apply, accept) lands in the next iter alongside the
//!     decode-loop integration.
//!   - `common/json-schema-to-grammar.cpp` — JSON-Schema → GBNF translator
//!     (ports in a subsequent iter).
//!
//! Decision #6 (ADR-005 Phase 2 refinement, 2026-04-23): grammar-constrained
//! decoding obviates post-hoc tool-call parsing. `response_format:
//! {type: "json_object"}` and `{type: "json_schema", ...}` ride this same
//! infrastructure via json-schema → GBNF translation.
//!
//! # Deliberate omissions vs. the llama.cpp parser
//!
//! - **`TOKEN` / `TOKEN_NOT` elements** (`<token>` / `!<[id]>` syntax):
//!   require a vocab at parse time. hf2q's first use case (OpenAI
//!   `response_format`) does not need token-level grammars; we port these
//!   when a concrete use case arises (e.g. tool-choice=required forcing a
//!   specific EOS).
//! - **Trigger patterns / lazy grammars**: a runtime-sampler feature, not
//!   a parser concern. Belongs with the sampler iter.

pub mod parser;

#[allow(unused_imports)]
pub use parser::{parse, Grammar, GretElement, GretType, ParseError};
