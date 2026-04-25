//! nomic-bert tokenizer.
//!
//! nomic-bert GGUFs ship the same WordPiece (WPM) vocab format that
//! BERT uses — `tokenizer.ggml.model = "bert"`, `▁`-prefixed
//! word-starter pieces, bare continuation pieces (per llama.cpp's
//! `llm_tokenizer_wpm_session::tokenize`). The
//! `bert::tokenizer::BertWpmTokenizer` is a byte-for-byte port of that
//! routine and is reused as-is.
//!
//! This module exists as a thin re-export so the nomic_bert lane has a
//! self-contained `tokenizer::*` namespace (parallel to
//! `bert::tokenizer::*`) — handler code shouldn't need to know that
//! nomic delegates to BERT internally.

#![allow(dead_code)]

use std::path::Path;

use anyhow::{anyhow, Result};
use mlx_native::gguf::GgufFile;

pub use crate::inference::models::bert::tokenizer::{BertVocab, BertWpmTokenizer};

/// Build a `BertWpmTokenizer` for a nomic-bert GGUF. Opens the GGUF,
/// pulls the WPM vocab via `BertVocab::from_gguf`, and constructs a
/// `BertWpmTokenizer` over it.
///
/// Errors when the file can't be opened, the GGUF lacks the
/// `tokenizer.ggml.*` keys the BERT-family loader expects, or the
/// special-token resolver can't find `[CLS]`/`[SEP]`. Forwarding the
/// underlying error verbatim so the failure surface matches the bge /
/// mxbai lane.
pub fn build_nomic_wordpiece_tokenizer(gguf_path: &Path) -> Result<BertWpmTokenizer> {
    let gguf = GgufFile::open(gguf_path)
        .map_err(|e| anyhow!("open nomic GGUF '{}': {e}", gguf_path.display()))?;
    let vocab = BertVocab::from_gguf(&gguf)
        .map_err(|e| anyhow!("parse nomic-bert WPM vocab from GGUF: {e}"))?;
    Ok(BertWpmTokenizer::new(&vocab))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Confirm that the nomic-bert GGUF tokenizes the same canonical
    /// "hello world" prompt to the same `[CLS] hello world [SEP]` token
    /// pattern that BERT does. The vocab IDs differ (different vocab
    /// table) but the structural pattern (length 4, bracketed by
    /// special tokens, `add_special_tokens=true`) is identical.
    #[test]
    fn nomic_gguf_tokenizes_hello_world_with_special_tokens() {
        let path = Path::new("/opt/hf2q/models/bert-test/nomic-embed-text-v1.5-f16.gguf");
        if !path.exists() {
            eprintln!("skipping: nomic GGUF fixture not at {}", path.display());
            return;
        }
        let tok = build_nomic_wordpiece_tokenizer(path).expect("build tokenizer");
        let ids = tok.encode("hello world", true);
        // Structural lock: 4 tokens with [CLS] and [SEP] bracketing.
        assert_eq!(ids.len(), 4, "expected 4 tokens for [CLS] hello world [SEP], got {ids:?}");
        // First and last are special tokens — vocab-IDs depend on the
        // model but [CLS]=101 and [SEP]=102 are the BERT-family
        // convention nomic inherits. Locking those values catches a
        // wrong-special-tokens regression.
        assert_eq!(ids.first(), Some(&101u32), "expected [CLS]=101, got {ids:?}");
        assert_eq!(ids.last(), Some(&102u32), "expected [SEP]=102, got {ids:?}");
    }
}
