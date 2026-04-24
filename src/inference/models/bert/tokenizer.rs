//! BERT tokenizer (WordPiece) — load from GGUF metadata.
//!
//! GGUF stores WordPiece vocabulary + special-token IDs in metadata keys
//! following llama.cpp's convention (`tokenizer.ggml.*`). This module:
//!
//!   - Defines `BertSpecialTokens` + `BertVocab` types holding the
//!     extracted state.
//!   - `BertVocab::from_gguf(&GgufFile)` reads the metadata arrays.
//!   - `build_wordpiece_tokenizer(vocab, specials)` wires up a
//!     `tokenizers::Tokenizer` with BERT normalizer + pre-tokenizer +
//!     WordPiece model.
//!
//! # Tested layers
//!
//! The vocab-extraction functions are unit-tested against synthetic vocab
//! arrays (`build_token_to_id_map`, `build_wordpiece_tokenizer` with a
//! 5-token probe vocab). The `from_gguf` wrapper is a thin metadata-read
//! layer; it's exercised end-to-end when a real BERT GGUF is loaded via
//! `--embedding-model` (live smoke path).

use anyhow::{anyhow, Result};
use mlx_native::gguf::{GgufFile, MetadataValue};
use std::collections::HashMap;

use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::Tokenizer;

// ---------------------------------------------------------------------------
// BertSpecialTokens
// ---------------------------------------------------------------------------

/// Special-token IDs for a BERT tokenizer. Present in every BERT variant;
/// IDs are stored in GGUF metadata as individual u32 keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BertSpecialTokens {
    /// `[CLS]` — classification token, prepended to input. Pool-type CLS
    /// reads its hidden state as the sentence embedding.
    pub cls: u32,
    /// `[SEP]` — separator, terminates input / divides segment pairs.
    pub sep: u32,
    /// `[PAD]` — padding token; masked out during attention.
    pub pad: u32,
    /// `[UNK]` — unknown token; emitted when no WordPiece matches.
    pub unk: u32,
    /// `[MASK]` — masking token for MLM pretraining (used by some
    /// sentence encoders' fine-tuning objective). Optional in
    /// embedding-only use; falls back to the same id as `unk` when the
    /// GGUF doesn't ship it.
    pub mask: u32,
}

impl BertSpecialTokens {
    /// Read special-token IDs from GGUF metadata. Uses llama.cpp's
    /// `tokenizer.ggml.*_token_id` convention. `mask` falls back to `unk`
    /// when absent (not every BERT GGUF includes it).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let read = |key: &str| -> Result<u32> {
            gguf.metadata_u32(key)
                .ok_or_else(|| anyhow!("GGUF missing u32 metadata '{}'", key))
        };
        let cls = read("tokenizer.ggml.cls_token_id")?;
        let sep = read("tokenizer.ggml.seperator_token_id")
            .or_else(|_| read("tokenizer.ggml.separator_token_id"))?;
        let pad = read("tokenizer.ggml.padding_token_id")?;
        let unk = read("tokenizer.ggml.unknown_token_id")?;
        let mask = read("tokenizer.ggml.mask_token_id").unwrap_or(unk);
        Ok(BertSpecialTokens { cls, sep, pad, unk, mask })
    }
}

// ---------------------------------------------------------------------------
// BertVocab
// ---------------------------------------------------------------------------

/// Extracted BERT vocabulary — token strings + special-token IDs.
#[derive(Debug, Clone)]
pub struct BertVocab {
    /// Token strings, indexed by token id. `tokens[id]` is the piece.
    pub tokens: Vec<String>,
    pub specials: BertSpecialTokens,
}

impl BertVocab {
    /// Read vocab + special tokens from GGUF metadata.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let tokens_array = gguf
            .metadata("tokenizer.ggml.tokens")
            .ok_or_else(|| anyhow!("GGUF missing tokenizer.ggml.tokens"))?;
        let arr = match tokens_array {
            MetadataValue::Array(a) => a,
            _ => return Err(anyhow!("tokenizer.ggml.tokens is not an array")),
        };
        let mut tokens: Vec<String> = Vec::with_capacity(arr.len());
        for (i, v) in arr.iter().enumerate() {
            let s = v.as_str().ok_or_else(|| {
                anyhow!("tokenizer.ggml.tokens[{}] is not a string", i)
            })?;
            tokens.push(s.to_string());
        }
        if tokens.is_empty() {
            return Err(anyhow!("tokenizer.ggml.tokens array is empty"));
        }
        let specials = BertSpecialTokens::from_gguf(gguf)?;
        // Bounds-check special token ids.
        let n = tokens.len() as u32;
        for (label, id) in [
            ("cls", specials.cls),
            ("sep", specials.sep),
            ("pad", specials.pad),
            ("unk", specials.unk),
            ("mask", specials.mask),
        ] {
            if id >= n {
                return Err(anyhow!(
                    "special token '{}' id {} out of range (vocab size {})",
                    label,
                    id,
                    n
                ));
            }
        }
        Ok(BertVocab { tokens, specials })
    }

    /// The `[UNK]` token's string (for WordPiece construction).
    pub fn unk_str(&self) -> &str {
        &self.tokens[self.specials.unk as usize]
    }

    /// Size of the vocab.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

// ---------------------------------------------------------------------------
// build_token_to_id_map
// ---------------------------------------------------------------------------

/// Build a `{token_string: id}` map from an ordered vocab. When duplicate
/// strings appear, **the last occurrence wins** — matches HuggingFace
/// WordPiece's `vocab.txt` semantics.
pub fn build_token_to_id_map(tokens: &[String]) -> HashMap<String, u32> {
    let mut m = HashMap::with_capacity(tokens.len());
    for (i, s) in tokens.iter().enumerate() {
        m.insert(s.clone(), i as u32);
    }
    m
}

// ---------------------------------------------------------------------------
// build_wordpiece_tokenizer
// ---------------------------------------------------------------------------

/// Construct a `tokenizers::Tokenizer` from an extracted `BertVocab`.
/// Uses the tokenizers crate's WordPiece model with BERT's standard
/// normalizer + pre-tokenizer.
///
/// This is the **final handoff point**: the returned `Tokenizer` is what
/// `engine.tokenizer()` will return for embedding requests. The calling
/// code can use it the same way the chat path uses its Gemma 4 tokenizer
/// today.
pub fn build_wordpiece_tokenizer(vocab: &BertVocab) -> Result<Tokenizer> {
    // `tokenizers` crate's `WordPieceBuilder::vocab` takes an
    // `ahash::AHashMap` (tokenizers >= 0.22). Build it directly from the
    // ordered token list.
    let mut token_to_id: ahash::AHashMap<String, u32> = ahash::AHashMap::default();
    for (i, s) in vocab.tokens.iter().enumerate() {
        token_to_id.insert(s.clone(), i as u32);
    }
    let wp = WordPiece::builder()
        .vocab(token_to_id)
        .unk_token(vocab.unk_str().to_string())
        .continuing_subword_prefix("##".to_string())
        .max_input_chars_per_word(100)
        .build()
        .map_err(|e| anyhow!("WordPiece builder: {e}"))?;
    // Use the simpler `Tokenizer::new` + setters path. This avoids
    // `TokenizerBuilder`'s generic-inference headache around the
    // unconstrained PostProcessor type parameter when we're not using
    // one yet. BertProcessing (adds [CLS]/[SEP]) lands when the
    // forward-pass path needs it; for now raw WordPiece tokenization is
    // sufficient to prove wiring.
    let mut tokenizer = Tokenizer::new(wp);
    tokenizer.with_normalizer(Some(BertNormalizer::default()));
    tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));
    Ok(tokenizer)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_token_to_id_map_basic() {
        let toks: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let m = build_token_to_id_map(&toks);
        assert_eq!(m.get("a"), Some(&0));
        assert_eq!(m.get("b"), Some(&1));
        assert_eq!(m.get("c"), Some(&2));
        assert_eq!(m.get("d"), None);
    }

    #[test]
    fn build_token_to_id_map_last_duplicate_wins() {
        // HuggingFace WordPiece semantics: when a vocab.txt has a duplicate
        // entry, the later line's index wins. Match that.
        let toks: Vec<String> = vec!["x".into(), "y".into(), "x".into()];
        let m = build_token_to_id_map(&toks);
        assert_eq!(m.get("x"), Some(&2));
        assert_eq!(m.get("y"), Some(&1));
    }

    #[test]
    fn build_wordpiece_tokenizer_with_minimal_vocab() {
        // 5 tokens: [UNK], [CLS], [SEP], [PAD], "hello".
        // Probe: encoding "hello" should produce a single id that maps to
        // "hello" in the vocab.
        let tokens: Vec<String> = vec![
            "[UNK]".into(),
            "[CLS]".into(),
            "[SEP]".into(),
            "[PAD]".into(),
            "hello".into(),
        ];
        let vocab = BertVocab {
            tokens,
            specials: BertSpecialTokens {
                cls: 1,
                sep: 2,
                pad: 3,
                unk: 0,
                mask: 0,
            },
        };
        let tokenizer = build_wordpiece_tokenizer(&vocab).unwrap();
        let enc = tokenizer.encode("hello", false).unwrap();
        let ids = enc.get_ids();
        // "hello" should produce the id for "hello" (4).
        assert!(ids.contains(&4), "expected 'hello' id 4 in {:?}", ids);
    }

    #[test]
    fn build_wordpiece_tokenizer_falls_back_to_unk() {
        // "mystery" is not in the vocab; WordPiece falls back to [UNK].
        let tokens: Vec<String> = vec![
            "[UNK]".into(), // id 0
            "hello".into(),
        ];
        let vocab = BertVocab {
            tokens,
            specials: BertSpecialTokens {
                cls: 0,
                sep: 0,
                pad: 0,
                unk: 0,
                mask: 0,
            },
        };
        let tokenizer = build_wordpiece_tokenizer(&vocab).unwrap();
        let enc = tokenizer.encode("mystery", false).unwrap();
        let ids = enc.get_ids();
        // Should contain the UNK id.
        assert!(ids.contains(&0), "expected [UNK]=0 in {:?}", ids);
    }

    #[test]
    fn bert_vocab_len_matches_input() {
        let tokens: Vec<String> = (0..100).map(|i| format!("tok_{}", i)).collect();
        let vocab = BertVocab {
            tokens,
            specials: BertSpecialTokens {
                cls: 0,
                sep: 1,
                pad: 2,
                unk: 3,
                mask: 4,
            },
        };
        assert_eq!(vocab.len(), 100);
        assert!(!vocab.is_empty());
    }

    #[test]
    fn bert_vocab_empty_is_empty() {
        let vocab = BertVocab {
            tokens: Vec::new(),
            specials: BertSpecialTokens {
                cls: 0,
                sep: 0,
                pad: 0,
                unk: 0,
                mask: 0,
            },
        };
        assert!(vocab.is_empty());
        assert_eq!(vocab.len(), 0);
    }

    #[test]
    fn bert_vocab_unk_str_reads_from_tokens() {
        let vocab = BertVocab {
            tokens: vec!["A".into(), "B".into(), "C".into()],
            specials: BertSpecialTokens {
                cls: 0,
                sep: 1,
                pad: 2,
                unk: 2,
                mask: 2,
            },
        };
        assert_eq!(vocab.unk_str(), "C");
    }

    #[test]
    fn wordpiece_tokenizer_handles_subword_continuation_prefix() {
        // Verify the continuing_subword_prefix "##" is configured so
        // tokens like "##ing" resolve.
        let tokens: Vec<String> = vec![
            "[UNK]".into(),
            "play".into(),
            "##ing".into(),
        ];
        let vocab = BertVocab {
            tokens,
            specials: BertSpecialTokens {
                cls: 0,
                sep: 0,
                pad: 0,
                unk: 0,
                mask: 0,
            },
        };
        let tokenizer = build_wordpiece_tokenizer(&vocab).unwrap();
        let enc = tokenizer.encode("playing", false).unwrap();
        let ids = enc.get_ids();
        // WordPiece should split "playing" → ["play", "##ing"].
        assert!(ids.contains(&1), "expected 'play'=1 in {:?}", ids);
        assert!(ids.contains(&2), "expected '##ing'=2 in {:?}", ids);
    }
}
