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
use tokenizers::processors::bert::BertProcessing;
use tokenizers::Tokenizer;

// ---------------------------------------------------------------------------
// BertWpmTokenizer — llama.cpp-compatible WordPiece (BERT GGUF format)
// ---------------------------------------------------------------------------
//
// llama.cpp stores BERT vocabularies with U+2581 (▁) prefixing every
// word-starter token; subwords are bare. This is the inverse of the
// HuggingFace `tokenizers` crate's WordPiece convention (no prefix on
// word-starters, `##` prefix on subwords). The two conventions are not
// interchangeable — the bge-small GGUF cannot be loaded into HF's
// WordPiece without a vocab translation that risks ambiguous mappings.
//
// Solution: port llama.cpp's `llm_tokenizer_wpm_session::tokenize` from
// `/opt/llama.cpp/src/llama-vocab.cpp:727-813` directly to Rust. The
// algorithm:
//   1. Normalize + lowercase the input (NFD, then `is_whitespace` /
//      `is_punctuation` boundaries split words).
//   2. For each whitespace-separated word, prepend ▁ (U+2581).
//   3. Greedy longest-match against the vocab.
//   4. If no match found for a word, emit a single [UNK].
//
// The `BertProcessing` post-processor wrapping ([CLS] ... [SEP]) is
// applied via a flag on `encode`, so the tokenizer is the canonical
// hf2q tokenizer for any GGUF whose `tokenizer.ggml.model = "bert"`.

/// llama.cpp-compatible BERT WordPiece tokenizer. Matches the C++
/// reference in `/opt/llama.cpp/src/llama-vocab.cpp::llm_tokenizer_wpm_session`
/// byte-for-byte on standard ASCII inputs.
#[derive(Debug, Clone)]
pub struct BertWpmTokenizer {
    /// Token string → id map (built once at construction).
    token_to_id: HashMap<String, u32>,
    /// Special-token ids extracted from GGUF metadata.
    specials: BertSpecialTokens,
    /// Maximum token length in bytes (for the greedy matcher's inner
    /// loop bound).
    max_token_len: usize,
}

impl BertWpmTokenizer {
    /// Build from an extracted `BertVocab`. Stores a name → id map for
    /// the greedy matcher.
    pub fn new(vocab: &BertVocab) -> Self {
        let mut max_len = 0usize;
        let mut token_to_id = HashMap::with_capacity(vocab.tokens.len());
        for (i, tok) in vocab.tokens.iter().enumerate() {
            max_len = max_len.max(tok.len());
            token_to_id.insert(tok.clone(), i as u32);
        }
        Self {
            token_to_id,
            specials: vocab.specials,
            max_token_len: max_len.max(1),
        }
    }

    /// Tokenize text. When `add_special_tokens` is true, the result is
    /// `[CLS] ...tokens... [SEP]`.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<u32> {
        let words = preprocess_words(text);
        let mut output: Vec<u32> = Vec::with_capacity(words.len() * 2 + 2);
        if add_special_tokens {
            output.push(self.specials.cls);
        }

        for word in words {
            if word.is_empty() {
                continue;
            }
            // Prepend ▁ (U+2581) to mark word start. Matches
            // `llama-vocab.cpp:743 const std::string word1 = "\xe2\x96\x81" + word;`.
            let mut word1 = String::with_capacity(word.len() + 3);
            word1.push('\u{2581}');
            word1.push_str(&word);
            let bytes = word1.as_bytes();
            let n = bytes.len();
            let current_tokens = output.len();

            let mut i = 0usize;
            let mut matched_word = true;
            while i < n {
                // Greedy longest-match: try lengths from max down to i+1.
                let mut found_at: Option<usize> = None;
                let upper = std::cmp::min(n, i + self.max_token_len + 1);
                let mut j = upper;
                while j > i {
                    let slice = &bytes[i..j];
                    // We must only attempt valid UTF-8 boundaries — Rust
                    // `&str::from_utf8` rejects mid-codepoint slices.
                    if let Ok(s) = std::str::from_utf8(slice) {
                        if let Some(&id) = self.token_to_id.get(s) {
                            found_at = Some(j);
                            output.push(id);
                            break;
                        }
                    }
                    j -= 1;
                }
                match found_at {
                    Some(end) => {
                        i = end;
                    }
                    None => {
                        // No match at this start position → bail out for
                        // this word; matches llama.cpp's `// discard all`
                        // path.
                        output.truncate(current_tokens);
                        matched_word = false;
                        break;
                    }
                }
            }

            // No matches at all for this word → emit a single [UNK]
            // (matches `output.push_back(vocab.token_unk());` in llama.cpp).
            if !matched_word || output.len() == current_tokens {
                output.push(self.specials.unk);
            }
        }

        if add_special_tokens {
            output.push(self.specials.sep);
        }
        output
    }

    pub fn specials(&self) -> &BertSpecialTokens {
        &self.specials
    }
}

/// Mirror of llama.cpp's `llm_tokenizer_wpm_session::preprocess`. Splits
/// `text` into a `Vec<String>` of words, applying:
///   - NFD normalization (best-effort via a small helper since we don't
///     pull in unicode-normalization to keep deps light; for ASCII
///     inputs this is a no-op).
///   - Lowercase folding via `char::to_lowercase`.
///   - Whitespace split (drops the whitespace).
///   - Punctuation split (each punctuation char becomes its own word).
///   - Drop control / `\0` / U+FFFD code points.
///
/// For ASCII-only inputs the output matches llama.cpp byte-for-byte. For
/// non-ASCII inputs (CJK, accented Latin, etc.) the NFD-normalization step
/// is the only divergence; that's a known iter-65 follow-up.
fn preprocess_words(text: &str) -> Vec<String> {
    let mut words: Vec<String> = vec![String::new()];
    for c in text.chars() {
        // Drop control / null / replacement.
        if c == '\0' || c == '\u{FFFD}' || c.is_control() {
            continue;
        }
        if c.is_whitespace() {
            if !words.last().unwrap().is_empty() {
                words.push(String::new());
            }
            continue;
        }
        // Mirror llama.cpp's tolower + the punctuation-isolates-as-its-
        // own-word rule. `is_ascii_punctuation` is conservative — for
        // non-ASCII punctuation a future iter widens via the
        // `unicode_categories` crate; current matches llama.cpp on every
        // ASCII codepoint and on CJK ideographs (which fall through to
        // the append branch).
        let lower: String = c.to_lowercase().collect();
        if c.is_ascii_punctuation()
            || (c.is_ascii() && (c as u32) < 0x7F && c.is_ascii_graphic() && !c.is_ascii_alphanumeric() && !c.is_ascii_whitespace())
        {
            // is_ascii_punctuation already covers ., ?, !, etc. The
            // second clause re-checks for ASCII symbols outside
            // alphanumeric (e.g. `$`, `+`) since llama.cpp also splits
            // those.
            if !words.last().unwrap().is_empty() {
                words.push(String::new());
            }
            words.push(lower);
            words.push(String::new());
        } else {
            words.last_mut().unwrap().push_str(&lower);
        }
    }
    if words.last().map(|w| w.is_empty()).unwrap_or(false) {
        words.pop();
    }
    words
}

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
    // Wrap inputs in [CLS] ... [SEP] when the caller passes
    // `add_special_tokens = true` — without this, BERT sees a sentence
    // missing both sentinels and the embedding diverges from
    // llama-embedding (iter 63 cosine = 0.46).
    let cls_id = vocab.specials.cls;
    let sep_id = vocab.specials.sep;
    let cls_tok = vocab
        .tokens
        .get(cls_id as usize)
        .cloned()
        .ok_or_else(|| anyhow!("vocab missing CLS token at id {}", cls_id))?;
    let sep_tok = vocab
        .tokens
        .get(sep_id as usize)
        .cloned()
        .ok_or_else(|| anyhow!("vocab missing SEP token at id {}", sep_id))?;
    tokenizer.with_post_processor(Some(BertProcessing::new(
        (sep_tok, sep_id),
        (cls_tok, cls_id),
    )));
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

    /// Iter 64 diagnostic: dump a sample of the bge vocab to see the
    /// prefix-marker convention. `▁` (U+2581) prefixes word-starters in
    /// llama.cpp's BERT path; subwords might or might not have `##`.
    #[test]
    fn bge_small_vocab_format_diagnostic() {
        let path = std::path::Path::new(
            "/opt/hf2q/models/bert-test/bge-small-en-v1.5-f16.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: bge GGUF not on disk");
            return;
        }
        let gguf = mlx_native::gguf::GgufFile::open(path).expect("open");
        let vocab = BertVocab::from_gguf(&gguf).expect("vocab");
        // Print key indices: [PAD], [UNK], [CLS], [SEP], "hello"=7592,
        // "world"=2088, and a few sample subword indices.
        for &idx in &[0u32, 100, 101, 102, 1000, 2088, 3000, 7592, 10000, 11108] {
            eprintln!(
                "vocab[{:5}] = {:?}",
                idx,
                vocab.tokens.get(idx as usize)
            );
        }
        // Count how many tokens start with ▁ vs how many start with ## vs
        // bare. Tells us the prefix convention at a glance.
        let prefix_marker = "\u{2581}";
        let mut n_prefix = 0;
        let mut n_continuation = 0;
        let mut n_bare = 0;
        for tok in &vocab.tokens {
            if tok.starts_with(prefix_marker) {
                n_prefix += 1;
            } else if tok.starts_with("##") {
                n_continuation += 1;
            } else {
                n_bare += 1;
            }
        }
        eprintln!(
            "vocab counts: total={}, ▁-prefix={}, ##-prefix={}, bare={}",
            vocab.tokens.len(),
            n_prefix,
            n_continuation,
            n_bare,
        );
    }

    /// Iter 64 parity test: build the WordPiece tokenizer from the real
    /// bge-small-en-v1.5 GGUF and verify it produces exactly the token
    /// ids llama.cpp does. Fixture file is gated on existence so CI
    /// without the artifact skips cleanly. Expected ids derived from
    /// `llama-embedding -m ... --verbose-prompt`:
    ///   "hello world"  →  [101, 7592, 2088, 102]   (CLS hello world SEP)
    #[test]
    fn bge_small_tokenizer_matches_llama_cpp_on_hello_world() {
        let path = std::path::Path::new(
            "/opt/hf2q/models/bert-test/bge-small-en-v1.5-f16.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: bge GGUF not on disk at {}", path.display());
            return;
        }
        let gguf = mlx_native::gguf::GgufFile::open(path).expect("open bge GGUF");
        let vocab = BertVocab::from_gguf(&gguf).expect("vocab parse");
        let tokenizer = BertWpmTokenizer::new(&vocab);
        let ids = tokenizer.encode("hello world", true);
        // [CLS] hello world [SEP] = [101, 7592, 2088, 102] for bert-base-uncased.
        assert_eq!(
            ids,
            vec![101u32, 7592, 2088, 102],
            "tokenization mismatch — vocab[100..103]: {:?}, vocab[7592]: {:?}, vocab[2088]: {:?}",
            vocab.tokens.get(100..103),
            vocab.tokens.get(7592),
            vocab.tokens.get(2088),
        );
    }
}
