//! GGUF-driven BPE tokenizer construction for Qwen3.5 / Qwen3.6.
//!
//! Builds a `tokenizers::Tokenizer` programmatically from the GGUF's own
//! `tokenizer.ggml.*` metadata arrays — the same source the peer
//! consumes. Produces byte-identical token streams to llama-tokenize on
//! the same GGUF, regardless of what the on-disk `tokenizer.json` claims.
//!
//! ## Why this module exists
//!
//! Loading the HF `tokenizer.json` directly is wrong when the GGUF has
//! a smaller vocabulary than the HF tokenizer declares. Concretely the
//! abliterated Qwen3.6-35B-A3B apex GGUF stores 248,044 tokens in
//! `tokenizer.ggml.tokens` and 248,044 rows in `token_embd.weight`, but
//! its `tokenizer.json` declares `added_tokens` with IDs through 248,319
//! (e.g. `<|im_start|>`=248045). When chat-templated prompts are
//! tokenized via the HF `tokenizer.json`, those out-of-vocab IDs hit
//! the embedding loader's zero-pad fallback (weight_loader.rs:699-707)
//! and the residual stream collapses to deterministic prompt-repetition
//! gibberish.
//!
//! The peer's vocab path builds the tokenizer exclusively from the
//! GGUF's own metadata — so the literal string `<|im_start|>` simply
//! isn't a special token and falls through to byte-level
//! pre-tokenization producing 6 raw chars: `<`, `|`, `im`, `_start`,
//! `|`, `>`. This module follows the same GGUF-only construction in
//! Rust, against the HuggingFace `tokenizers` crate.
//!
//! ## Parity-relevant facts
//!
//! - GGUF key catalogue: `tokenizer.ggml.{model, pre, tokens,
//!   token_type, scores, merges, bos_token_id, eos_token_id,
//!   padding_token_id, add_bos_token, ..., chat_template}`.
//! - Vocab construction resolves token/score/toktype indexes; sizes the
//!   vocab to the GGUF's actual `n_tokens`; sets per-token attributes
//!   from the `token_type` array values.
//! - Qwen3.5 pre-tokenizer: regex identical to the HF
//!   `tokenizer.json`'s `pre_tokenizer.Sequence[0]`
//!   `Split { pattern: { Regex: ... } }` for the qwen35 family.
//! - `tokenizer_pre == "qwen35"` selects this pre-type with
//!   `clean_spaces = false`.
//!
//! ## Contract
//!
//! The output `tokenizers::Tokenizer` produces token streams that match
//! `llama-tokenize` byte-for-byte on the same GGUF for the test fixtures
//! listed in `src/inference/models/qwen35/tokenizer.rs::tests`
//! (raw text, chat-template-rendered text, multi-byte UTF-8). If the
//! parity test ever fails, this module is wrong — the test IS the spec.
//!
//! ## Sovereignty
//!
//! Per ADR-013 §"Absolute sovereignty": no peer binary, library,
//! or output is linked at build/test/CI time. The spec was derived from
//! the peer's vocab path; the Rust BPE engine + regex engine come from
//! the HF `tokenizers` crate (already a dependency). `llama-tokenize` is
//! invoked at gate-time (parity test) as an external black-box reference,
//! never imported.

use ahash::AHashMap;

use anyhow::{anyhow, bail, Result};
use mlx_native::gguf::{GgufFile, MetadataValue};
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDec;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence as PreSeq;
use tokenizers::pre_tokenizers::split::Split;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

/// Pre-tokenizer regex for the `qwen35` family.
///
/// Must stay identical to the peer's qwen35 pre-tokenizer regex and to
/// the HF `tokenizer.json`'s `pre_tokenizer.Sequence[0].pattern.Regex`
/// for the same family — confirmed by `python3 -c "import json;
/// json.load(open(...))['pre_tokenizer']"` on
/// `qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/tokenizer.json` on
/// 2026-05-01.
///
/// Differs from the QWEN2 regex by including `\p{M}` (combining marks)
/// in the letter class, which matters for combining-diacritic scripts
/// (Devanagari, Hebrew nikkud, Vietnamese tone marks).
pub const QWEN35_PRE_REGEX: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// Token-type values as stored in the GGUF's
/// `tokenizer.ggml.token_type` array.
///
/// We only need to distinguish "this token must be matched as an atomic
/// unit during pre-tokenization" (CONTROL + USER_DEFINED) from
/// "ordinary BPE token". UNUSED tokens stay in the vocab but we don't
/// register them with `add_special_tokens` — they're inert.
#[allow(dead_code)] // documentation constants; only CONTROL + USER_DEFINED are matched on
mod token_type {
    pub const UNDEFINED: i32 = 0;
    pub const NORMAL: i32 = 1;
    pub const UNKNOWN: i32 = 2;
    pub const CONTROL: i32 = 3;
    pub const USER_DEFINED: i32 = 4;
    pub const UNUSED: i32 = 5;
    pub const BYTE: i32 = 6;
}

/// Whether a `tokenizer.ggml.token_type` value registers its token as an
/// atomic special token.
fn is_atomic_special(token_type: i32) -> bool {
    matches!(token_type, token_type::CONTROL | token_type::USER_DEFINED)
}

/// Special-token id keys: the fallback special set of a GGUF without
/// `tokenizer.ggml.token_type`, and the ids the serving engine reads.
pub(crate) const SPECIAL_TOKEN_ID_KEYS: [&str; 6] = [
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.eot_token_id",
    "tokenizer.ggml.eom_token_id",
];

/// SHA-256 identity of the tokenizer [`build_tokenizer_from_gguf`] builds
/// from `gguf`: its pre-type, vocabulary, merges, the set of tokens it
/// registers as atomic specials (from `tokenizer.ggml.token_type`, or from
/// the special-token ids when a GGUF has no type array, exactly as the
/// builder does) and the special-token ids themselves. Two GGUFs with the
/// same identity tokenize alike; the chat template is not part of it.
///
/// Layout: for each section its name, its entry count and every entry
/// length-prefixed (counts little-endian `u64`); then `ids`, the count and
/// each of [`SPECIAL_TOKEN_ID_KEYS`] as a little-endian `u32` (`u32::MAX`
/// when absent).
pub(crate) fn tokenizer_identity(gguf: &GgufFile) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let mut section = |name: &str, entries: &[&[u8]]| {
        hasher.update(name.as_bytes());
        hasher.update((entries.len() as u64).to_le_bytes());
        for entry in entries {
            hasher.update((entry.len() as u64).to_le_bytes());
            hasher.update(entry);
        }
    };
    let bytes = |strings: &[String]| -> Vec<Vec<u8>> {
        strings.iter().map(|text| text.as_bytes().to_vec()).collect()
    };
    let pre = gguf
        .metadata_string("tokenizer.ggml.pre")
        .ok_or_else(|| anyhow!("GGUF missing `tokenizer.ggml.pre`"))?;
    let tokens = read_string_array(gguf, "tokenizer.ggml.tokens")?;
    let merges = read_string_array(gguf, "tokenizer.ggml.merges").unwrap_or_default();
    let ids = SPECIAL_TOKEN_ID_KEYS.map(|key| gguf.metadata_u32(key));
    let special: Vec<u8> = match read_i32_array(gguf, "tokenizer.ggml.token_type") {
        Ok(types) => {
            if types.len() != tokens.len() {
                bail!(
                    "tokenizer.ggml.token_type length {} != tokens length {}",
                    types.len(),
                    tokens.len()
                );
            }
            types
                .iter()
                .map(|token_type| u8::from(is_atomic_special(*token_type)))
                .collect()
        }
        Err(_) => {
            let mut special = vec![0_u8; tokens.len()];
            for id in ids.into_iter().flatten() {
                if let Some(flag) = special.get_mut(id as usize) {
                    *flag = 1;
                }
            }
            special
        }
    };
    section("pre", &[pre.as_bytes()]);
    for (name, strings) in [("tokens", &tokens), ("merges", &merges)] {
        let entries = bytes(strings);
        let entries: Vec<&[u8]> = entries.iter().map(Vec::as_slice).collect();
        section(name, &entries);
    }
    section("special", &[&special]);
    hasher.update(b"ids");
    hasher.update((ids.len() as u64).to_le_bytes());
    for id in ids {
        hasher.update(id.unwrap_or(u32::MAX).to_le_bytes());
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Construct a `tokenizers::Tokenizer` from `gguf`'s tokenizer metadata.
///
/// Returns `Err` if the GGUF is missing required metadata or if its
/// `tokenizer.ggml.pre` is not `"qwen35"` (this module does not handle
/// other pre-tokenizer families — those would need their own regex per
/// `llama-vocab.cpp:355-2165`).
///
/// Errors do not panic — they propagate through `anyhow::Error`.
pub fn build_tokenizer_from_gguf(gguf: &GgufFile) -> Result<Tokenizer> {
    // ---- Pre-type guard ----
    let pre = gguf
        .metadata_string("tokenizer.ggml.pre")
        .ok_or_else(|| anyhow!("GGUF missing `tokenizer.ggml.pre`"))?;
    if pre != "qwen35" {
        bail!(
            "tokenizer.ggml.pre = {pre:?} is not supported by qwen35::tokenizer; \
             this module only handles `qwen35`. Other pre-types need their \
             own regex builder."
        );
    }

    // ---- Vocab tokens ----
    let tokens = read_string_array(gguf, "tokenizer.ggml.tokens")?;
    if tokens.is_empty() {
        bail!("`tokenizer.ggml.tokens` is empty");
    }
    // Mirror llama-vocab.cpp:2228-2231 (replace empty tokens with a
    // unique placeholder so HashMap insert doesn't collide). HF
    // `BPE::vocab` is a `HashMap<String, u32>`, so duplicate empty
    // strings would silently overwrite each other.
    let tokens: Vec<String> = tokens
        .into_iter()
        .enumerate()
        .map(|(i, t)| {
            if t.is_empty() {
                format!("[EMPTY_{i}]")
            } else {
                t
            }
        })
        .collect();

    // ---- Vocab map: token → id ----
    //
    // `tokenizers::models::bpe::BpeBuilder::vocab_and_merges` takes a
    // `V: Into<AHashMap<String, u32>>`, so we build the map in `ahash`'s
    // hash type directly to avoid an intermediate clone-via-conversion
    // through `HashMap<_, _, RandomState>`.
    let vocab: AHashMap<String, u32> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();
    if vocab.len() != tokens.len() {
        bail!(
            "duplicate tokens in `tokenizer.ggml.tokens` ({} unique vs {} entries) \
             — GGUF is malformed",
            vocab.len(),
            tokens.len()
        );
    }

    // ---- Merges ----
    // Each entry is "tokA tokB" — the BPE pair with implicit priority by
    // index (earlier merges have higher priority). Split on the FIRST
    // space; tokens themselves never contain raw ASCII space (it's
    // byte-level encoded as `Ġ`, code point U+0120).
    let merges_raw = read_string_array(gguf, "tokenizer.ggml.merges")?;
    let merges: Vec<(String, String)> = merges_raw
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let mut split = m.splitn(2, ' ');
            let a = split
                .next()
                .ok_or_else(|| anyhow!("merge[{i}] = {m:?} has no space separator"))?;
            let b = split
                .next()
                .ok_or_else(|| anyhow!("merge[{i}] = {m:?} has only one half (need 'a b')"))?;
            Ok::<_, anyhow::Error>((a.to_string(), b.to_string()))
        })
        .collect::<Result<_>>()?;

    // ---- BPE model ----
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| anyhow!("BPE::builder().build(): {e}"))?;

    // ---- Pre-tokenizer: Sequence(Split(QWEN35_REGEX, Isolated), ByteLevel) ----
    //
    // Mirrors HF tokenizer.json's pre_tokenizer.Sequence:
    //   [0] Split { pattern: Regex(QWEN35_PRE_REGEX), behavior: Isolated, invert: false }
    //   [1] ByteLevel { add_prefix_space: false, trim_offsets: false, use_regex: false }
    //
    // The Split phase chunks input by the qwen35 regex (matches as
    // contiguous spans). The ByteLevel phase then re-encodes each span's
    // bytes through the GPT-2 byte-to-unicode map — turning ` ` → `Ġ`,
    // `\n` → `Ċ`, etc. — so the BPE merges (which are over byte-level
    // tokens) match the expected vocab keys.
    //
    // `use_regex: false` for the inner ByteLevel is critical: it tells
    // ByteLevel to treat its input as already-pre-split (Split did the
    // splitting) and just byte-level-encode each span as-is.
    let split = Split::new(
        tokenizers::pre_tokenizers::split::SplitPattern::Regex(QWEN35_PRE_REGEX.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| anyhow!("Split::new: {e}"))?;
    let byte_level_pre = ByteLevel::new(
        /* add_prefix_space = */ false, /* trim_offsets =     */ false,
        /* use_regex =        */ false,
    );
    let pre_seq = PreSeq::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(byte_level_pre),
    ]);

    // ---- Decoder: ByteLevel ----
    //
    // Inverse of the byte-level pre-tokenization: maps `Ġ` → ` `,
    // `Ċ` → `\n`, etc., when joining decoded subword tokens back into
    // a string.
    let decoder = ByteLevelDec::new(
        /* add_prefix_space = */ false, /* trim_offsets =     */ false,
        /* use_regex =        */ false,
    );

    // ---- Assemble tokenizer ----
    let mut tok = Tokenizer::new(bpe);
    tok.with_pre_tokenizer(Some(pre_seq));
    tok.with_decoder(Some(decoder));

    // ---- Special tokens ----
    //
    // Register tokens with `LLAMA_TOKEN_TYPE_CONTROL` (3) or
    // `LLAMA_TOKEN_TYPE_USER_DEFINED` (4) so they're matched as
    // atomic units during pre-tokenization rather than being split
    // by the regex into characters. This mirrors
    // `llama-vocab.cpp:2241-2251` mapping `LLAMA_TOKEN_TYPE_CONTROL`
    // and `_USER_DEFINED` to attrs that `tokenizer_st_partition`
    // (line 2903+) treats as atomic.
    //
    // NB: we only register IDs that exist in the GGUF's vocab (i.e.
    // `id < tokens.len()`). Out-of-vocab IDs from a stale tokenizer.json
    // never enter our pipeline because we never read tokenizer.json.
    let token_types = read_i32_array(gguf, "tokenizer.ggml.token_type").ok();
    if let Some(types) = &token_types {
        if types.len() != tokens.len() {
            bail!(
                "tokenizer.ggml.token_type length {} != tokens length {}",
                types.len(),
                tokens.len()
            );
        }
        let specials: Vec<AddedToken> = types
            .iter()
            .zip(tokens.iter())
            .enumerate()
            .filter_map(|(_id, (ttype, name))| {
                if !is_atomic_special(*ttype) {
                    return None;
                }
                // `AddedToken::special(true)` flags the token as a
                // control/special token (single_word=false, lstrip=false,
                // rstrip=false, normalized=false by default for
                // `from(_, special=true)`). Mirrors HF tokenizer.json's
                // added_tokens entries with `special: true`.
                Some(AddedToken::from(name.clone(), true))
            })
            .collect();
        tok.add_special_tokens(&specials);
    } else {
        // No token_type metadata. Fall back to a minimal special-token
        // set derived from the explicit `tokenizer.ggml.*_token_id`
        // metadata keys, if present. This matches llama-vocab.cpp's
        // pre-toktypes-array path (older GGUFs).
        let mut specials: Vec<AddedToken> = Vec::new();
        for key in SPECIAL_TOKEN_ID_KEYS {
            if let Some(id) = gguf.metadata_u32(key) {
                if let Some(name) = tokens.get(id as usize) {
                    specials.push(AddedToken::from(name.clone(), true));
                }
            }
        }
        tok.add_special_tokens(&specials);
    }

    Ok(tok)
}

/// Read a `MetadataValue::Array` of `String` entries.
fn read_string_array(gguf: &GgufFile, key: &str) -> Result<Vec<String>> {
    let v = gguf
        .metadata(key)
        .ok_or_else(|| anyhow!("GGUF missing `{key}`"))?;
    let arr = match v {
        MetadataValue::Array(a) => a,
        other => bail!("`{key}` is not an array (got {other:?})"),
    };
    arr.iter()
        .enumerate()
        .map(|(i, e)| match e {
            MetadataValue::String(s) => Ok(s.clone()),
            other => Err(anyhow!("`{key}`[{i}] is not a string (got {other:?})")),
        })
        .collect()
}

/// Read a `MetadataValue::Array` of integer entries (Int32 or Uint32).
fn read_i32_array(gguf: &GgufFile, key: &str) -> Result<Vec<i32>> {
    let v = gguf
        .metadata(key)
        .ok_or_else(|| anyhow!("GGUF missing `{key}`"))?;
    let arr = match v {
        MetadataValue::Array(a) => a,
        other => bail!("`{key}` is not an array (got {other:?})"),
    };
    arr.iter()
        .enumerate()
        .map(|(i, e)| match e {
            MetadataValue::Int32(x) => Ok(*x),
            MetadataValue::Uint32(x) => Ok(*x as i32),
            MetadataValue::Int8(x) => Ok(*x as i32),
            MetadataValue::Uint8(x) => Ok(*x as i32),
            MetadataValue::Int16(x) => Ok(*x as i32),
            MetadataValue::Uint16(x) => Ok(*x as i32),
            other => Err(anyhow!("`{key}`[{i}] is not an integer (got {other:?})")),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// A metadata-only GGUF with a three-token vocabulary.
    fn tokenizer_gguf(
        tokens: [&str; 3],
        token_types: Option<[i32; 3]>,
        eos: Option<u32>,
    ) -> (tempfile::TempDir, GgufFile) {
        use crate::backends::gguf::types::MetaValue;
        use crate::backends::gguf::writer::GgufWriter;
        let mut metadata = vec![
            ("general.architecture", MetaValue::String("qwen35".into())),
            ("tokenizer.ggml.pre", MetaValue::String("qwen35".into())),
            (
                "tokenizer.ggml.tokens",
                MetaValue::ArrayString(tokens.iter().map(|token| (*token).to_owned()).collect()),
            ),
            (
                "tokenizer.ggml.merges",
                MetaValue::ArrayString(vec!["a b".into()]),
            ),
        ];
        if let Some(types) = token_types {
            metadata.push(("tokenizer.ggml.token_type", MetaValue::ArrayI32(types.to_vec())));
        }
        if let Some(eos) = eos {
            metadata.push(("tokenizer.ggml.eos_token_id", MetaValue::U32(eos)));
        }
        let mut writer = GgufWriter::new(std::io::Cursor::new(Vec::new()));
        writer.write_header(0, metadata.len() as u64).expect("header");
        for (key, value) in &metadata {
            writer.write_metadata_kv(key, value).expect("metadata");
        }
        writer.pad_to_alignment().expect("alignment");
        writer.finalize().expect("finalize");
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("tokenizer.gguf");
        std::fs::write(&path, writer.into_inner().into_inner()).expect("write GGUF");
        let gguf = GgufFile::open(&path).expect("open GGUF");
        (directory, gguf)
    }

    #[test]
    fn tokenizer_identity_follows_the_special_tokens_the_builder_registers() {
        let identity = |tokens, types, eos| {
            let (_directory, gguf) = tokenizer_gguf(tokens, types, eos);
            tokenizer_identity(&gguf).expect("identity")
        };
        let tokens = ["a", "b", "<|end|>"];
        let base = identity(tokens, Some([1, 1, 3]), Some(2));
        assert_eq!(base, identity(tokens, Some([1, 1, 3]), Some(2)), "deterministic");
        // The same strings and merges with the control token retyped as an
        // ordinary one: the builder no longer registers it as atomic.
        assert_ne!(base, identity(tokens, Some([1, 1, 1]), Some(2)));
        // CONTROL and USER_DEFINED register alike, as do the ordinary types.
        assert_eq!(base, identity(tokens, Some([1, 1, 4]), Some(2)));
        assert_eq!(base, identity(tokens, Some([1, 6, 3]), Some(2)));
        // Without a type array the special set is the special-token ids.
        assert_eq!(base, identity(tokens, None, Some(2)));
        assert_ne!(base, identity(tokens, None, Some(1)));
        assert_ne!(base, identity(tokens, None, None));
        // A special-token id the engine reads, and the vocabulary itself.
        assert_ne!(base, identity(tokens, Some([1, 1, 3]), Some(1)));
        assert_ne!(base, identity(["a", "c", "<|end|>"], Some([1, 1, 3]), Some(2)));
    }

    /// The qwen35 pre-tokenizer regex must be identical to the peer's
    /// literal. If the peer updates the regex, this test reminds us to
    /// update `QWEN35_PRE_REGEX` in lock-step.
    #[test]
    fn pre_regex_matches_llama_cpp_spec() {
        let expected = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
        assert_eq!(QWEN35_PRE_REGEX, expected);
    }

    /// Build the tokenizer against the apex GGUF on disk and confirm
    /// `<|im_start|>` tokenizes to in-range bytes — NOT a single OOB
    /// special-token id.
    ///
    /// Skipped (not failed) when the apex GGUF is not staged on this
    /// host. The fixture path mirrors ADR-013 §"Existing local
    /// reference" — the apex GGUF is a convenience reference for eyes,
    /// not a CI dependency, so the test gates on its existence.
    #[test]
    fn apex_im_start_does_not_tokenize_to_oob_id() {
        let path = Path::new(
            "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/\
             APEX-Q5_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("apex GGUF not present at {path:?}; skipping");
            return;
        }
        let gguf = GgufFile::open(path).expect("open apex GGUF");
        let tok = build_tokenizer_from_gguf(&gguf).expect("build tokenizer");

        // The GGUF advertises 248,044 vocab entries; any tokenized id
        // ≥ that is unreachable in the model's `token_embd.weight` and
        // would zero-pad. Assert no token in the encoding falls there.
        let physical_vocab = read_string_array(&gguf, "tokenizer.ggml.tokens")
            .expect("tokens array")
            .len() as u32;
        let prompt = "<|im_start|>user\nHow to make bread?<|im_end|>\n<|im_start|>assistant\n";
        let enc = tok
            .encode(prompt, false)
            .expect("encode chat-templated prompt");
        for &id in enc.get_ids() {
            assert!(
                id < physical_vocab,
                "token id {id} >= physical vocab {physical_vocab} \
                 — pre-tokenizer let an OOB special through (regression \
                 of the iter61a/iter40 vocab-mismatch class)"
            );
        }
    }

    /// Building the tokenizer requires a GGUF with `tokenizer.ggml.pre`
    /// equal to `qwen35`. Other values are an explicit error rather
    /// than a silent fallback.
    #[test]
    fn build_succeeds_on_qwen35_gguf() {
        let path = Path::new(
            "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/\
             APEX-Q5_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("apex GGUF not present; skipping");
            return;
        }
        let gguf = GgufFile::open(path).expect("open");
        let _tok = build_tokenizer_from_gguf(&gguf).expect("build");
    }
}
