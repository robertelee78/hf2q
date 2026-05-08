//! GGUF-driven Gemma4 SPM-BPE tokenizer construction (ADR-022 P1.11).
//!
//! Builds a `tokenizers::Tokenizer` programmatically from the GGUF's own
//! `tokenizer.ggml.*` metadata arrays — no on-disk `tokenizer.json`
//! required. Sibling to `qwen35::tokenizer::build_tokenizer_from_gguf`
//! but for the Gemma4 SentencePiece-derived BPE family.
//!
//! ## Why Gemma4 differs from Qwen3.5
//!
//! Gemma4 uses a SentencePiece-derived BPE with raw UTF-8 (NOT GPT-2
//! byte encoding):
//!   - Normalizer: replace ASCII space with `▁` (U+2581) — sentencepiece
//!     convention for word-prefix marking.
//!   - Pre-tokenizer: `Split(" ", MergedWithPrevious)` — keeps each word
//!     attached to its leading space.
//!   - BPE model: `byte_fallback=true`, `fuse_unk=true`, `unk="<unk>"`,
//!     `ignore_merges=false`. Vocab + merges read directly from
//!     `tokenizer.ggml.tokens` and `tokenizer.ggml.merges`.
//!   - Decoder: `Sequence([Replace("▁" → " "), ByteFallback, Fuse])` —
//!     inverse of the normalizer + handles byte-fallback'd raw bytes.
//!
//! Qwen3.5 in contrast uses a complex regex pre-tokenizer + GPT-2
//! byte-level encoding. The two pipelines share zero of their normalizer
//! + pre-tokenizer + decoder configuration.
//!
//! ## Spec citations
//!
//! - llama.cpp `LLAMA_VOCAB_PRE_TYPE_GEMMA4` regex:
//!   `/opt/llama.cpp/src/llama-vocab.cpp:496-505` —
//!   `regex_exprs = {"[^\\n]+|[\\n]+"}; byte_encode = false;`.
//! - HF `tokenizer.json` schema for the abliterated Gemma4 file lives
//!   at `/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/tokenizer.json`;
//!   the structure this builder reproduces was inspected key-by-key
//!   (see ADR-022 §P1.11 progress notes).
//!
//! ## Contract
//!
//! Token streams produced by this builder match the on-disk
//! `tokenizer.json` byte-for-byte for the same input on the same GGUF.
//! The parity test in `tests/adr_022_phase1_p11_gemma4_tokenizer_parity.rs`
//! is the falsifier.

use ahash::AHashMap;
use anyhow::{anyhow, bail, Context, Result};
use mlx_native::gguf::{GgufFile, MetadataValue};
use tokenizers::decoders::byte_fallback::ByteFallback;
use tokenizers::decoders::fuse::Fuse;
use tokenizers::decoders::sequence::Sequence as DecSeq;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::replace::Replace as NormReplace;
use tokenizers::pre_tokenizers::split::Split;
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

const SPM_SPACE: &str = "\u{2581}";

mod token_type {
    #![allow(dead_code)]
    pub const NORMAL: i32 = 1;
    pub const UNKNOWN: i32 = 2;
    pub const CONTROL: i32 = 3;
    pub const USER_DEFINED: i32 = 4;
    pub const UNUSED: i32 = 5;
    pub const BYTE: i32 = 6;
}

/// Construct a Gemma4 `tokenizers::Tokenizer` from GGUF metadata.
pub fn build_tokenizer_from_gguf(gguf: &GgufFile) -> Result<Tokenizer> {
    let model = gguf
        .metadata_string("tokenizer.ggml.model")
        .ok_or_else(|| anyhow!("GGUF missing `tokenizer.ggml.model`"))?;
    if model != "gemma4" {
        bail!(
            "tokenizer.ggml.model = {model:?} is not supported by gemma4::tokenizer; \
             this module only handles `gemma4` (see llama-vocab.cpp:496)."
        );
    }

    let tokens = read_string_array(gguf, "tokenizer.ggml.tokens")?;
    if tokens.is_empty() {
        bail!("`tokenizer.ggml.tokens` is empty");
    }
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

    let vocab: AHashMap<String, u32> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();
    if vocab.len() != tokens.len() {
        bail!(
            "duplicate tokens in `tokenizer.ggml.tokens` ({} unique vs {} entries)",
            vocab.len(),
            tokens.len()
        );
    }

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
                .ok_or_else(|| anyhow!("merge[{i}] = {m:?} has only one half"))?;
            Ok::<_, anyhow::Error>((a.to_string(), b.to_string()))
        })
        .collect::<Result<_>>()?;

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .byte_fallback(true)
        .ignore_merges(false)
        .fuse_unk(true)
        .unk_token("<unk>".to_string())
        .build()
        .map_err(|e| anyhow!("BPE::builder().build(): {e}"))?;

    let normalizer = NormReplace::new(" ", SPM_SPACE)
        .map_err(|e| anyhow!("NormReplace::new for space→{SPM_SPACE}: {e}"))?;

    let pre = Split::new(
        tokenizers::pre_tokenizers::split::SplitPattern::String(" ".to_string()),
        SplitDelimiterBehavior::MergedWithPrevious,
        false,
    )
    .map_err(|e| anyhow!("Split::new(\" \", MergedWithPrevious): {e}"))?;

    let decoder_replace = NormReplace::new(SPM_SPACE, " ")
        .map_err(|e| anyhow!("Decoder Replace {SPM_SPACE}→space: {e}"))?;
    let decoder = DecSeq::new(vec![
        DecoderWrapper::Replace(decoder_replace),
        DecoderWrapper::ByteFallback(ByteFallback::new()),
        DecoderWrapper::Fuse(Fuse::new()),
    ]);

    let mut tok = Tokenizer::new(bpe);
    tok.with_normalizer(Some(normalizer));
    tok.with_pre_tokenizer(Some(pre));
    tok.with_decoder(Some(decoder));

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
            .filter_map(|(ttype, name)| {
                let is_special = matches!(
                    *ttype,
                    token_type::CONTROL | token_type::USER_DEFINED | token_type::UNKNOWN
                );
                if !is_special {
                    return None;
                }
                Some(AddedToken::from(name.clone(), true))
            })
            .collect();
        tok.add_special_tokens(&specials);
    } else {
        let mut specials: Vec<AddedToken> = Vec::new();
        for key in [
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.padding_token_id",
            "tokenizer.ggml.unknown_token_id",
            "tokenizer.ggml.mask_token_id",
        ] {
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

fn read_string_array(gguf: &GgufFile, key: &str) -> Result<Vec<String>> {
    let v = gguf
        .metadata(key)
        .with_context(|| format!("GGUF missing `{key}`"))?;
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

fn read_i32_array(gguf: &GgufFile, key: &str) -> Result<Vec<i32>> {
    let v = gguf
        .metadata(key)
        .with_context(|| format!("GGUF missing `{key}`"))?;
    let arr = match v {
        MetadataValue::Array(a) => a,
        other => bail!("`{key}` is not an array (got {other:?})"),
    };
    arr.iter()
        .enumerate()
        .map(|(i, e)| match e {
            MetadataValue::Int32(v) => Ok(*v),
            MetadataValue::Uint32(v) => Ok(*v as i32),
            MetadataValue::Int64(v) => Ok(*v as i32),
            MetadataValue::Uint64(v) => Ok(*v as i32),
            other => Err(anyhow!("`{key}`[{i}] is not an integer (got {other:?})")),
        })
        .collect()
}
