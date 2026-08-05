//! GGUF-driven DeepSeek-V4 GPT-2 BPE tokenizer.
//!
//! The pre-tokenizer is the exact three-Split DeepSeek-V3 sequence shipped by
//! the official 0731 checkpoint, followed by byte-level encoding. No Python or
//! external tokenizer process participates at runtime.

use ahash::AHashMap;
use anyhow::{anyhow, bail, Result};
use mlx_native::gguf::{GgufFile, MetadataValue};
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::Split;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::byte_level::ByteLevel as ByteLevelProcessor;
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

pub const NUMBER_REGEX: &str = "\\p{N}{1,3}";
pub const CJK_REGEX: &str = "[一-龥぀-ゟ゠-ヿ]+";
pub const MAIN_REGEX: &str = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";

mod token_type {
    pub const CONTROL: i32 = 3;
    pub const USER_DEFINED: i32 = 4;
}

/// Build the tokenizer solely from the converted artifact's embedded GGUF
/// metadata, mirroring the same source consumed by llama.cpp.
pub fn build_tokenizer_from_gguf(gguf: &GgufFile) -> Result<Tokenizer> {
    let model = gguf
        .metadata_string("tokenizer.ggml.model")
        .ok_or_else(|| anyhow!("GGUF missing `tokenizer.ggml.model`"))?;
    if model != "gpt2" {
        bail!("DeepSeek-V4 tokenizer model must be gpt2, got {model:?}");
    }
    let pre = gguf
        .metadata_string("tokenizer.ggml.pre")
        .ok_or_else(|| anyhow!("GGUF missing `tokenizer.ggml.pre`"))?;
    if pre != "deepseek-v3" {
        bail!("DeepSeek-V4 tokenizer pre-type must be deepseek-v3, got {pre:?}");
    }

    let tokens = read_string_array(gguf, "tokenizer.ggml.tokens")?;
    if tokens.is_empty() {
        bail!("`tokenizer.ggml.tokens` is empty");
    }
    let tokens: Vec<String> = tokens
        .into_iter()
        .enumerate()
        .map(|(index, token)| {
            if token.is_empty() {
                format!("[EMPTY_{index}]")
            } else {
                token
            }
        })
        .collect();
    let vocab: AHashMap<String, u32> = tokens
        .iter()
        .enumerate()
        .map(|(index, token)| (token.clone(), index as u32))
        .collect();
    if vocab.len() != tokens.len() {
        bail!(
            "duplicate DeepSeek-V4 GGUF tokens: {} unique vs {} entries",
            vocab.len(),
            tokens.len()
        );
    }
    let merges = read_string_array(gguf, "tokenizer.ggml.merges")?
        .into_iter()
        .enumerate()
        .map(|(index, merge)| {
            let mut halves = merge.splitn(2, ' ');
            let left = halves
                .next()
                .ok_or_else(|| anyhow!("merge[{index}] has no left half"))?;
            let right = halves
                .next()
                .ok_or_else(|| anyhow!("merge[{index}] = {merge:?} has no separator"))?;
            Ok((left.to_string(), right.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|error| anyhow!("build DeepSeek-V4 BPE: {error}"))?;

    let splits = [NUMBER_REGEX, CJK_REGEX, MAIN_REGEX]
        .into_iter()
        .map(|regex| {
            Split::new(
                tokenizers::pre_tokenizers::split::SplitPattern::Regex(regex.to_string()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .map(PreTokenizerWrapper::Split)
            .map_err(|error| anyhow!("compile DeepSeek-V4 tokenizer regex: {error}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut pre_tokenizers = splits;
    pre_tokenizers.push(PreTokenizerWrapper::ByteLevel(ByteLevel::new(
        false, true, false,
    )));

    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(Sequence::new(pre_tokenizers)));
    tokenizer.with_post_processor(Some(ByteLevelProcessor::new(true, false, true)));
    tokenizer.with_decoder(Some(ByteLevelDecoder::new(true, true, true)));
    register_atomic_tokens(gguf, &tokens, &mut tokenizer)?;
    Ok(tokenizer)
}

fn register_atomic_tokens(
    gguf: &GgufFile,
    tokens: &[String],
    tokenizer: &mut Tokenizer,
) -> Result<()> {
    let token_types = read_i32_array(gguf, "tokenizer.ggml.token_type").ok();
    if let Some(types) = token_types {
        if types.len() != tokens.len() {
            bail!(
                "tokenizer.ggml.token_type length {} != tokens length {}",
                types.len(),
                tokens.len()
            );
        }
        let control = types
            .iter()
            .zip(tokens)
            .filter(|(kind, _)| **kind == token_type::CONTROL)
            .map(|(_, token)| AddedToken::from(token.clone(), true))
            .collect::<Vec<_>>();
        let user_defined = types
            .iter()
            .zip(tokens)
            .filter(|(kind, _)| **kind == token_type::USER_DEFINED)
            .map(|(_, token)| AddedToken::from(token.clone(), false))
            .collect::<Vec<_>>();
        tokenizer.add_special_tokens(&control);
        tokenizer.add_tokens(&user_defined);
    } else {
        let control = [
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.padding_token_id",
        ]
        .into_iter()
        .filter_map(|key| gguf.metadata_u32(key))
        .filter_map(|id| tokens.get(id as usize))
        .map(|token| AddedToken::from(token.clone(), true))
        .collect::<Vec<_>>();
        tokenizer.add_special_tokens(&control);
    }
    Ok(())
}

fn read_string_array(gguf: &GgufFile, key: &str) -> Result<Vec<String>> {
    let value = gguf
        .metadata(key)
        .ok_or_else(|| anyhow!("GGUF missing `{key}`"))?;
    let MetadataValue::Array(entries) = value else {
        bail!("`{key}` is not an array (got {value:?})");
    };
    entries
        .iter()
        .enumerate()
        .map(|(index, entry)| match entry {
            MetadataValue::String(value) => Ok(value.clone()),
            other => Err(anyhow!("`{key}`[{index}] is not a string ({other:?})")),
        })
        .collect()
}

fn read_i32_array(gguf: &GgufFile, key: &str) -> Result<Vec<i32>> {
    let value = gguf
        .metadata(key)
        .ok_or_else(|| anyhow!("GGUF missing `{key}`"))?;
    let MetadataValue::Array(entries) = value else {
        bail!("`{key}` is not an array (got {value:?})");
    };
    entries
        .iter()
        .enumerate()
        .map(|(index, entry)| match entry {
            MetadataValue::Int32(value) => Ok(*value),
            MetadataValue::Uint32(value) => Ok(*value as i32),
            MetadataValue::Int16(value) => Ok(*value as i32),
            MetadataValue::Uint16(value) => Ok(*value as i32),
            MetadataValue::Int8(value) => Ok(*value as i32),
            MetadataValue::Uint8(value) => Ok(*value as i32),
            other => Err(anyhow!("`{key}`[{index}] is not an integer ({other:?})")),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_regex_sequence_is_pinned() {
        assert_eq!(NUMBER_REGEX, "\\p{N}{1,3}");
        assert_eq!(CJK_REGEX, "[一-龥぀-ゟ゠-ヿ]+");
        assert!(MAIN_REGEX.contains("[\\p{L}\\p{M}]+"));
        assert!(MAIN_REGEX.contains("\\s+(?!\\S)"));
    }

    #[test]
    #[ignore = "opens the locally converted official GGUF tokenizer metadata"]
    fn official_embedded_tokenizer_matches_source_json() {
        let gguf_path = std::env::var("HF2Q_DEEPSEEK4_GGUF")
            .expect("set HF2Q_DEEPSEEK4_GGUF to the official converted artifact");
        let source_path = std::env::var("HF2Q_DEEPSEEK4_TOKENIZER_JSON").unwrap_or_else(|_| {
            "/opt/hf2q/cache/deepseek-v4-flash-0731-source/tokenizer.json".into()
        });
        let gguf = GgufFile::open(std::path::Path::new(&gguf_path)).expect("open official GGUF");
        let embedded = build_tokenizer_from_gguf(&gguf).expect("build embedded tokenizer");
        let source = Tokenizer::from_file(source_path).expect("load official tokenizer.json");
        for text in [
            "Hello, world!",
            "Numbers 1 12 123 1234 12345",
            "汉字とカタカナ mixed text",
            "<｜begin▁of▁sentence｜><｜User｜>Why?<｜Assistant｜></think>",
            "e\u{301}lan\n\n punctuation?!",
        ] {
            let expected = source.encode(text, false).expect("source encode");
            let actual = embedded.encode(text, false).expect("embedded encode");
            assert_eq!(
                actual.get_ids(),
                expected.get_ids(),
                "token drift for {text:?}"
            );
        }
        let assistant = embedded
            .token_to_id("<｜Assistant｜>")
            .expect("assistant atom exists");
        assert_eq!(assistant, 128_804);
        assert_eq!(
            embedded.decode(&[assistant], false).expect("decode atom"),
            "<｜Assistant｜>",
            "prompt atoms must survive the runtime's skip_special=false decode"
        );
    }
}
