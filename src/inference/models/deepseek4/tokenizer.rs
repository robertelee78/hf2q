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
/// metadata, mirroring the same source consumed by the peer.
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
    use sha2::{Digest, Sha256};

    fn sha256_hex(bytes: &[u8]) -> String {
        hex::encode(Sha256::digest(bytes))
    }

    fn token_ids_sha256(ids: &[u32]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"hf2q-u32le-v1\0");
        for id in ids {
            hasher.update(id.to_le_bytes());
        }
        hex::encode(hasher.finalize())
    }

    fn sort_json_object_keys_recursively(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Array(values) => {
                for value in values {
                    sort_json_object_keys_recursively(value);
                }
            }
            serde_json::Value::Object(map) => {
                let mut entries = std::mem::take(map).into_iter().collect::<Vec<_>>();
                entries.sort_by(|left, right| left.0.cmp(&right.0));
                for (_, value) in &mut entries {
                    sort_json_object_keys_recursively(value);
                }
                map.extend(entries);
            }
            _ => {}
        }
    }

    fn render_agentic_request(
        template: &str,
        request: &crate::serve::api::schema::ChatCompletionRequest,
    ) -> String {
        crate::serve::api::engine::render_chat_prompt_with_tools(
            template,
            &request.messages,
            request.tools.as_deref(),
            crate::serve::template_supports_enable_thinking(template),
            request.chat_template_kwargs.as_ref(),
        )
        .expect("render request through the hf2q DeepSeek-V4 path")
    }

    fn render_agentic_request_with_tools_json(
        request: &crate::serve::api::schema::ChatCompletionRequest,
        tools: Option<serde_json::Value>,
    ) -> String {
        crate::serve::api::engine::render_deepseek_v4_prompt_with_serialized_tools(
            &request.messages,
            tools,
            request
                .chat_template_kwargs
                .as_ref()
                .and_then(|kwargs| kwargs.get("enable_thinking"))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            request.chat_template_kwargs.as_ref(),
        )
        .expect("render request with pre-serialized DeepSeek-V4 tools")
    }

    #[test]
    fn official_regex_sequence_is_pinned() {
        assert_eq!(NUMBER_REGEX, "\\p{N}{1,3}");
        assert_eq!(CJK_REGEX, "[一-龥぀-ゟ゠-ヿ]+");
        assert!(MAIN_REGEX.contains("[\\p{L}\\p{M}]+"));
        assert!(MAIN_REGEX.contains("\\s+(?!\\S)"));
    }

    #[test]
    #[ignore = "opens the release DeepSeek GGUF and exact agentic request fixture"]
    fn release_agentic_fixture_preserve_order_contract() {
        let gguf_path = std::env::var("HF2Q_DEEPSEEK4_GGUF")
            .expect("set HF2Q_DEEPSEEK4_GGUF to the release DeepSeek artifact");
        let request_path = std::env::var("HF2Q_DEEPSEEK4_AGENTIC_REQUEST_JSON")
            .expect("set HF2Q_DEEPSEEK4_AGENTIC_REQUEST_JSON to the exact request fixture");
        let contract_path = std::env::var("HF2Q_DEEPSEEK4_AGENTIC_PROMPT_CONTRACT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("scripts/fixtures/deepseek4-agentic-prompt-contract-v2.json")
            });
        let contract_bytes = std::fs::read(&contract_path)
            .unwrap_or_else(|error| panic!("read contract {}: {error}", contract_path.display()));
        let contract_sha256 = sha256_hex(&contract_bytes);
        let contract: serde_json::Value = serde_json::from_slice(&contract_bytes)
            .unwrap_or_else(|error| panic!("parse contract {}: {error}", contract_path.display()));
        let request_bytes = std::fs::read(&request_path)
            .unwrap_or_else(|error| panic!("read request {request_path:?}: {error}"));
        let request_sha256 = sha256_hex(&request_bytes);
        let expected_agent = contract["agents"]
            .as_array()
            .expect("prompt contract agents array")
            .iter()
            .find(|agent| agent["request_sha256"].as_str() == Some(request_sha256.as_str()))
            .unwrap_or_else(|| panic!("request {request_sha256} is absent from prompt contract"));
        let request: crate::serve::api::schema::ChatCompletionRequest =
            serde_json::from_slice(&request_bytes)
                .unwrap_or_else(|error| panic!("parse request {request_path:?}: {error}"));

        let gguf =
            GgufFile::open(std::path::Path::new(&gguf_path)).expect("open release DeepSeek GGUF");
        let chat_template = gguf
            .metadata_string("tokenizer.chat_template")
            .expect("release DeepSeek GGUF carries tokenizer.chat_template");
        assert_eq!(
            chat_template.len() as u64,
            contract["chat_template"]["bytes"]
                .as_u64()
                .expect("chat-template byte count")
        );
        assert_eq!(
            sha256_hex(chat_template.as_bytes()),
            contract["chat_template"]["sha256"]
                .as_str()
                .expect("chat-template digest")
        );
        let tokenizer = build_tokenizer_from_gguf(&gguf)
            .expect("build tokenizer from the release DeepSeek GGUF");
        let rendered = render_agentic_request(chat_template, &request);
        let encoded = tokenizer
            .encode(rendered.as_str(), false)
            .expect("tokenize insertion-ordered release prompt");

        let mut legacy_tools = request
            .tools
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .expect("serialize legacy DeepSeek-V4 tools");
        if let Some(tools) = legacy_tools.as_mut() {
            sort_json_object_keys_recursively(tools);
        }
        let legacy_rendered = render_agentic_request_with_tools_json(&request, legacy_tools);
        let legacy_encoded = tokenizer
            .encode(legacy_rendered.as_str(), false)
            .expect("tokenize legacy key-sorted release prompt");

        let rendered_sha256 = sha256_hex(rendered.as_bytes());
        let prompt_token_ids_sha256 = token_ids_sha256(encoded.get_ids());
        let legacy_rendered_sha256 = sha256_hex(legacy_rendered.as_bytes());
        let legacy_prompt_token_ids_sha256 = token_ids_sha256(legacy_encoded.get_ids());
        eprintln!(
            "release agentic fixture: request={request_sha256} rendered={rendered_sha256} tokens={} token_ids={prompt_token_ids_sha256} legacy_rendered={legacy_rendered_sha256} legacy_tokens={} legacy_token_ids={legacy_prompt_token_ids_sha256}",
            encoded.len(),
            legacy_encoded.len(),
        );

        assert_eq!(
            request_bytes.len() as u64,
            expected_agent["request_bytes"]
                .as_u64()
                .expect("agent request bytes")
        );
        assert_eq!(
            rendered_sha256,
            expected_agent["rendered_prompt_sha256"]
                .as_str()
                .expect("agent rendered prompt hash")
        );
        assert_eq!(
            prompt_token_ids_sha256,
            expected_agent["prompt_token_ids_sha256"]
                .as_str()
                .expect("agent prompt token hash")
        );
        assert_eq!(
            encoded.len() as u64,
            contract["serialization"]["expected_prompt_tokens"]
                .as_u64()
                .expect("expected prompt tokens")
        );
        assert_eq!(
            legacy_rendered_sha256,
            expected_agent["legacy_key_sorted_rendered_prompt_sha256"]
                .as_str()
                .expect("agent legacy rendered prompt hash")
        );
        assert_eq!(
            legacy_prompt_token_ids_sha256,
            expected_agent["legacy_key_sorted_prompt_token_ids_sha256"]
                .as_str()
                .expect("agent legacy prompt token hash")
        );
        assert_eq!(
            legacy_encoded.len() as u64,
            contract["serialization"]["legacy_rejected_prompt_tokens"]
                .as_u64()
                .expect("legacy rejected prompt tokens")
        );
        assert_ne!(rendered, legacy_rendered);

        if let Ok(receipt_path) = std::env::var("HF2Q_DEEPSEEK4_AGENTIC_CONTRACT_RECEIPT") {
            let receipt = serde_json::json!({
                "schema_version": 2,
                "status": "pass",
                "agent": expected_agent["agent"],
                "prompt_contract_sha256": contract_sha256,
                "serialization_policy": contract["serialization"]["policy"],
                "token_id_digest_encoding": contract["serialization"]["token_id_digest_encoding"],
                "request_sha256": request_sha256,
                "request_bytes": request_bytes.len(),
                "rendered_prompt_sha256": rendered_sha256,
                "rendered_prompt_bytes": rendered.len(),
                "prompt_token_ids_sha256": prompt_token_ids_sha256,
                "prompt_tokens": encoded.len(),
                "legacy_key_sorted_rendered_prompt_sha256": legacy_rendered_sha256,
                "legacy_key_sorted_rendered_prompt_bytes": legacy_rendered.len(),
                "legacy_key_sorted_prompt_token_ids_sha256": legacy_prompt_token_ids_sha256,
                "legacy_key_sorted_prompt_tokens": legacy_encoded.len(),
                "preserve_order_delta_proven": rendered != legacy_rendered,
            });
            let receipt_bytes = serde_json::to_vec_pretty(&receipt)
                .expect("serialize agentic prompt contract receipt");
            std::fs::write(&receipt_path, receipt_bytes)
                .unwrap_or_else(|error| panic!("write receipt {receipt_path:?}: {error}"));
        }
    }

    #[test]
    #[ignore = "opens the locally converted official GGUF tokenizer metadata"]
    fn official_embedded_tokenizer_matches_source_json() {
        let gguf_path = std::env::var("HF2Q_DEEPSEEK4_GGUF")
            .expect("set HF2Q_DEEPSEEK4_GGUF to the official converted artifact");
        let source_path = std::env::var("HF2Q_DEEPSEEK4_TOKENIZER_JSON")
            .expect("set HF2Q_DEEPSEEK4_TOKENIZER_JSON to the source tokenizer.json");
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
        if let Ok(prompt_path) = std::env::var("HF2Q_DEEPSEEK4_RENDERED_PROMPT") {
            let prompt = std::fs::read_to_string(&prompt_path)
                .unwrap_or_else(|error| panic!("read rendered prompt {prompt_path:?}: {error}"));
            if let Ok(request_path) = std::env::var("HF2Q_DEEPSEEK4_REQUEST_JSON") {
                let request_bytes = std::fs::read(&request_path)
                    .unwrap_or_else(|error| panic!("read request {request_path:?}: {error}"));
                let request: crate::serve::api::schema::ChatCompletionRequest =
                    serde_json::from_slice(&request_bytes)
                        .unwrap_or_else(|error| panic!("parse request {request_path:?}: {error}"));
                let template = crate::core::chat_templates::DEEPSEEK_V4_FLASH_0731;
                let rendered = crate::serve::api::engine::render_chat_prompt_with_tools(
                    template,
                    &request.messages,
                    request.tools.as_deref(),
                    crate::serve::template_supports_enable_thinking(template),
                    request.chat_template_kwargs.as_ref(),
                )
                .expect("render request through the hf2q DeepSeek-V4 path");
                if rendered != prompt {
                    let first_mismatch = rendered
                        .as_bytes()
                        .iter()
                        .zip(prompt.as_bytes())
                        .position(|(actual, expected)| actual != expected);
                    let offset = first_mismatch.unwrap_or(rendered.len().min(prompt.len()));
                    let start = offset.saturating_sub(120);
                    let end = offset.saturating_add(240);
                    panic!(
                        "hf2q native render drifted from supplied prompt: hf2q_len={} supplied_len={} first_mismatch={first_mismatch:?} hf2q_context={:?} supplied_context={:?}",
                        rendered.len(),
                        prompt.len(),
                        String::from_utf8_lossy(
                            &rendered.as_bytes()[start.min(rendered.len())..end.min(rendered.len())]
                        ),
                        String::from_utf8_lossy(
                            &prompt.as_bytes()[start.min(prompt.len())..end.min(prompt.len())]
                        ),
                    );
                }
            }
            let expected = source
                .encode(prompt.as_str(), false)
                .expect("source prompt encode");
            let actual = embedded
                .encode(prompt.as_str(), false)
                .expect("embedded prompt encode");
            let first_mismatch = actual
                .get_ids()
                .iter()
                .zip(expected.get_ids())
                .position(|(actual, expected)| actual != expected);
            assert_eq!(
                actual.get_ids(),
                expected.get_ids(),
                "rendered prompt token drift: embedded_len={} source_len={} first_mismatch={first_mismatch:?}",
                actual.len(),
                expected.len()
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
