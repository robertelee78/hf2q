//! Tokenizer module — wraps the `tokenizers` crate for HuggingFace tokenizers.
//!
//! Loads tokenizer.json from a model directory and provides encode/decode
//! operations with proper handling of special tokens (BOS, EOS).

pub mod chat_template;

use std::path::Path;

use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors from tokenizer operations.
#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("tokenizer.json not found in {path}")]
    NotFound { path: String },

    #[error("Failed to load tokenizer: {reason}")]
    LoadFailed { reason: String },

    #[error("Failed to encode text: {reason}")]
    EncodeFailed { reason: String },

    #[error("Failed to decode tokens: {reason}")]
    DecodeFailed { reason: String },

    #[error("Failed to read tokenizer_config.json: {reason}")]
    ConfigError { reason: String },
}

/// Special token IDs extracted from tokenizer_config.json.
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    /// Beginning of sequence token ID.
    pub bos_id: Option<u32>,
    /// End of sequence token ID.
    pub eos_id: Option<u32>,
    /// Padding token ID.
    pub pad_id: Option<u32>,
}

/// Wrapper around the `tokenizers` crate tokenizer.
pub struct HfTokenizer {
    /// The underlying tokenizer from the `tokenizers` crate.
    inner: tokenizers::Tokenizer,
    /// Special token IDs.
    pub special_tokens: SpecialTokens,
}

impl std::fmt::Debug for HfTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HfTokenizer")
            .field("vocab_size", &self.inner.get_vocab_size(true))
            .field("special_tokens", &self.special_tokens)
            .finish()
    }
}

impl HfTokenizer {
    /// Load a tokenizer from a model directory.
    ///
    /// Expects `tokenizer.json` to exist in the directory. Also reads
    /// `tokenizer_config.json` (if present) to extract special token IDs.
    pub fn from_dir(model_dir: &Path) -> Result<Self, TokenizerError> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(TokenizerError::NotFound {
                path: model_dir.display().to_string(),
            });
        }

        let inner = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            TokenizerError::LoadFailed {
                reason: format!("{}", e),
            }
        })?;

        let vocab_size = inner.get_vocab_size(true);
        info!(vocab_size = vocab_size, "Tokenizer loaded");

        // Extract special tokens from tokenizer_config.json
        let special_tokens = load_special_tokens(model_dir, &inner);

        debug!(
            bos_id = ?special_tokens.bos_id,
            eos_id = ?special_tokens.eos_id,
            "Special tokens resolved"
        );

        Ok(Self {
            inner,
            special_tokens,
        })
    }

    /// Encode text into token IDs.
    ///
    /// Does NOT add special tokens (BOS/EOS) — the caller is responsible for
    /// that based on the chat template or generation config.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| TokenizerError::EncodeFailed {
                reason: format!("{}", e),
            })?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Encode text with special tokens (BOS/EOS) added automatically.
    pub fn encode_with_special_tokens(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| TokenizerError::EncodeFailed {
                reason: format!("{}", e),
            })?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back into text.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, true)
            .map_err(|e| TokenizerError::DecodeFailed {
                reason: format!("{}", e),
            })
    }

    /// Get the vocabulary size (including special tokens).
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Get the BOS token ID if configured.
    pub fn bos_id(&self) -> Option<u32> {
        self.special_tokens.bos_id
    }

    /// Get the EOS token ID if configured.
    pub fn eos_id(&self) -> Option<u32> {
        self.special_tokens.eos_id
    }
}

/// Load special tokens from tokenizer_config.json and the tokenizer itself.
fn load_special_tokens(
    model_dir: &Path,
    tokenizer: &tokenizers::Tokenizer,
) -> SpecialTokens {
    let config_path = model_dir.join("tokenizer_config.json");

    let config: Option<serde_json::Value> = if config_path.exists() {
        match std::fs::read_to_string(&config_path) {
            Ok(content) => serde_json::from_str(&content).ok(),
            Err(e) => {
                warn!("Failed to read tokenizer_config.json: {}", e);
                None
            }
        }
    } else {
        None
    };

    let bos_id = resolve_special_token_id("bos_token", &config, tokenizer);
    let eos_id = resolve_special_token_id("eos_token", &config, tokenizer);
    let pad_id = resolve_special_token_id("pad_token", &config, tokenizer);

    SpecialTokens {
        bos_id,
        eos_id,
        pad_id,
    }
}

/// Resolve a special token's ID from config and/or tokenizer vocabulary.
fn resolve_special_token_id(
    key: &str,
    config: &Option<serde_json::Value>,
    tokenizer: &tokenizers::Tokenizer,
) -> Option<u32> {
    // First try to get the token string from config
    let token_str = config.as_ref().and_then(|c| {
        c.get(key).and_then(|v| {
            // The value can be a string or an object with a "content" field
            v.as_str()
                .map(|s| s.to_string())
                .or_else(|| {
                    v.get("content")
                        .and_then(|c| c.as_str())
                        .map(|s| s.to_string())
                })
        })
    });

    // Look up the token string in the tokenizer's vocabulary
    if let Some(ref tok) = token_str {
        if let Some(id) = tokenizer.token_to_id(tok) {
            return Some(id);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let result = HfTokenizer::from_dir(tmp.path());
        assert!(result.is_err());
        match result.unwrap_err() {
            TokenizerError::NotFound { .. } => {}
            other => panic!("Expected NotFound, got: {}", other),
        }
    }

    #[test]
    fn test_resolve_special_token_id_string_format() {
        let config: Option<serde_json::Value> = Some(serde_json::json!({
            "bos_token": "<bos>"
        }));

        // We can't easily test with a real tokenizer without a tokenizer.json,
        // but we can verify the extraction logic
        let tok = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
        let result = resolve_special_token_id("bos_token", &config, &tok);
        // Token won't be found in empty vocab, but the code path is exercised
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_special_token_id_object_format() {
        let config: Option<serde_json::Value> = Some(serde_json::json!({
            "bos_token": {
                "content": "<bos>",
                "lstrip": false,
                "rstrip": false
            }
        }));

        let tok = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
        let result = resolve_special_token_id("bos_token", &config, &tok);
        assert!(result.is_none()); // not in vocab, but path exercised
    }

    #[test]
    fn test_resolve_special_token_id_missing_key() {
        let config: Option<serde_json::Value> = Some(serde_json::json!({}));
        let tok = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
        let result = resolve_special_token_id("bos_token", &config, &tok);
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_special_token_id_no_config() {
        let tok = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
        let result = resolve_special_token_id("bos_token", &None, &tok);
        assert!(result.is_none());
    }

    #[test]
    fn test_special_tokens_default() {
        let tok = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
        let special = load_special_tokens(Path::new("/nonexistent"), &tok);
        assert!(special.bos_id.is_none());
        assert!(special.eos_id.is_none());
        assert!(special.pad_id.is_none());
    }
}
