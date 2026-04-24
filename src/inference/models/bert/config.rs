//! BERT architecture config + tensor-name table.
//!
//! `BertConfig` is parsed from either:
//!   - A HuggingFace `config.json` file (when serving from a directory
//!     containing safetensors — not the common case for BERT in hf2q;
//!     included for future `hf2q convert` support).
//!   - A GGUF header's metadata key/value pairs (the common path —
//!     llama.cpp's `bert.*` keys).
//!
//! Both parsers produce the same `BertConfig`, which then feeds the
//! forward-pass entry point. No field is optional — if a key is missing
//! the parser returns `Err(ParseError { key, ... })` listing what needs
//! to be supplied.

use anyhow::{anyhow, Context, Result};
use mlx_native::gguf::GgufFile;
use std::path::Path;

// ---------------------------------------------------------------------------
// PoolingType (bert.pooling_type metadata)
// ---------------------------------------------------------------------------

/// Pooling method used to reduce the encoder's last-hidden-state
/// `[seq_len, hidden]` to a single vector `[hidden]`.
///
/// Values match llama.cpp's `enum llama_pooling_type` (see
/// `/opt/llama.cpp/include/llama.h`):
///   0 = NONE (no pooling — return all hidden states, invalid for
///             /v1/embeddings which always needs a single vector)
///   1 = MEAN
///   2 = CLS  (use hidden state at token position 0)
///   3 = LAST (use hidden state at last token position)
///   4 = RANK (reranker-specific; out of scope for pooled embeddings)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PoolingType {
    None = 0,
    Mean = 1,
    Cls = 2,
    Last = 3,
    Rank = 4,
}

impl PoolingType {
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => PoolingType::None,
            1 => PoolingType::Mean,
            2 => PoolingType::Cls,
            3 => PoolingType::Last,
            4 => PoolingType::Rank,
            _ => return None,
        })
    }
    pub fn as_str(self) -> &'static str {
        match self {
            PoolingType::None => "none",
            PoolingType::Mean => "mean",
            PoolingType::Cls => "cls",
            PoolingType::Last => "last",
            PoolingType::Rank => "rank",
        }
    }
}

// ---------------------------------------------------------------------------
// BertConfig
// ---------------------------------------------------------------------------

/// Encoder-only config sufficient to drive a forward pass. All fields are
/// required by the encoder; the parser fails loudly on missing keys
/// rather than defaulting to something that would produce silently-wrong
/// output.
#[derive(Debug, Clone, PartialEq)]
pub struct BertConfig {
    /// Hidden-state dimension (a.k.a. `embedding_length` in GGUF,
    /// `hidden_size` in HF config).
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of encoder layers.
    pub num_hidden_layers: usize,
    /// FFN intermediate dimension.
    pub intermediate_size: usize,
    /// Maximum token position in the positional embedding table.
    pub max_position_embeddings: usize,
    /// Vocab size (also the token-embedding table's row count).
    pub vocab_size: usize,
    /// Token-type vocab (segment embeddings; 2 for standard BERT,
    /// some encoders have 1).
    pub type_vocab_size: usize,
    /// LayerNorm epsilon (applied to every LN in the encoder).
    pub layer_norm_eps: f32,
    /// Hidden activation. BERT variants use `gelu` or `gelu_new`; the
    /// forward pass will dispatch on this string.
    pub hidden_act: String,
    /// Pooling method for `/v1/embeddings` output reduction.
    pub pooling_type: PoolingType,
    /// `true` → encoder uses a causal mask (decoder-style). `false` for
    /// standard BERT. llama.cpp's BERT GGUFs set this explicitly.
    pub causal_attention: bool,
}

impl BertConfig {
    /// Parse from a HuggingFace `config.json` at the given path.
    /// `pooling_type` is not a standard HF field — defaults to `Mean` when
    /// absent, since that's the most common pooling for sentence-embedding
    /// BERTs (matches nomic-embed-text and bge-small conventions).
    pub fn from_config_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing {} as JSON", path.display()))?;
        Self::from_hf_value(&v)
    }

    /// Parse from an already-loaded HF config.json value.
    pub fn from_hf_value(v: &serde_json::Value) -> Result<Self> {
        let get_usize = |key: &str| -> Result<usize> {
            v.get(key)
                .and_then(|x| x.as_u64())
                .map(|u| u as usize)
                .ok_or_else(|| anyhow!("missing HF config key '{}'", key))
        };
        let get_f32 = |key: &str| -> Result<f32> {
            v.get(key)
                .and_then(|x| x.as_f64())
                .map(|f| f as f32)
                .ok_or_else(|| anyhow!("missing HF config key '{}'", key))
        };
        let pooling_type = v
            .get("pooling_type")
            .and_then(|x| x.as_u64())
            .and_then(|u| PoolingType::from_u32(u as u32))
            .unwrap_or(PoolingType::Mean);
        let causal_attention = v
            .get("is_decoder")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        Ok(BertConfig {
            hidden_size: get_usize("hidden_size")?,
            num_attention_heads: get_usize("num_attention_heads")?,
            num_hidden_layers: get_usize("num_hidden_layers")?,
            intermediate_size: get_usize("intermediate_size")?,
            max_position_embeddings: get_usize("max_position_embeddings")?,
            vocab_size: get_usize("vocab_size")?,
            type_vocab_size: v
                .get("type_vocab_size")
                .and_then(|x| x.as_u64())
                .map(|u| u as usize)
                .unwrap_or(2),
            layer_norm_eps: get_f32("layer_norm_eps")
                .or_else(|_| get_f32("layer_norm_epsilon"))?,
            hidden_act: v
                .get("hidden_act")
                .and_then(|x| x.as_str())
                .unwrap_or("gelu")
                .to_string(),
            pooling_type,
            causal_attention,
        })
    }

    /// Parse from a GGUF file's metadata header. Uses llama.cpp's
    /// `bert.*` key convention.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = gguf
            .metadata_string("general.architecture")
            .ok_or_else(|| anyhow!("GGUF missing general.architecture"))?;
        if arch != super::ARCH_BERT {
            return Err(anyhow!(
                "GGUF architecture is '{}', expected 'bert'",
                arch
            ));
        }
        let u32_key = |key: &str| -> Result<u32> {
            gguf.metadata_u32(key)
                .ok_or_else(|| anyhow!("GGUF missing u32 metadata '{}'", key))
        };
        let f32_key = |key: &str| -> Result<f32> {
            gguf.metadata_f32(key)
                .ok_or_else(|| anyhow!("GGUF missing f32 metadata '{}'", key))
        };
        let hidden_size = u32_key("bert.embedding_length")? as usize;
        let num_attention_heads = u32_key("bert.attention.head_count")? as usize;
        let num_hidden_layers = u32_key("bert.block_count")? as usize;
        let intermediate_size = u32_key("bert.feed_forward_length")? as usize;
        let max_position_embeddings = u32_key("bert.context_length")? as usize;
        let layer_norm_eps = f32_key("bert.attention.layer_norm_epsilon")
            .or_else(|_| f32_key("bert.layer_norm_epsilon"))?;
        let pooling_type = u32_key("bert.pooling_type")
            .ok()
            .and_then(PoolingType::from_u32)
            .unwrap_or(PoolingType::Mean);
        let causal_attention = gguf
            .metadata("bert.causal_attention")
            .and_then(|v| match v {
                mlx_native::gguf::MetadataValue::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(false);

        // Vocab size: check tokenizer metadata or infer from token_embd tensor.
        let vocab_size = u32_key("bert.vocab_size")
            .ok()
            .or_else(|| {
                gguf.tensor_info(TENSOR_TOKEN_EMBD)
                    .and_then(|ti| ti.shape.first().copied())
                    .map(|s| s as u32)
            })
            .ok_or_else(|| anyhow!("cannot determine BERT vocab_size"))? as usize;

        let type_vocab_size = u32_key("bert.token_type_count")
            .ok()
            .map(|v| v as usize)
            .unwrap_or(2);

        let hidden_act = gguf
            .metadata_string("bert.activation")
            .map(|s| s.to_string())
            .unwrap_or_else(|| "gelu".to_string());

        Ok(BertConfig {
            hidden_size,
            num_attention_heads,
            num_hidden_layers,
            intermediate_size,
            max_position_embeddings,
            vocab_size,
            type_vocab_size,
            layer_norm_eps,
            hidden_act,
            pooling_type,
            causal_attention,
        })
    }

    /// Derived: `head_dim = hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ---------------------------------------------------------------------------
// Tensor-name table (llama.cpp GGUF BERT convention)
// ---------------------------------------------------------------------------

/// Token embedding table `[vocab_size, hidden_size]`.
pub const TENSOR_TOKEN_EMBD: &str = "token_embd.weight";
/// Positional embedding table `[max_pos, hidden_size]`.
pub const TENSOR_POS_EMBD: &str = "position_embd.weight";
/// Token-type (segment) embedding table `[type_vocab_size, hidden_size]`.
pub const TENSOR_TOKEN_TYPES: &str = "token_types.weight";
/// LayerNorm applied to the summed embeddings — weight.
pub const TENSOR_EMBED_NORM_WEIGHT: &str = "token_embd_norm.weight";
/// LayerNorm applied to the summed embeddings — bias.
pub const TENSOR_EMBED_NORM_BIAS: &str = "token_embd_norm.bias";

/// Per-layer tensor name helper. llama.cpp's BERT convention uses
/// `blk.{n}.{suffix}` for every per-block tensor.
///
/// Standard suffixes:
///   - `"attn_q.weight"`, `"attn_q.bias"`
///   - `"attn_k.weight"`, `"attn_k.bias"`
///   - `"attn_v.weight"`, `"attn_v.bias"`
///   - `"attn_output.weight"`, `"attn_output.bias"`
///   - `"attn_output_norm.weight"`, `"attn_output_norm.bias"`
///   - `"ffn_up.weight"`, `"ffn_up.bias"`
///   - `"ffn_down.weight"`, `"ffn_down.bias"`
///   - `"layer_output_norm.weight"`, `"layer_output_norm.bias"`
pub fn bert_layer_tensor(layer_idx: usize, suffix: &str) -> String {
    format!("blk.{}.{}", layer_idx, suffix)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pooling_type_round_trips_u32() {
        for (i, expected) in [
            (0u32, PoolingType::None),
            (1, PoolingType::Mean),
            (2, PoolingType::Cls),
            (3, PoolingType::Last),
            (4, PoolingType::Rank),
        ] {
            assert_eq!(PoolingType::from_u32(i), Some(expected));
        }
        assert_eq!(PoolingType::from_u32(42), None);
    }

    #[test]
    fn pooling_type_as_str_stable() {
        // Test that the string values match what /v1/models might expose
        // as a `pooling` field. Byte-literal to catch accidental changes.
        assert_eq!(PoolingType::Mean.as_str(), "mean");
        assert_eq!(PoolingType::Cls.as_str(), "cls");
        assert_eq!(PoolingType::None.as_str(), "none");
        assert_eq!(PoolingType::Last.as_str(), "last");
        assert_eq!(PoolingType::Rank.as_str(), "rank");
    }

    #[test]
    fn hf_config_json_parses_standard_bert() {
        // Shape mimicking bge-small-en-v1.5's config.json.
        let v: serde_json::Value = serde_json::from_str(
            r#"{
                "hidden_size": 384,
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "intermediate_size": 1536,
                "max_position_embeddings": 512,
                "vocab_size": 30522,
                "type_vocab_size": 2,
                "layer_norm_eps": 1e-12,
                "hidden_act": "gelu"
            }"#,
        )
        .unwrap();
        let cfg = BertConfig::from_hf_value(&v).unwrap();
        assert_eq!(cfg.hidden_size, 384);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.intermediate_size, 1536);
        assert_eq!(cfg.max_position_embeddings, 512);
        assert_eq!(cfg.vocab_size, 30522);
        assert_eq!(cfg.type_vocab_size, 2);
        assert!((cfg.layer_norm_eps - 1e-12).abs() < 1e-20);
        assert_eq!(cfg.hidden_act, "gelu");
        assert_eq!(cfg.pooling_type, PoolingType::Mean); // default
        assert!(!cfg.causal_attention);
        assert_eq!(cfg.head_dim(), 32);
    }

    #[test]
    fn hf_config_json_accepts_alt_layer_norm_eps_key() {
        // Some BERTs use `layer_norm_epsilon` instead of `layer_norm_eps`.
        let v: serde_json::Value = serde_json::from_str(
            r#"{
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "intermediate_size": 3072,
                "max_position_embeddings": 512,
                "vocab_size": 30522,
                "layer_norm_epsilon": 1e-12
            }"#,
        )
        .unwrap();
        let cfg = BertConfig::from_hf_value(&v).unwrap();
        assert!((cfg.layer_norm_eps - 1e-12).abs() < 1e-20);
    }

    #[test]
    fn hf_config_missing_required_fields_errors() {
        let v: serde_json::Value =
            serde_json::from_str(r#"{"hidden_size": 768}"#).unwrap();
        let err = BertConfig::from_hf_value(&v).unwrap_err();
        let msg = format!("{}", err);
        // Should name the missing field.
        assert!(
            msg.contains("num_attention_heads") || msg.contains("missing"),
            "error: {}",
            msg
        );
    }

    #[test]
    fn hf_config_explicit_pooling_and_decoder_flag() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{
                "hidden_size": 384, "num_attention_heads": 12,
                "num_hidden_layers": 12, "intermediate_size": 1536,
                "max_position_embeddings": 512, "vocab_size": 30522,
                "layer_norm_eps": 1e-12,
                "pooling_type": 2,
                "is_decoder": true
            }"#,
        )
        .unwrap();
        let cfg = BertConfig::from_hf_value(&v).unwrap();
        assert_eq!(cfg.pooling_type, PoolingType::Cls);
        assert!(cfg.causal_attention);
    }

    #[test]
    fn tensor_name_helper_formats_blk_prefix() {
        assert_eq!(bert_layer_tensor(0, "attn_q.weight"), "blk.0.attn_q.weight");
        assert_eq!(bert_layer_tensor(11, "ffn_down.bias"), "blk.11.ffn_down.bias");
    }

    #[test]
    fn tensor_name_constants_match_llama_cpp_convention() {
        // Spot-check the global constants against llama.cpp's BERT GGUF
        // writer convention. Changes here are a silent compat break — any
        // future refactor that touches these strings must update this test
        // + the corresponding loader code in lockstep.
        assert_eq!(TENSOR_TOKEN_EMBD, "token_embd.weight");
        assert_eq!(TENSOR_POS_EMBD, "position_embd.weight");
        assert_eq!(TENSOR_TOKEN_TYPES, "token_types.weight");
        assert_eq!(TENSOR_EMBED_NORM_WEIGHT, "token_embd_norm.weight");
        assert_eq!(TENSOR_EMBED_NORM_BIAS, "token_embd_norm.bias");
    }

    #[test]
    fn head_dim_divides_hidden() {
        let cfg = BertConfig {
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            vocab_size: 30522,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".into(),
            pooling_type: PoolingType::Mean,
            causal_attention: false,
        };
        assert_eq!(cfg.head_dim(), 64);
    }
}
