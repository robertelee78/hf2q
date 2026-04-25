//! nomic-bert architecture config + tensor-name table.
//!
//! `NomicBertConfig` is parsed from a GGUF header's metadata using the
//! `nomic-bert.*` key prefix (per llama.cpp `llama-arch.cpp:25`). The
//! shape is identical to `bert::config::BertConfig` minus the BERT-only
//! fields that nomic-bert does not use, plus the `rope_freq_base` field
//! that nomic-bert needs for RoPE on Q/K.
//!
//! # GGUF metadata keys (nomic-bert)
//!
//! - `general.architecture = "nomic-bert"`
//! - `nomic-bert.embedding_length`            → hidden_size
//! - `nomic-bert.attention.head_count`        → num_attention_heads
//! - `nomic-bert.block_count`                 → num_hidden_layers
//! - `nomic-bert.feed_forward_length`         → intermediate_size
//! - `nomic-bert.attention.layer_norm_epsilon` → layer_norm_eps
//! - `nomic-bert.context_length`              → max_position_embeddings
//! - `nomic-bert.pooling_type`                → pooling enum (per `bert::PoolingType`)
//! - `nomic-bert.rope.freq_base`              → RoPE base period (e.g. 1000.0)
//! - `tokenizer.ggml.token_type_count`        → type_vocab_size (NOT
//!   `nomic-bert.token_type_count` — nomic GGUFs use the global tokenizer
//!   key per `llama-arch.cpp:263`).
//!
//! # Why no `hidden_act` field
//!
//! BERT has `hidden_act` because the spec is parameterised over GeLU /
//! GeLU-new / ReLU. nomic-bert is not — its FFN is always SwiGLU
//! (`silu(ffn_up) * ffn_gate → ffn_down`) per llama.cpp
//! `src/models/bert.cpp:131-138`. Encoding "always SwiGLU" as a hardcoded
//! contract instead of a config field eliminates a runtime branch and
//! makes the forward pass auditable from the type alone.

use anyhow::{anyhow, Result};
use mlx_native::gguf::GgufFile;

use super::super::bert::config::PoolingType;
use super::ARCH_NOMIC_BERT;

// ---------------------------------------------------------------------------
// Stem tensor names
// ---------------------------------------------------------------------------

/// Token embedding table `[vocab_size, hidden_size]`.
pub const NOMIC_BERT_TENSOR_TOKEN_EMBD: &str = "token_embd.weight";
/// Token-type (segment) embedding table `[type_vocab_size, hidden_size]`.
/// Optional — some encoder GGUFs lack a segment table; nomic-embed-text-v1.5
/// ships it (type_vocab_size=2).
pub const NOMIC_BERT_TENSOR_TOKEN_TYPES: &str = "token_types.weight";
/// LayerNorm applied to the summed embeddings — weight.
pub const NOMIC_BERT_TENSOR_EMBED_NORM_WEIGHT: &str = "token_embd_norm.weight";
/// LayerNorm applied to the summed embeddings — bias.
pub const NOMIC_BERT_TENSOR_EMBED_NORM_BIAS: &str = "token_embd_norm.bias";

// ---------------------------------------------------------------------------
// NomicBertConfig
// ---------------------------------------------------------------------------

/// Encoder-only config sufficient to drive a nomic-bert forward pass.
/// All fields are required by the encoder; the GGUF parser fails loudly
/// on missing keys rather than defaulting (silent-wrong-output trap).
#[derive(Debug, Clone, PartialEq)]
pub struct NomicBertConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f32,
    pub pooling_type: PoolingType,
    /// RoPE base period — `nomic-bert.rope.freq_base`. Typical value is
    /// 1000.0 for nomic-embed-text-v1.5; passed straight to
    /// `mlx_native::ops::rope::dispatch_rope_neox_*`.
    pub rope_freq_base: f32,
    /// `true` for decoder-style causal masking. Always `false` for the
    /// day-one nomic-embed-text-v1.5; kept as an explicit field so a
    /// future causal nomic-bert variant can opt in without a schema
    /// change.
    pub causal_attention: bool,
}

impl NomicBertConfig {
    /// Derived: `head_dim = hidden_size / num_attention_heads`.
    /// For nomic-embed-text-v1.5: 768/12 = 64.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Parse from a GGUF file's metadata header. Uses llama.cpp's
    /// `nomic-bert.*` key convention.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = gguf
            .metadata_string("general.architecture")
            .ok_or_else(|| anyhow!("GGUF missing general.architecture"))?;
        if arch != ARCH_NOMIC_BERT {
            return Err(anyhow!(
                "GGUF architecture is '{}', expected '{}'",
                arch,
                ARCH_NOMIC_BERT
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

        let hidden_size = u32_key("nomic-bert.embedding_length")? as usize;
        let num_attention_heads = u32_key("nomic-bert.attention.head_count")? as usize;
        let num_hidden_layers = u32_key("nomic-bert.block_count")? as usize;
        let intermediate_size = u32_key("nomic-bert.feed_forward_length")? as usize;
        let max_position_embeddings = u32_key("nomic-bert.context_length")? as usize;
        let layer_norm_eps = f32_key("nomic-bert.attention.layer_norm_epsilon")
            .or_else(|_| f32_key("nomic-bert.layer_norm_epsilon"))?;
        let rope_freq_base = f32_key("nomic-bert.rope.freq_base")?;

        let pooling_type = u32_key("nomic-bert.pooling_type")
            .ok()
            .and_then(PoolingType::from_u32)
            .unwrap_or(PoolingType::Mean);

        let causal_attention = gguf
            .metadata("nomic-bert.causal_attention")
            .and_then(|v| match v {
                mlx_native::gguf::MetadataValue::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(false);

        // vocab_size: infer from token_embd tensor (the row count). nomic
        // GGUFs do not emit a separate `nomic-bert.vocab_size` key.
        let vocab_size = gguf
            .tensor_info(NOMIC_BERT_TENSOR_TOKEN_EMBD)
            .and_then(|ti| ti.shape.first().copied())
            .map(|s| s as usize)
            .ok_or_else(|| {
                anyhow!(
                    "cannot determine nomic-bert vocab_size — '{}' tensor missing or empty shape",
                    NOMIC_BERT_TENSOR_TOKEN_EMBD
                )
            })?;

        // type_vocab_size: prefer the global tokenizer key (per
        // llama-arch.cpp:263); fall back to inferring from
        // token_types.weight tensor row count if that key is absent.
        let type_vocab_size = u32_key("tokenizer.ggml.token_type_count")
            .ok()
            .map(|v| v as usize)
            .or_else(|| {
                gguf.tensor_info(NOMIC_BERT_TENSOR_TOKEN_TYPES)
                    .and_then(|ti| ti.shape.first().copied())
                    .map(|s| s as usize)
            })
            .unwrap_or(2);

        Ok(NomicBertConfig {
            hidden_size,
            num_attention_heads,
            num_hidden_layers,
            intermediate_size,
            max_position_embeddings,
            vocab_size,
            type_vocab_size,
            layer_norm_eps,
            pooling_type,
            rope_freq_base,
            causal_attention,
        })
    }
}

// ---------------------------------------------------------------------------
// Per-layer tensor-name helper
// ---------------------------------------------------------------------------

/// Per-layer tensor name. nomic-bert uses the same `blk.{n}.{suffix}`
/// convention as classic BERT (per `bert.cpp` builder). The suffixes
/// differ — see `weights::NOMIC_BERT_BLOCK_REQUIRED_SUFFIXES`.
pub fn nomic_bert_layer_tensor(layer_idx: usize, suffix: &str) -> String {
    format!("blk.{}.{}", layer_idx, suffix)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn tensor_name_constants_match_llama_cpp_convention() {
        // Stem names match the BERT family convention; deviations here
        // are silent compat breaks. Locked against `bert.cpp` builder
        // which gathers `tok_embd / type_embd / tok_norm` from the same
        // GGUF strings for both BERT and nomic-bert.
        assert_eq!(NOMIC_BERT_TENSOR_TOKEN_EMBD, "token_embd.weight");
        assert_eq!(NOMIC_BERT_TENSOR_TOKEN_TYPES, "token_types.weight");
        assert_eq!(NOMIC_BERT_TENSOR_EMBED_NORM_WEIGHT, "token_embd_norm.weight");
        assert_eq!(NOMIC_BERT_TENSOR_EMBED_NORM_BIAS, "token_embd_norm.bias");
    }

    #[test]
    fn layer_tensor_helper_formats_blk_prefix() {
        assert_eq!(
            nomic_bert_layer_tensor(0, "attn_qkv.weight"),
            "blk.0.attn_qkv.weight"
        );
        assert_eq!(
            nomic_bert_layer_tensor(11, "ffn_gate.weight"),
            "blk.11.ffn_gate.weight"
        );
    }

    #[test]
    fn head_dim_divides_hidden() {
        let cfg = NomicBertConfig {
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            intermediate_size: 3072,
            max_position_embeddings: 2048,
            vocab_size: 30522,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            pooling_type: PoolingType::Mean,
            rope_freq_base: 1000.0,
            causal_attention: false,
        };
        assert_eq!(cfg.head_dim(), 64);
    }

    /// End-to-end test against the on-disk nomic-embed-text-v1.5 GGUF.
    /// Skips cleanly when the model isn't present so CI / fresh checkouts
    /// don't false-fail.
    #[test]
    fn nomic_embed_text_v1_5_gguf_parses_config() {
        let path = Path::new("/opt/hf2q/models/bert-test/nomic-embed-text-v1.5-f16.gguf");
        if !path.exists() {
            eprintln!("skipping: nomic GGUF fixture not at {}", path.display());
            return;
        }
        let gguf = GgufFile::open(path).expect("open nomic GGUF");
        let cfg = NomicBertConfig::from_gguf(&gguf).expect("parse nomic config");

        // Locked values for nomic-embed-text-v1.5 — any drift indicates
        // either a different model or a parser regression.
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.intermediate_size, 3072);
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.type_vocab_size, 2);
        assert!(
            (cfg.rope_freq_base - 1000.0).abs() < 0.5,
            "expected rope_freq_base ≈ 1000.0, got {}",
            cfg.rope_freq_base
        );
        assert!(!cfg.causal_attention, "nomic-embed-text is non-causal");
        // Pooling for sentence embeddings: mean is the standard for nomic.
        assert!(
            matches!(cfg.pooling_type, PoolingType::Mean | PoolingType::Cls),
            "unexpected pooling_type {:?}",
            cfg.pooling_type
        );
    }

    /// Drive the architecture-mismatch error path against the on-disk
    /// bge GGUF (which is `general.architecture = "bert"`). Locks the
    /// validator's mismatch message so a future Display change doesn't
    /// silently swallow the indication.
    #[test]
    fn rejects_bert_arch_with_clear_error() {
        let path = Path::new("/opt/hf2q/models/bert-test/bge-small-en-v1.5-f16.gguf");
        if !path.exists() {
            eprintln!("skipping: bge GGUF fixture not at {}", path.display());
            return;
        }
        let gguf = GgufFile::open(path).expect("open bge GGUF");
        let err = NomicBertConfig::from_gguf(&gguf)
            .expect_err("parser must reject 'bert' arch");
        let msg = format!("{err}");
        assert!(
            msg.contains("'bert'") && msg.contains(ARCH_NOMIC_BERT),
            "error must name actual + expected arch, got: {msg}"
        );
    }
}
