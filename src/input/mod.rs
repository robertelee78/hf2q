//! Input module — owns all external model I/O.
//!
//! Nothing outside this module touches raw model files directly.
//! Sub-modules:
//! - `config_parser`: HF config.json -> ModelMetadata
//! - `safetensors`: Streaming mmap shard reader -> TensorMap
//! - `hf_download`: HF Hub download (Epic 3)

pub mod config_parser;
pub mod hf_download;
pub mod safetensors;

use std::path::Path;

use thiserror::Error;

use crate::ir::{ModelMetadata, TensorMap};
use crate::progress::ProgressReporter;

/// Errors from input operations.
#[derive(Error, Debug)]
pub enum InputError {
    #[error("Input directory does not exist: {path}")]
    DirectoryNotFound { path: String },

    #[error("No config.json found in {path}")]
    NoConfig { path: String },

    #[allow(dead_code)]
    #[error("No safetensors files found in {path}")]
    NoSafetensors { path: String },

    #[error("Config parse error: {0}")]
    ConfigParse(#[from] config_parser::ConfigParseError),

    #[error("Safetensors read error: {0}")]
    SafetensorsRead(#[from] safetensors::SafetensorsError),

    #[error("HF download error: {0}")]
    HfDownload(#[from] hf_download::DownloadError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Read a model from a local directory: parse config and load tensor map.
pub fn read_model(
    input_dir: &Path,
    progress: &ProgressReporter,
) -> Result<(ModelMetadata, TensorMap), InputError> {
    if !input_dir.exists() {
        return Err(InputError::DirectoryNotFound {
            path: input_dir.display().to_string(),
        });
    }

    let config_path = input_dir.join("config.json");
    if !config_path.exists() {
        return Err(InputError::NoConfig {
            path: input_dir.display().to_string(),
        });
    }

    // Parse model metadata from config.json
    let metadata = config_parser::parse_config(&config_path)?;

    // Load tensors from safetensors files
    let tensor_map = safetensors::read_tensors(input_dir, progress)?;

    Ok((metadata, tensor_map))
}

/// Detect the actual (de-padded) vocabulary size from `tokenizer.json`.
///
/// HF safetensors often pad the embedding tensor to a multiple of 64 / 128
/// for hardware-friendly shapes (e.g. Qwen3.6-27B emits `vocab_size: 248320`
/// in `config.json` while the tokenizer only owns 248044 unique token ids).
/// llama.cpp's loader compares the embedding tensor's row count against the
/// emitted `tokenizer.ggml.tokens` array length and rejects mismatches with
///
/// > tensor 'token_embd.weight' has wrong shape; expected H, T, got H, P
///
/// where T is the de-padded count and P is the padded one.
///
/// Returns `Ok(Some(true_vocab))` when `tokenizer.json` is parseable and the
/// max token id + 1 differs from `metadata.vocab_size`.  Returns `Ok(None)`
/// when there's no padding to remove (or no tokenizer.json — caller decides
/// whether absence is fatal).
pub fn detect_padded_vocab(
    input_dir: &Path,
    metadata: &ModelMetadata,
) -> Result<Option<u64>, InputError> {
    let tokenizer_path = input_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Ok(None);
    }

    let raw = std::fs::read_to_string(&tokenizer_path)?;
    let parsed: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };

    // tokenizer.json schema: { "model": { "vocab": { "<token>": <id>, ... } }, "added_tokens": [...] }
    let model_section = match parsed.get("model") {
        Some(v) => v,
        None => return Ok(None),
    };
    let vocab_obj = match model_section.get("vocab").and_then(|v| v.as_object()) {
        Some(o) => o,
        None => return Ok(None),
    };

    // The de-padded vocab MUST match what the GGUF emit puts into
    // `tokenizer.ggml.tokens` — llama.cpp's loader compares the embedding
    // row count against that array.  `src/backends/gguf.rs::emit_vocab_kv`
    // currently sizes that array from `vocab_obj` only (max id + 1 over
    // the base vocab map), NOT including `added_tokens`.  For the
    // Qwen3.6-27B + apex MoE this gives 248044 (vocab) vs the union of
    // 248070 (vocab ∪ added_tokens, ids 248044..248069 are the 26 chat
    // special tokens).  llama.cpp expects 248044, so this routine returns
    // the same.
    //
    // **Known follow-up:** extending the GGUF emit to also include
    // added_tokens (and bumping the embedding to match, 248070 rows) is
    // the strictly-correct end state — it would preserve the trained
    // embeddings for the 26 chat special tokens (currently those rows in
    // safetensors are dropped during truncation).  Tracked separately;
    // this routine returns 248044 to keep the convert + load contract
    // consistent with current emit code.
    let mut max_id: u64 = 0;
    let mut any_token: bool = false;
    for (_, id) in vocab_obj.iter() {
        if let Some(n) = id.as_u64() {
            any_token = true;
            if n > max_id {
                max_id = n;
            }
        }
    }
    if !any_token {
        return Ok(None);
    }
    let true_vocab = max_id + 1;

    if true_vocab < metadata.vocab_size {
        Ok(Some(true_vocab))
    } else {
        Ok(None)
    }
}

/// Truncate `model.embed_tokens.weight` and `lm_head.weight` to the actual
/// (de-padded) vocab row count — matches the standard `convert_hf_to_gguf.py`
/// behaviour for padded vocabularies (e.g. Qwen3.5-family) and prevents the
/// `tensor 'token_embd.weight' has wrong shape` rejection at llama.cpp load.
///
/// Mutates `metadata.vocab_size` to the de-padded value so downstream
/// metadata emission is consistent.
///
/// Returns the number of tensors truncated (0, 1, or 2 — embed and/or lm_head).
///
/// # Sovereignty
///
/// Pure-Rust truncation; reads tokenizer.json from the source `input_dir`
/// only, never consults llama.cpp output to "verify" the padding.
pub fn truncate_padded_vocab(
    tensor_map: &mut TensorMap,
    metadata: &mut ModelMetadata,
    true_vocab: u64,
) -> usize {
    use crate::ir::DType;

    // ADR-012 Bug 6 (2026-04-26): detect the actual padded row count from
    // one of the embed tensors rather than relying solely on
    // `metadata.vocab_size`.  This makes the helper idempotent and lets
    // it work on the P9b re-read path where `metadata.vocab_size` was
    // already de-padded by the first-pass call but the freshly-re-read
    // `tensor_map` still has the padded safetensors rows (e.g. 248320 for
    // Qwen3.6-27B with `metadata.vocab_size==248044`).  Falls back to
    // `metadata.vocab_size` when no embed tensor is present so callers
    // that only want a metadata-level de-pad still see the post-call
    // invariant `metadata.vocab_size == true_vocab` (preserves the
    // existing `truncate_padded_vocab_skips_when_no_embed_present` test
    // contract).
    let original_vocab = ["model.embed_tokens.weight", "lm_head.weight"]
        .iter()
        .find_map(|k| {
            tensor_map
                .tensors
                .get(*k)
                .and_then(|t| t.shape.first().map(|&r| r as u64))
        })
        .unwrap_or(metadata.vocab_size);
    if true_vocab >= original_vocab {
        return 0;
    }

    let embed_keys = ["model.embed_tokens.weight", "lm_head.weight"];
    let mut truncated = 0usize;

    for &key in &embed_keys {
        let Some(tensor) = tensor_map.tensors.get_mut(key) else {
            continue;
        };
        // Expected shape [vocab_size, hidden_size] in row-major safetensors order.
        if tensor.shape.len() != 2 {
            continue;
        }
        let rows = tensor.shape[0] as u64;
        let cols = tensor.shape[1];
        if rows != original_vocab {
            continue; // not actually padded along axis 0; skip
        }
        let elem_bytes = match tensor.dtype {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::U8 => 1,
            _ => continue, // unknown layout; safer to skip than corrupt
        };
        let new_rows = true_vocab as usize;
        let new_byte_len = new_rows * cols * elem_bytes;
        if new_byte_len > tensor.data.len() {
            continue; // shouldn't happen if shape matches, but be defensive
        }
        tensor.data.truncate(new_byte_len);
        tensor.shape[0] = new_rows;
        truncated += 1;
        tracing::info!(
            tensor = key,
            from_rows = original_vocab,
            to_rows = new_rows,
            "ADR-012 vocab-pad fix: truncated embedding to de-padded vocab"
        );
    }

    metadata.vocab_size = true_vocab;
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, ModelMetadata, TensorMap, TensorRef};

    fn metadata_with_vocab(vocab: u64) -> ModelMetadata {
        ModelMetadata {
            architecture: "test".into(),
            model_type: "test".into(),
            param_count: 0,
            hidden_size: 4,
            num_layers: 1,
            layer_types: vec![],
            num_attention_heads: 1,
            num_kv_heads: Some(1),
            vocab_size: vocab,
            dtype: "float16".into(),
            shard_count: 0,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: None,
            raw_config: serde_json::Value::Null,
            explicit_layer_types: None,
            full_attention_interval: None,
            attn_output_gate: None,
            head_dim: None,
            partial_rotary_factor: None,
            rope_parameters: None,
            linear_conv_kernel_dim: None,
            linear_key_head_dim: None,
            linear_num_key_heads: None,
            linear_value_head_dim: None,
            linear_num_value_heads: None,
            mamba_ssm_dtype: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            mtp_num_hidden_layers: None,
            mtp_use_dedicated_embeddings: None,
            output_router_logits: None,
            router_aux_loss_coef: None,
        }
    }

    fn padded_embed_tensor(name: &str, padded_rows: usize, hidden: usize) -> TensorRef {
        // Each row is filled with its row index (mod 256) so we can verify
        // truncation kept the right rows.
        let mut data = Vec::with_capacity(padded_rows * hidden * 2);
        for row in 0..padded_rows {
            let byte = (row & 0xFF) as u8;
            for _ in 0..hidden * 2 {
                data.push(byte);
            }
        }
        TensorRef {
            name: name.to_string(),
            shape: vec![padded_rows, hidden],
            dtype: DType::F16,
            data,
        }
    }

    #[test]
    fn truncate_padded_vocab_de_pads_both_embed_and_lm_head() {
        let mut metadata = metadata_with_vocab(248_320);
        let mut tensor_map = TensorMap::new();
        tensor_map.insert(padded_embed_tensor("model.embed_tokens.weight", 248_320, 4));
        tensor_map.insert(padded_embed_tensor("lm_head.weight", 248_320, 4));

        let truncated = truncate_padded_vocab(&mut tensor_map, &mut metadata, 248_044);
        assert_eq!(truncated, 2);
        assert_eq!(metadata.vocab_size, 248_044);

        let embed = tensor_map.tensors.get("model.embed_tokens.weight").unwrap();
        assert_eq!(embed.shape, vec![248_044, 4]);
        assert_eq!(embed.data.len(), 248_044 * 4 * 2);
        // Last surviving row is row 248_043 → byte (248_043 & 0xFF) = 235.
        let expected_last_row_byte = (248_043u32 & 0xFF) as u8;
        assert_eq!(embed.data[(248_044 - 1) * 4 * 2], expected_last_row_byte);

        let lm_head = tensor_map.tensors.get("lm_head.weight").unwrap();
        assert_eq!(lm_head.shape, vec![248_044, 4]);
    }

    #[test]
    fn truncate_padded_vocab_noop_when_already_aligned() {
        let mut metadata = metadata_with_vocab(248_044);
        let mut tensor_map = TensorMap::new();
        tensor_map.insert(padded_embed_tensor("model.embed_tokens.weight", 248_044, 4));

        let truncated = truncate_padded_vocab(&mut tensor_map, &mut metadata, 248_044);
        assert_eq!(truncated, 0);
        assert_eq!(metadata.vocab_size, 248_044);
        assert_eq!(
            tensor_map.tensors["model.embed_tokens.weight"].shape,
            vec![248_044, 4]
        );
    }

    #[test]
    fn truncate_padded_vocab_skips_when_no_embed_present() {
        let mut metadata = metadata_with_vocab(248_320);
        let mut tensor_map = TensorMap::new();
        let truncated = truncate_padded_vocab(&mut tensor_map, &mut metadata, 248_044);
        assert_eq!(truncated, 0);
        // Metadata IS updated even when no tensors found, since callers may
        // still use the corrected vocab_size for downstream emission.
        assert_eq!(metadata.vocab_size, 248_044);
    }

    /// ADR-012 Bug 6 regression guard (2026-04-26): the helper must work
    /// on the P9b re-read path where `metadata.vocab_size` was already
    /// de-padded by the first-pass call but the freshly-re-read
    /// `tensor_map` still has the padded safetensors rows.  Without the
    /// "detect padded rows from tensor_map" fix in `truncate_padded_vocab`,
    /// this scenario silently no-ops because the early-return branch fires
    /// when `true_vocab >= metadata.vocab_size` (both at de-padded value),
    /// leaving the freshly-loaded padded embedding intact in the final
    /// DWQ GGUF and tripping llama.cpp's
    /// `tensor 'token_embd.weight' has wrong shape; expected H, T, got H, P`
    /// load rejection.
    #[test]
    fn truncate_padded_vocab_handles_re_read_path_metadata_already_de_padded() {
        let mut metadata = metadata_with_vocab(248_044);
        let mut tensor_map = TensorMap::new();
        tensor_map.insert(padded_embed_tensor("model.embed_tokens.weight", 248_320, 4));
        tensor_map.insert(padded_embed_tensor("lm_head.weight", 248_320, 4));

        let truncated = truncate_padded_vocab(&mut tensor_map, &mut metadata, 248_044);
        assert_eq!(truncated, 2, "both embed and lm_head must be truncated");
        assert_eq!(metadata.vocab_size, 248_044);

        let embed = tensor_map.tensors.get("model.embed_tokens.weight").unwrap();
        assert_eq!(embed.shape, vec![248_044, 4]);
        let lm_head = tensor_map.tensors.get("lm_head.weight").unwrap();
        assert_eq!(lm_head.shape, vec![248_044, 4]);
    }

    #[test]
    fn detect_padded_vocab_returns_none_when_no_tokenizer() {
        let tmp = tempfile::tempdir().unwrap();
        let metadata = metadata_with_vocab(248_320);
        let result = detect_padded_vocab(tmp.path(), &metadata).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn detect_padded_vocab_finds_de_padded_count_from_tokenizer_json() {
        let tmp = tempfile::tempdir().unwrap();
        // Synthetic tokenizer.json with max id = 99 (so true_vocab = 100).
        let mut vocab = serde_json::Map::new();
        for i in 0u64..100 {
            vocab.insert(format!("tok{i}"), serde_json::json!(i));
        }
        let tokenizer = serde_json::json!({
            "model": { "vocab": vocab },
            "added_tokens": []
        });
        std::fs::write(tmp.path().join("tokenizer.json"), tokenizer.to_string()).unwrap();

        let metadata = metadata_with_vocab(128); // padded
        let result = detect_padded_vocab(tmp.path(), &metadata).unwrap();
        assert_eq!(result, Some(100));
    }

    #[test]
    fn detect_padded_vocab_ignores_added_tokens_per_emit_contract() {
        // Match the current `gguf.rs::emit_vocab_kv` contract: the GGUF's
        // tokenizer.ggml.tokens array sizes from base vocab only, so this
        // routine must too — otherwise the embedding would be truncated
        // to a count llama.cpp doesn't expect (rejection at load).
        let tmp = tempfile::tempdir().unwrap();
        let mut vocab = serde_json::Map::new();
        for i in 0u64..50 {
            vocab.insert(format!("tok{i}"), serde_json::json!(i));
        }
        let tokenizer = serde_json::json!({
            "model": { "vocab": vocab },
            "added_tokens": [
                // Even with high-id added tokens, true_vocab tracks the base
                // vocab count (50, max id 49) — NOT the union (200).
                {"id": 199, "content": "<|special|>", "special": true}
            ]
        });
        std::fs::write(tmp.path().join("tokenizer.json"), tokenizer.to_string()).unwrap();

        let metadata = metadata_with_vocab(256);
        let result = detect_padded_vocab(tmp.path(), &metadata).unwrap();
        assert_eq!(result, Some(50));
    }

    #[test]
    fn detect_padded_vocab_returns_none_when_no_padding() {
        let tmp = tempfile::tempdir().unwrap();
        let mut vocab = serde_json::Map::new();
        for i in 0u64..100 {
            vocab.insert(format!("tok{i}"), serde_json::json!(i));
        }
        let tokenizer = serde_json::json!({"model": {"vocab": vocab}});
        std::fs::write(tmp.path().join("tokenizer.json"), tokenizer.to_string()).unwrap();

        // metadata vocab matches actual count → no padding.
        let metadata = metadata_with_vocab(100);
        let result = detect_padded_vocab(tmp.path(), &metadata).unwrap();
        assert!(result.is_none());
    }
}
