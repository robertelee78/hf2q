//! BERT HF→GGUF tensor-name + metadata mapper.
//!
//! Canonical BERT name mapping plus the BERT-specific
//! metadata overlay on top of the common text-model keys. Strictly the
//! `BertModel` / `BertForMaskedLM` encoder-only path — no RoBERTa,
//! DistilBert, NomicBert, ModernBert, JinaBert, NeoBERT, EuroBert or
//! XLMRoberta special-cases (each gets its own module file when wired).
//!
//! Reference shape: `BAAI/bge-large-en-v1.5` (24-layer encoder-only,
//! hidden=1024, ffn=4096, heads=16, ctx=512, vocab=30522,
//! layer_norm_eps=1e-12, MEAN pooling).
//!
//! BERT quirks (vs. Llama-3):
//!
//! - **Encoder-only / bidirectional**: emits `bert.attention.causal =
//!   false`. Llama-3 is causal (implicit `true`, never emitted).
//! - **No rotary**: BERT uses absolute *learned* position embeddings
//!   (`position_embeddings`), so we emit a `position_embd.weight`
//!   tensor *and* skip every `*.rope.*` KV.
//! - **Three embedding tables**: `token_embd`, `position_embd`,
//!   `token_types` (Sequence-A / Sequence-B). Llama-3 has only
//!   `token_embd`.
//! - **Embedding LayerNorm**: post-embedding LayerNorm with both
//!   `weight` and `bias` — `token_embd_norm.{weight,bias}`. Llama-3
//!   has no embedding norm.
//! - **Separate Q/K/V** (not packed `qkv_proj`), every linear projection
//!   carries a **bias** (HF `BertSelfAttention` has `bias=True` on Q,
//!   K, V, output.dense, intermediate.dense, output.dense, and both
//!   LayerNorms).
//! - **No GQA**: `bert.attention.head_count_kv == bert.attention.head_count`
//!   always (BERT has no `num_key_value_heads` field; we mirror
//!   `num_attention_heads` per the user spec).
//! - **FFN is up-then-down (no gate)**: `intermediate.dense` →
//!   `ffn_up`, `output.dense` → `ffn_down`. There is no `ffn_gate` —
//!   the activation is GELU, not SwiGLU.
//! - **Two per-block post-norms**: `attn_output_norm` (after attention
//!   add+residual) and `layer_output_norm` (after FFN add+residual).
//!   Llama-3 has `attn_norm` + `ffn_norm` as *pre-norms* before each
//!   sublayer.
//! - **Optional pooler**: `pooler.dense.{weight,bias}` → `cls.{weight,bias}`.
//!   The canonical conversion drops the pooler (line 82 of `bert.py`:
//!   *"we are only using BERT for embeddings so we don't need the
//!   pooling layer"*); we still map it so models that *do* ship a
//!   pooler — e.g. classifier checkpoints — round-trip.
//! - **Pooling type KV**: emits `bert.pooling_type` as a `u32`
//!   (`PoolingType` per `gguf-py/gguf/constants.py:4102-4107` — NONE=0,
//!   MEAN=1, CLS=2, LAST=3, RANK=4). Default `1` (MEAN), matching the
//!   BAAI/bge reference shape. Llama-3 emits no `pooling_type`.
//! - **Layer-norm epsilon key is `layer_norm_epsilon`** (not
//!   `rms_norm_eps`); BERT uses standard LayerNorm, not RMSNorm.
//!
//! Per ADR-033 §P0 "Per-arch convert-side mapping": this is the
//! convert-side tensor-name + KV mapper for `LLM_ARCH_BERT`.
//!
//! Per [[feedback-no-backwards-compat-2026-05-18]]: every HF name we
//! recognize maps to exactly one GGUF name; every other name returns
//! `None`. Per [[feedback-no-loop-suppression-2026-05-17]]: callers
//! MUST NOT silently skip a `None` — propagate as a typed error.

use crate::backends::gguf::types::MetaValue;

/// Strip a leading `bert.` prefix if present.
///
/// Some HF BERT checkpoints (notably the original `bert-base-*` family
/// when loaded under `BertModel` rather than `BertForMaskedLM`) ship
/// with a `bert.` prefix on every parameter; others (e.g. sentence-
/// transformers checkpoints, BAAI/bge-large-en-v1.5) ship the bare
/// `embeddings.*` / `encoder.*` / `pooler.*` layout. Mirrors
/// `conversion/bert.py::BertModel.filter_tensors` lines 72-73.
fn strip_bert_prefix(name: &str) -> &str {
    name.strip_prefix("bert.").unwrap_or(name)
}

/// Translate one HuggingFace tensor name (as seen in `model.safetensors`)
/// to its canonical GGUF tensor name. Returns `None` if `hf_name` is not
/// one of the BERT weight kinds.
///
/// BERT weight kinds:
///
/// | HF name                                                            | GGUF name                              |
/// |--------------------------------------------------------------------|----------------------------------------|
/// | `embeddings.word_embeddings.weight`                                | `token_embd.weight`                    |
/// | `embeddings.position_embeddings.weight`                            | `position_embd.weight`                 |
/// | `embeddings.token_type_embeddings.weight`                          | `token_types.weight`                   |
/// | `embeddings.LayerNorm.weight`                                      | `token_embd_norm.weight`               |
/// | `embeddings.LayerNorm.bias`                                        | `token_embd_norm.bias`                 |
/// | `encoder.layer.<N>.attention.self.query.{weight,bias}`             | `blk.<N>.attn_q.{weight,bias}`         |
/// | `encoder.layer.<N>.attention.self.key.{weight,bias}`               | `blk.<N>.attn_k.{weight,bias}`         |
/// | `encoder.layer.<N>.attention.self.value.{weight,bias}`             | `blk.<N>.attn_v.{weight,bias}`         |
/// | `encoder.layer.<N>.attention.output.dense.{weight,bias}`           | `blk.<N>.attn_output.{weight,bias}`    |
/// | `encoder.layer.<N>.attention.output.LayerNorm.{weight,bias}`       | `blk.<N>.attn_output_norm.{weight,bias}` |
/// | `encoder.layer.<N>.intermediate.dense.{weight,bias}`               | `blk.<N>.ffn_up.{weight,bias}`         |
/// | `encoder.layer.<N>.output.dense.{weight,bias}`                     | `blk.<N>.ffn_down.{weight,bias}`       |
/// | `encoder.layer.<N>.output.LayerNorm.{weight,bias}`                 | `blk.<N>.layer_output_norm.{weight,bias}` |
/// | `pooler.dense.{weight,bias}` (optional)                            | `cls.{weight,bias}`                    |
///
/// HF input names are accepted with or without a leading `bert.`
/// prefix (some checkpoints carry it, sentence-transformers strip it).
pub fn map_tensor_name(hf_name: &str) -> Option<String> {
    let name = strip_bert_prefix(hf_name);

    // ---- Embedding globals -----------------------------------------------
    match name {
        "embeddings.word_embeddings.weight" => {
            return Some("token_embd.weight".to_string());
        }
        "embeddings.position_embeddings.weight" => {
            return Some("position_embd.weight".to_string());
        }
        "embeddings.token_type_embeddings.weight" => {
            return Some("token_types.weight".to_string());
        }
        "embeddings.LayerNorm.weight" => {
            return Some("token_embd_norm.weight".to_string());
        }
        "embeddings.LayerNorm.bias" => {
            return Some("token_embd_norm.bias".to_string());
        }
        // Optional pooler — present on classifier / BertModel-style
        // checkpoints, dropped by upstream `bert.py:filter_tensors` for
        // pure embedding models but mapped here for completeness.
        "pooler.dense.weight" => return Some("cls.weight".to_string()),
        "pooler.dense.bias" => return Some("cls.bias".to_string()),
        _ => {}
    }

    // ---- Per-block: `encoder.layer.<N>.<rest>` ---------------------------
    let stripped = name.strip_prefix("encoder.layer.")?;
    let dot = stripped.find('.')?;
    let (layer_str, rest_with_dot) = stripped.split_at(dot);
    // Parse layer index (must be a bare non-negative integer; reject
    // leading zeros / signs to keep the mapper strict — matches the
    // Llama-3 sibling's tolerance).
    let layer: usize = layer_str.parse().ok()?;
    if layer.to_string() != layer_str {
        return None;
    }
    let rest = &rest_with_dot[1..]; // skip the dot

    // Each arm is `(hf_local, gguf_local)` — split on the trailing
    // `.weight` / `.bias` so we can route both halves of a linear
    // (weight + bias) through one table entry. Per-block table is the
    // 8 BERT sublayer pieces (Q/K/V/O/O_norm/up/down/layer_norm).
    let (head, suffix) = if let Some(stem) = rest.strip_suffix(".weight") {
        (stem, ".weight")
    } else if let Some(stem) = rest.strip_suffix(".bias") {
        (stem, ".bias")
    } else {
        return None;
    };

    let local = match head {
        "attention.self.query" => "attn_q",
        "attention.self.key" => "attn_k",
        "attention.self.value" => "attn_v",
        "attention.output.dense" => "attn_output",
        "attention.output.LayerNorm" => "attn_output_norm",
        "intermediate.dense" => "ffn_up",
        "output.dense" => "ffn_down",
        "output.LayerNorm" => "layer_output_norm",
        _ => return None,
    };

    Some(format!("blk.{layer}.{local}{suffix}"))
}

/// Translate the user-facing pooling-mode string into the GGUF
/// `PoolingType` enum's `u32` representation.
///
/// GGUF PoolingType values:
/// `NONE=0`, `MEAN=1`, `CLS=2`, `LAST=3`, `RANK=4`. Default (None
/// supplied) is `MEAN=1` — the BAAI/bge reference shape.
///
/// Returns `None` for an unrecognized mode string (caller decides how
/// to surface the error).
fn pooling_type_u32(mode: Option<&str>) -> Option<u32> {
    match mode {
        None => Some(1), // default MEAN
        Some("mean") | Some("MEAN") => Some(1),
        Some("cls") | Some("CLS") => Some(2),
        Some("last") | Some("lasttoken") | Some("LAST") => Some(3),
        Some("none") | Some("NONE") => Some(0),
        Some("rank") | Some("RANK") => Some(4),
        _ => None,
    }
}

/// Build the GGUF metadata KV pairs for a BERT model from its HF
/// `config.json`. Port of `conversion/bert.py::BertModel::set_gguf_parameters`
/// (the two overrides — `add_causal_attention(False)` and
/// `_try_set_pooling_type()`) layered over the
/// `TextModel::set_gguf_parameters` base (`base.py:1111-1221`).
///
/// Required HF keys (mandatory; missing key → caller-side panic from
/// the `[]` indexing):
///   - `hidden_size`
///   - `num_hidden_layers`
///   - `intermediate_size`
///   - `num_attention_heads`
///   - `max_position_embeddings`
///   - `layer_norm_eps`
///
/// Optional HF keys (defaulted):
///   - `_name_or_path` — defaults to `"model"`.
///   - `pooling` — defaults to `"mean"` (MEAN, the BAAI/bge default
///     and the most common sentence-transformers default). Recognized
///     values: `mean` | `cls` | `last` | `none` | `rank`. Unknown
///     values panic at conversion time — per
///     [[feedback-no-loop-suppression-2026-05-17]] we surface bad
///     metadata rather than silently downgrade to a default.
///
/// `file_type` is the chosen `GgufFtype` as a `u32` (matches
/// `gguf_writer.add_file_type(self.ftype)` at base.py:1220).
pub fn build_metadata(
    config: &serde_json::Value,
    file_type: u32,
    model_card: Option<&crate::convert::model_card::ModelCard>,
    sampling: Option<&crate::convert::model_card::SamplingConfig>,
    model_dir_basename: Option<&str>,
    pooling_override: Option<u32>,
) -> Vec<(String, MetaValue)> {
    use crate::convert::model_card::{
        emit_general_postlude, emit_general_prelude, get_model_id_components,
    };
    // BERT bge config.json carries `_name_or_path = "/root/.cache/..."`
    // which is a noisy filesystem path. Prefer the model directory's
    // basename (e.g. "BAAI-bge-large-en-v1.5") so canonical's
    // `get_model_id_components` heuristic produces the same
    // basename/finetune/size_label/name as canonical's GGUF dump.
    let raw_name = model_dir_basename
        .map(|s| s.to_string())
        .or_else(|| {
            config
                .get("_name_or_path")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "model".to_string());
    let id_components = get_model_id_components(&raw_name);
    let display_name = id_components
        .name
        .clone()
        .unwrap_or_else(|| raw_name.clone());

    let hidden_size = config["hidden_size"]
        .as_u64()
        .expect("config.json missing required key `hidden_size`") as u32;
    let n_layers = config["num_hidden_layers"]
        .as_u64()
        .expect("config.json missing required key `num_hidden_layers`") as u32;
    let ffn_len = config["intermediate_size"]
        .as_u64()
        .expect("config.json missing required key `intermediate_size`") as u32;
    let n_head = config["num_attention_heads"]
        .as_u64()
        .expect("config.json missing required key `num_attention_heads`") as u32;
    let ctx_len = config["max_position_embeddings"]
        .as_u64()
        .expect("config.json missing required key `max_position_embeddings`")
        as u32;
    let ln_eps = config["layer_norm_eps"]
        .as_f64()
        .expect("config.json missing required key `layer_norm_eps`") as f32;

    // Pooling type resolution order:
    //   1. Explicit `pooling_override` from cli_driver (canonical's
    //      `_try_set_pooling_type` reads modules.json + 1_Pooling/
    //      config.json — that lives in cli_driver where the model_dir
    //      Path is available).
    //   2. `config["pooling"]` (string: mean | cls | last | none | rank).
    //   3. Default MEAN.
    let pooling_u32 = pooling_override.unwrap_or_else(|| {
        let pooling_mode = config.get("pooling").and_then(|v| v.as_str());
        pooling_type_u32(pooling_mode).unwrap_or_else(|| {
            panic!(
                "config.json key `pooling` has unrecognized value {pooling_mode:?}; \
                 expected one of mean | cls | last | none | rank"
            )
        })
    });

    // Canonical bert.py:25-29: cls_out_labels from id2label, but
    // dropped if exactly 2 labels with index 0 = "LABEL_0" (dummy
    // labels AutoConfig adds). Keep otherwise.
    let cls_out_labels: Option<Vec<String>> = config
        .get("id2label")
        .and_then(|v| v.as_object())
        .and_then(|m| {
            // Sort by integer key
            let mut entries: Vec<(i64, String)> = m
                .iter()
                .filter_map(|(k, v)| Some((k.parse::<i64>().ok()?, v.as_str()?.to_string())))
                .collect();
            if entries.is_empty() {
                return None;
            }
            if entries.len() == 2 && entries.iter().any(|(k, v)| *k == 0 && v == "LABEL_0") {
                // Skip dummy LABEL_0 + LABEL_1 pair
                return None;
            }
            entries.sort_by_key(|e| e.0);
            Some(entries.into_iter().map(|(_, v)| v).collect())
        });

    // Canonical `general.*` prelude — architecture, type, sampling.*,
    // name, version/organization/finetune/basename, size_label,
    // license, base_model.*, tags, languages.
    let mut kv: Vec<(String, MetaValue)> = emit_general_prelude(
        "bert",
        display_name,
        &id_components,
        None,
        model_card,
        sampling,
    );
    // Canonical BERT bge arch-KV emit order (verified against
    // /opt/hf2q/cache/byte_cmp/BAAI-bge-large-en-v1.5_canonical_q4_k_m.gguf
    // dump positions 14-22):
    //   block_count, context_length, embedding_length,
    //   feed_forward_length, attention.head_count,
    //   attention.layer_norm_epsilon, attention.causal, pooling_type,
    //   classifier.output_labels (when present)
    //
    // Note: canonical does NOT emit `bert.attention.head_count_kv`
    // (BERT has no GQA; the C runtime treats absent as = n_head).
    kv.push(("bert.block_count".into(), MetaValue::U32(n_layers)));
    kv.push(("bert.context_length".into(), MetaValue::U32(ctx_len)));
    kv.push(("bert.embedding_length".into(), MetaValue::U32(hidden_size)));
    kv.push(("bert.feed_forward_length".into(), MetaValue::U32(ffn_len)));
    kv.push(("bert.attention.head_count".into(), MetaValue::U32(n_head)));
    kv.push((
        "bert.attention.layer_norm_epsilon".into(),
        MetaValue::F32(ln_eps),
    ));
    kv.push(("bert.attention.causal".into(), MetaValue::Bool(false)));
    kv.push(("bert.pooling_type".into(), MetaValue::U32(pooling_u32)));
    if let Some(labels) = cls_out_labels {
        kv.push((
            "bert.classifier.output_labels".into(),
            MetaValue::ArrayString(labels),
        ));
    }
    kv.extend(emit_general_postlude(file_type));
    kv
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Acceptance test 1 — round-trip every BERT weight kind. Covers
    /// the five embedding globals + 8 per-block sublayers (× 2 for
    /// weight/bias on the 7 of those 8 that have a bias) + the
    /// optional pooler. Sample layers at L=0, L=11, L=23 to cover edge
    /// / mid / depth across the bge-large 24-layer shape.
    #[test]
    fn bert_tensor_name_round_trip() {
        let cases: &[(&str, &str)] = &[
            // ----- Embedding globals -----
            ("embeddings.word_embeddings.weight", "token_embd.weight"),
            (
                "embeddings.position_embeddings.weight",
                "position_embd.weight",
            ),
            (
                "embeddings.token_type_embeddings.weight",
                "token_types.weight",
            ),
            ("embeddings.LayerNorm.weight", "token_embd_norm.weight"),
            ("embeddings.LayerNorm.bias", "token_embd_norm.bias"),
            // ----- Per-block: edge (L=0) -----
            (
                "encoder.layer.0.attention.self.query.weight",
                "blk.0.attn_q.weight",
            ),
            (
                "encoder.layer.0.attention.self.query.bias",
                "blk.0.attn_q.bias",
            ),
            (
                "encoder.layer.0.attention.self.key.weight",
                "blk.0.attn_k.weight",
            ),
            (
                "encoder.layer.0.attention.self.key.bias",
                "blk.0.attn_k.bias",
            ),
            (
                "encoder.layer.0.attention.self.value.weight",
                "blk.0.attn_v.weight",
            ),
            (
                "encoder.layer.0.attention.self.value.bias",
                "blk.0.attn_v.bias",
            ),
            (
                "encoder.layer.0.attention.output.dense.weight",
                "blk.0.attn_output.weight",
            ),
            (
                "encoder.layer.0.attention.output.dense.bias",
                "blk.0.attn_output.bias",
            ),
            (
                "encoder.layer.0.attention.output.LayerNorm.weight",
                "blk.0.attn_output_norm.weight",
            ),
            (
                "encoder.layer.0.attention.output.LayerNorm.bias",
                "blk.0.attn_output_norm.bias",
            ),
            (
                "encoder.layer.0.intermediate.dense.weight",
                "blk.0.ffn_up.weight",
            ),
            (
                "encoder.layer.0.intermediate.dense.bias",
                "blk.0.ffn_up.bias",
            ),
            (
                "encoder.layer.0.output.dense.weight",
                "blk.0.ffn_down.weight",
            ),
            ("encoder.layer.0.output.dense.bias", "blk.0.ffn_down.bias"),
            (
                "encoder.layer.0.output.LayerNorm.weight",
                "blk.0.layer_output_norm.weight",
            ),
            (
                "encoder.layer.0.output.LayerNorm.bias",
                "blk.0.layer_output_norm.bias",
            ),
            // ----- Per-block: mid (L=11) -----
            (
                "encoder.layer.11.attention.self.query.weight",
                "blk.11.attn_q.weight",
            ),
            (
                "encoder.layer.11.intermediate.dense.bias",
                "blk.11.ffn_up.bias",
            ),
            // ----- Per-block: depth (L=23, bge-large terminal layer) -----
            (
                "encoder.layer.23.attention.output.LayerNorm.bias",
                "blk.23.attn_output_norm.bias",
            ),
            (
                "encoder.layer.23.output.LayerNorm.weight",
                "blk.23.layer_output_norm.weight",
            ),
            // ----- Optional pooler -----
            ("pooler.dense.weight", "cls.weight"),
            ("pooler.dense.bias", "cls.bias"),
        ];

        for &(hf, expected_gguf) in cases {
            let got = map_tensor_name(hf);
            assert_eq!(
                got.as_deref(),
                Some(expected_gguf),
                "map_tensor_name({hf:?}) = {got:?}, want Some({expected_gguf:?})"
            );
        }
    }

    /// Acceptance test 2 — verify the `bert.` prefix is stripped
    /// transparently when present. Same round-trip as test 1 but on
    /// the prefix-carrying form found in original `bert-base-uncased`
    /// safetensors.
    #[test]
    fn bert_tensor_name_strips_bert_prefix() {
        let cases: &[(&str, &str)] = &[
            (
                "bert.embeddings.word_embeddings.weight",
                "token_embd.weight",
            ),
            ("bert.embeddings.LayerNorm.bias", "token_embd_norm.bias"),
            (
                "bert.encoder.layer.5.attention.self.value.bias",
                "blk.5.attn_v.bias",
            ),
            (
                "bert.encoder.layer.5.output.LayerNorm.weight",
                "blk.5.layer_output_norm.weight",
            ),
            ("bert.pooler.dense.weight", "cls.weight"),
        ];
        for &(hf, expected) in cases {
            assert_eq!(
                map_tensor_name(hf).as_deref(),
                Some(expected),
                "stripped-prefix mapping for {hf:?} failed"
            );
        }
    }

    /// Sibling — unknown names must surface as `None`. Per
    /// [[feedback-no-loop-suppression-2026-05-17]]: the caller is
    /// expected to error on this, never silently skip.
    #[test]
    fn bert_tensor_name_rejects_unknown_kinds() {
        // Unknown embedding global.
        assert_eq!(map_tensor_name("embeddings.unknown.weight"), None);
        // Wrong prefix (transformer-style — not BERT).
        assert_eq!(map_tensor_name("transformer.h.0.attn.c_attn.weight"), None);
        // Llama-3-style key shouldn't match BERT mapper.
        assert_eq!(
            map_tensor_name("model.layers.0.self_attn.q_proj.weight"),
            None
        );
        // Malformed layer index (leading zero).
        assert_eq!(
            map_tensor_name("encoder.layer.01.attention.self.query.weight"),
            None
        );
        // Empty layer index.
        assert_eq!(
            map_tensor_name("encoder.layer..attention.self.query.weight"),
            None
        );
        // No layer index at all.
        assert_eq!(
            map_tensor_name("encoder.layer.attention.self.query.weight"),
            None
        );
        // Negative layer index.
        assert_eq!(
            map_tensor_name("encoder.layer.-1.attention.self.query.weight"),
            None
        );
        // Unknown per-block suffix.
        assert_eq!(map_tensor_name("encoder.layer.0.unknown.weight"), None);
        // BERT has no rotary / no ffn_gate — these must NOT map.
        assert_eq!(
            map_tensor_name("encoder.layer.0.attention.self.rotary_emb.inv_freq"),
            None
        );
        // Suffix that's neither .weight nor .bias.
        assert_eq!(
            map_tensor_name("encoder.layer.0.attention.self.query.gamma"),
            None
        );
    }

    /// Acceptance test 3 — feed a minimal hand-written config.json
    /// (matching BAAI/bge-large-en-v1.5 shape: 24 layers × hidden 1024
    /// × ffn 4096 × 16 heads × ctx 512, layer_norm_eps=1e-12) and
    /// verify all 12 KV pairs come back with the right types + values.
    #[test]
    fn bert_metadata_built_from_config() {
        let cfg = json!({
            "_name_or_path": "BAAI/bge-large-en-v1.5",
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "max_position_embeddings": 512,
            "layer_norm_eps": 1.0e-12,
            "pooling": "cls",
        });

        let kv = build_metadata(&cfg, 1 /* MostlyF16 */, None, None, None, None);

        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();

        assert_eq!(
            by_key["general.architecture"],
            MetaValue::String("bert".into())
        );
        assert!(
            matches!(by_key.get("general.name"), Some(MetaValue::String(_))),
            "general.name must be present"
        );
        assert_eq!(by_key["bert.context_length"], MetaValue::U32(512));
        assert_eq!(by_key["bert.embedding_length"], MetaValue::U32(1024));
        assert_eq!(by_key["bert.block_count"], MetaValue::U32(24));
        assert_eq!(by_key["bert.feed_forward_length"], MetaValue::U32(4096));
        assert_eq!(by_key["bert.attention.head_count"], MetaValue::U32(16));
        assert!(
            by_key.get("bert.attention.head_count_kv").is_none(),
            "canonical does NOT emit head_count_kv for BERT"
        );
        assert_eq!(
            by_key["bert.attention.layer_norm_epsilon"],
            MetaValue::F32(1.0e-12)
        );
        assert_eq!(
            by_key["bert.attention.causal"],
            MetaValue::Bool(false),
            "BERT is encoder-only / bidirectional"
        );
        assert_eq!(
            by_key["bert.pooling_type"],
            MetaValue::U32(2),
            "pooling=cls → PoolingType::CLS = 2"
        );
        assert_eq!(by_key["general.file_type"], MetaValue::U32(1));
        assert_eq!(by_key["general.quantization_version"], MetaValue::U32(2));
    }

    /// Sibling — verify the optional-key defaults: missing
    /// `_name_or_path` → "model", missing `pooling` → MEAN (=1).
    #[test]
    fn bert_metadata_optional_key_defaults() {
        let cfg = json!({
            // _name_or_path omitted → defaults to "model"
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "max_position_embeddings": 512,
            "layer_norm_eps": 1.0e-12,
            // pooling omitted → defaults to MEAN
        });
        let kv = build_metadata(&cfg, 0, None, None, None, None);
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        // get_model_id_components("model") → title-cased "Model"
        assert_eq!(
            by_key["general.name"],
            MetaValue::String("Model".into()),
            "name defaults to title-cased 'Model' when no source available"
        );
        assert_eq!(
            by_key["bert.pooling_type"],
            MetaValue::U32(1),
            "pooling defaults to MEAN (=1)"
        );
        assert!(
            by_key.get("bert.attention.head_count_kv").is_none(),
            "canonical does NOT emit head_count_kv for BERT"
        );
        assert_eq!(
            by_key["bert.attention.causal"],
            MetaValue::Bool(false),
            "causal=false even without explicit config opt-in"
        );
    }
}
