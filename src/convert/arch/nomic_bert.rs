//! NomicBert (nomic-embed-text v1 / v1.5) HF→GGUF tensor-name + metadata
//! mapper.
//!
//! Port of `/opt/llama.cpp/conversion/bert.py::NomicBertModel`'s name
//! mapping (transitively via `BertModel::modify_tensors` →
//! `gguf-py/gguf/tensor_mapping.py`) and `set_gguf_parameters`. Strictly
//! the dense `nomic-bert` path — the MoE variant (`nomic-bert-moe`) is
//! out of v1 scope and routes through a separate mapper.
//!
//! Per ADR-033 §P0 "Per-arch convert-side mapping": this is the
//! convert-side tensor-name + KV mapper for `LLM_ARCH_NOMIC_BERT`.
//!
//! NomicBert highlights (vs plain BERT, verified against
//! `/opt/llama.cpp/gguf-py/gguf/tensor_mapping.py` lines 233/326/345/495/555/624/710
//! and `conversion/bert.py::NomicBertModel.__init__` lines 340-354):
//!
//!   - **Fused QKV.** The three Q/K/V projections live in one packed
//!     tensor `encoder.layers.<N>.attn.Wqkv` (no `query`/`key`/`value`
//!     split). Maps to GGUF `blk.<N>.attn_qkv.weight`.
//!   - **Rotary position embeddings, not absolute.** NomicBert uses RoPE
//!     (`rotary_emb_fraction == 1.0`, `rotary_emb_interleaved is False`),
//!     not learned position embeddings. The HF checkpoint typically
//!     omits `embeddings.position_embeddings` entirely; if it is present
//!     (legacy / mistakenly-trained), it is unused — we still return
//!     `None` so the caller can decide whether to skip-with-warning or
//!     error per [[feedback-no-loop-suppression-2026-05-17]].
//!   - **SwiGLU FFN, not up-then-down.** Non-MoE NomicBert asserts
//!     `activation_function == "swiglu"`. The FFN is two parallel
//!     up-projections `mlp.fc11` (gate side, applied through SiLU) and
//!     `mlp.fc12` (up side), then a single down `mlp.fc2`. Maps to
//!     GGUF `blk.<N>.{ffn_gate, ffn_up, ffn_down}.weight`. Note: this
//!     is the OPPOSITE of plain GELU BERT, which has only `mlp.fc1`
//!     (up) + `mlp.fc2` (down). The MoE variant flips back to plain
//!     `fc1`+`fc2` GELU per-expert — out of scope here.
//!   - **No biases on linear projections.** NomicBertModel asserts
//!     `qkv_proj_bias == is_moe`, `mlp_fc1_bias == is_moe`,
//!     `mlp_fc2_bias == is_moe`. With `is_moe == false` (the v1 path),
//!     attn.Wqkv, attn.out_proj, mlp.fc11/fc12/fc2 are all
//!     weights-only. LayerNorms (`norm1`, `norm2`, `emb_ln`) retain
//!     their `.bias` half (standard LayerNorm has γ + β).
//!   - **Mean pooling.** Nomic embeddings are mean-pooled over tokens;
//!     emitted as `nomic-bert.pooling_type = u32 1` (MEAN). Matches
//!     llama.cpp's `LLAMA_POOLING_TYPE_MEAN`.
//!
//! Per [[feedback-no-backwards-compat-2026-05-18]]: no fallback / no
//! aliasing — every HF name we recognize maps to exactly one GGUF name,
//! and every other name returns `None` (the caller decides how to
//! surface the error). Per [[feedback-no-loop-suppression-2026-05-17]]:
//! callers MUST NOT silently skip a `None` — propagate as a typed
//! error.

use crate::backends::gguf::types::MetaValue;

/// Translate one HuggingFace tensor name (as seen in `model.safetensors`)
/// to its canonical GGUF tensor name. Returns `None` if `hf_name` is not
/// one of the NomicBert weight kinds.
///
/// Accepts an optional `bert.` HF prefix (some checkpoints — typically
/// those exported from a wrapping `BertForMaskedLM`-style head — carry
/// it; nomic-embed-text-v1/v1.5 typically does not). Mirrors the strip
/// in `BertModel.filter_tensors` (`/opt/llama.cpp/conversion/bert.py:72-73`).
///
/// NomicBert weight kinds (one global pair + seven per-block):
///
/// | HF name                                          | GGUF name                            |
/// |--------------------------------------------------|--------------------------------------|
/// | `embeddings.word_embeddings.weight`              | `token_embd.weight`                  |
/// | `emb_ln.weight`                                  | `token_embd_norm.weight`             |
/// | `emb_ln.bias`                                    | `token_embd_norm.bias`               |
/// | `encoder.layers.<N>.attn.Wqkv.weight`            | `blk.<N>.attn_qkv.weight`            |
/// | `encoder.layers.<N>.attn.out_proj.weight`        | `blk.<N>.attn_output.weight`         |
/// | `encoder.layers.<N>.norm1.weight`                | `blk.<N>.attn_output_norm.weight`    |
/// | `encoder.layers.<N>.norm1.bias`                  | `blk.<N>.attn_output_norm.bias`      |
/// | `encoder.layers.<N>.mlp.fc11.weight`             | `blk.<N>.ffn_gate.weight`            |
/// | `encoder.layers.<N>.mlp.fc12.weight`             | `blk.<N>.ffn_up.weight`              |
/// | `encoder.layers.<N>.mlp.fc2.weight`              | `blk.<N>.ffn_down.weight`            |
/// | `encoder.layers.<N>.norm2.weight`                | `blk.<N>.layer_output_norm.weight`   |
/// | `encoder.layers.<N>.norm2.bias`                  | `blk.<N>.layer_output_norm.bias`     |
///
/// Returns `None` for:
///   - `embeddings.position_embeddings.weight` (unused; RoPE replaces
///     absolute positions). Caller should warn + drop.
///   - `embeddings.token_type_embeddings.weight` (unused; NomicBert
///     drops segment IDs vs plain BERT). Caller should warn + drop.
///   - Any linear-projection bias (`.attn.Wqkv.bias`, `.attn.out_proj.bias`,
///     `.mlp.fc11.bias`, `.mlp.fc12.bias`, `.mlp.fc2.bias`) — NomicBertModel
///     asserts these don't exist on the non-MoE path.
///   - Any other name.
pub fn map_tensor_name(hf_name: &str) -> Option<String> {
    // Strip the optional `bert.` prefix — see BertModel.filter_tensors.
    let name = hf_name.strip_prefix("bert.").unwrap_or(hf_name);

    // ---- Globals (embedding table + embedding LayerNorm) -----------------
    match name {
        "embeddings.word_embeddings.weight" => {
            return Some("token_embd.weight".to_string())
        }
        "emb_ln.weight" => return Some("token_embd_norm.weight".to_string()),
        "emb_ln.bias" => return Some("token_embd_norm.bias".to_string()),
        _ => {}
    }

    // ---- Per-block: `encoder.layers.<N>.<rest>` --------------------------
    let stripped = name.strip_prefix("encoder.layers.")?;
    let dot = stripped.find('.')?;
    let (layer_str, rest_with_dot) = stripped.split_at(dot);
    // Parse layer index (must be a bare non-negative integer; reject
    // leading zeros / signs to keep the mapper strict — mirrors the
    // llama3 sibling).
    let layer: usize = layer_str.parse().ok()?;
    if layer.to_string() != layer_str {
        return None;
    }
    let rest = &rest_with_dot[1..]; // skip the dot

    let suffix = match rest {
        // Fused QKV — weight only (no bias on non-MoE NomicBert).
        "attn.Wqkv.weight" => "attn_qkv.weight",
        // Attention output projection — weight only.
        "attn.out_proj.weight" => "attn_output.weight",
        // Post-attention LayerNorm — weight + bias.
        "norm1.weight" => "attn_output_norm.weight",
        "norm1.bias" => "attn_output_norm.bias",
        // SwiGLU FFN — fc11 is the gate, fc12 is the up, fc2 is the down.
        // All weight-only.
        "mlp.fc11.weight" => "ffn_gate.weight",
        "mlp.fc12.weight" => "ffn_up.weight",
        "mlp.fc2.weight" => "ffn_down.weight",
        // End-of-layer LayerNorm — weight + bias.
        "norm2.weight" => "layer_output_norm.weight",
        "norm2.bias" => "layer_output_norm.bias",
        _ => return None,
    };

    Some(format!("blk.{layer}.{suffix}"))
}

/// Build the GGUF metadata KV pairs for a NomicBert model from its HF
/// `config.json`. Port of `conversion/bert.py::NomicBertModel::set_gguf_parameters`
/// + the `BertModel.set_gguf_parameters` + `TextModel.set_gguf_parameters`
/// base it transitively `super()`s into.
///
/// NomicBert HF configs are typically GPT-Neo / Mosaic-style and use
/// `n_embd` / `n_layer` / `n_head` / `n_inner` (not the
/// `hidden_size` / `num_hidden_layers` family). We accept BOTH name
/// conventions and prefer the GPT-style key when present, to match
/// what `find_hparam(["n_embd", "hidden_size"])` does in the Python
/// base.
///
/// Required HF keys (at least one of each pair must be present;
/// missing-both → caller-side panic from the `.expect`):
///   - `n_embd`           OR `hidden_size`
///   - `n_layer`          OR `num_hidden_layers`
///   - `n_head`           OR `num_attention_heads`
///   - `max_position_embeddings` (mandatory)
///
/// Optional HF keys (defaulted):
///   - `n_inner` — defaults to `4 * embedding_length` per the
///     GPT-Neo `Config.__init__` default (NomicBert v1.5 supplies it
///     explicitly = 2048 for the 137M model).
///   - `layer_norm_epsilon` — defaults to `1.0e-12` per
///     `transformers.BertConfig` (NomicBert v1/v1.5 supply it).
///   - `rotary_emb_base` — defaults to `10000.0` per the
///     `NomicBertConfig` factory default.
///   - `_name_or_path` — defaults to `"model"`.
///
/// `file_type` is the chosen ggml file-type as a `u32` (matches
/// `gguf_writer.add_file_type(self.ftype)` at base.py).
pub fn build_metadata(config: &serde_json::Value, file_type: u32) -> Vec<(String, MetaValue)> {
    let name = config
        .get("_name_or_path")
        .and_then(|v| v.as_str())
        .unwrap_or("model")
        .to_string();

    // Pull a required `u64` from either of two HF keys.
    let pick_u64 = |k_gpt: &str, k_hf: &str| -> u32 {
        let v = config
            .get(k_gpt)
            .and_then(|v| v.as_u64())
            .or_else(|| config.get(k_hf).and_then(|v| v.as_u64()))
            .unwrap_or_else(|| {
                panic!(
                    "config.json missing required key (`{k_gpt}` or `{k_hf}`)"
                )
            });
        v as u32
    };

    let hidden_size = pick_u64("n_embd", "hidden_size");
    let n_layers = pick_u64("n_layer", "num_hidden_layers");
    let n_head = pick_u64("n_head", "num_attention_heads");

    let ctx_len = config["max_position_embeddings"]
        .as_u64()
        .expect("config.json missing required key `max_position_embeddings`") as u32;

    // Optional with defaults.
    let ffn_len = config
        .get("n_inner")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32)
        .unwrap_or(4 * hidden_size);

    let ln_eps = config
        .get("layer_norm_epsilon")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0e-12) as f32;

    let rope_theta = config
        .get("rotary_emb_base")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0) as f32;

    // NomicBert has no GQA — head_count_kv == head_count.
    let n_head_kv = n_head;

    vec![
        (
            "general.architecture".into(),
            MetaValue::String("nomic-bert".into()),
        ),
        ("general.name".into(), MetaValue::String(name)),
        ("nomic-bert.context_length".into(), MetaValue::U32(ctx_len)),
        (
            "nomic-bert.embedding_length".into(),
            MetaValue::U32(hidden_size),
        ),
        ("nomic-bert.block_count".into(), MetaValue::U32(n_layers)),
        (
            "nomic-bert.feed_forward_length".into(),
            MetaValue::U32(ffn_len),
        ),
        (
            "nomic-bert.attention.head_count".into(),
            MetaValue::U32(n_head),
        ),
        (
            "nomic-bert.attention.head_count_kv".into(),
            MetaValue::U32(n_head_kv),
        ),
        (
            "nomic-bert.attention.layer_norm_epsilon".into(),
            MetaValue::F32(ln_eps),
        ),
        (
            "nomic-bert.rope.freq_base".into(),
            MetaValue::F32(rope_theta),
        ),
        // Pooling type 1 == LLAMA_POOLING_TYPE_MEAN. Nomic Embed v1/v1.5
        // are mean-pooled embedding models.
        ("nomic-bert.pooling_type".into(), MetaValue::U32(1)),
        ("general.file_type".into(), MetaValue::U32(file_type)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Acceptance test 1 — round-trip the canonical table for every HF
    /// name kind. Asserts `map_tensor_name(hf) → Some(gguf)` with the
    /// exact pair from the rustdoc table above.
    #[test]
    fn nomic_bert_tensor_name_round_trip() {
        let cases: &[(&str, &str)] = &[
            // Globals
            ("embeddings.word_embeddings.weight", "token_embd.weight"),
            ("emb_ln.weight", "token_embd_norm.weight"),
            ("emb_ln.bias", "token_embd_norm.bias"),
            // Per-block — sample at L=0, L=5, L=11 (nomic-embed v1.5
            // has 12 layers so 0/5/11 covers edge/mid/depth).
            (
                "encoder.layers.0.attn.Wqkv.weight",
                "blk.0.attn_qkv.weight",
            ),
            (
                "encoder.layers.5.attn.Wqkv.weight",
                "blk.5.attn_qkv.weight",
            ),
            (
                "encoder.layers.11.attn.Wqkv.weight",
                "blk.11.attn_qkv.weight",
            ),
            (
                "encoder.layers.0.attn.out_proj.weight",
                "blk.0.attn_output.weight",
            ),
            (
                "encoder.layers.3.norm1.weight",
                "blk.3.attn_output_norm.weight",
            ),
            (
                "encoder.layers.3.norm1.bias",
                "blk.3.attn_output_norm.bias",
            ),
            (
                "encoder.layers.7.mlp.fc11.weight",
                "blk.7.ffn_gate.weight",
            ),
            (
                "encoder.layers.7.mlp.fc12.weight",
                "blk.7.ffn_up.weight",
            ),
            (
                "encoder.layers.7.mlp.fc2.weight",
                "blk.7.ffn_down.weight",
            ),
            (
                "encoder.layers.9.norm2.weight",
                "blk.9.layer_output_norm.weight",
            ),
            (
                "encoder.layers.9.norm2.bias",
                "blk.9.layer_output_norm.bias",
            ),
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

    /// Acceptance test 1b — same canonical mappings work when the HF
    /// checkpoint carries the optional `bert.` prefix (some
    /// `BertForMaskedLM`-style exports do; nomic-embed-text-v1.5
    /// typically does not but the strip is harmless and mirrors
    /// `BertModel.filter_tensors`).
    #[test]
    fn nomic_bert_strips_optional_bert_prefix() {
        assert_eq!(
            map_tensor_name("bert.embeddings.word_embeddings.weight").as_deref(),
            Some("token_embd.weight"),
        );
        assert_eq!(
            map_tensor_name("bert.encoder.layers.0.attn.Wqkv.weight").as_deref(),
            Some("blk.0.attn_qkv.weight"),
        );
        assert_eq!(
            map_tensor_name("bert.encoder.layers.4.mlp.fc11.weight").as_deref(),
            Some("blk.4.ffn_gate.weight"),
        );
        assert_eq!(
            map_tensor_name("bert.emb_ln.bias").as_deref(),
            Some("token_embd_norm.bias"),
        );
    }

    /// Sibling — unknown names must surface as `None`. Per
    /// [[feedback-no-loop-suppression-2026-05-17]]: the caller is
    /// expected to error on this, never silently skip. This covers the
    /// three NomicBert-specific quirks that MUST be rejected:
    ///   - position embeddings (RoPE replaces them; HF checkpoints
    ///     typically omit this anyway)
    ///   - token_type embeddings (NomicBert drops segments)
    ///   - linear-projection biases (non-MoE NomicBert is weights-only
    ///     on Wqkv / out_proj / fc11 / fc12 / fc2)
    #[test]
    fn nomic_bert_tensor_name_rejects_unknown_kinds() {
        // Unused position-embedding table (RoPE replaces it).
        assert_eq!(
            map_tensor_name("embeddings.position_embeddings.weight"),
            None
        );
        // Unused token-type-embedding table.
        assert_eq!(
            map_tensor_name("embeddings.token_type_embeddings.weight"),
            None
        );
        // Linear-projection biases (NomicBert non-MoE has none).
        assert_eq!(
            map_tensor_name("encoder.layers.0.attn.Wqkv.bias"),
            None
        );
        assert_eq!(
            map_tensor_name("encoder.layers.0.attn.out_proj.bias"),
            None
        );
        assert_eq!(
            map_tensor_name("encoder.layers.0.mlp.fc11.bias"),
            None
        );
        assert_eq!(
            map_tensor_name("encoder.layers.0.mlp.fc12.bias"),
            None
        );
        assert_eq!(
            map_tensor_name("encoder.layers.0.mlp.fc2.bias"),
            None
        );
        // Wrong prefix (singular `layer` is BERT, not nomic-bert).
        assert_eq!(
            map_tensor_name("encoder.layer.0.attention.self.Wqkv.weight"),
            None
        );
        // BERT-style split QKV is NOT nomic — fused only.
        assert_eq!(
            map_tensor_name("encoder.layers.0.attention.self.query.weight"),
            None
        );
        // Plain-GELU BERT FFN names (fc1) do NOT exist on nomic non-MoE.
        assert_eq!(
            map_tensor_name("encoder.layers.0.mlp.fc1.weight"),
            None
        );
        // Malformed layer index (leading zero).
        assert_eq!(
            map_tensor_name("encoder.layers.01.attn.Wqkv.weight"),
            None
        );
        // Empty layer index.
        assert_eq!(
            map_tensor_name("encoder.layers..attn.Wqkv.weight"),
            None
        );
        // Unknown per-block suffix.
        assert_eq!(
            map_tensor_name("encoder.layers.0.unknown.weight"),
            None
        );
        // pooler — BertModel filter_tensors drops these; we just
        // return None (the convert pipeline reads it as "not a
        // weight we emit").
        assert_eq!(map_tensor_name("pooler.dense.weight"), None);
    }

    /// Acceptance test 2 — feed a minimal hand-written config.json with
    /// the canonical GPT-Neo-style keys (`n_embd`, `n_layer`, `n_head`,
    /// `n_inner`) used by nomic-embed-text-v1.5, and verify all 12 KV
    /// pairs come back with the right types + values.
    #[test]
    fn nomic_bert_metadata_built_from_gpt_style_config() {
        let cfg = json!({
            "_name_or_path": "nomic-ai/nomic-embed-text-v1.5",
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "n_inner": 3072,
            "max_position_embeddings": 2048,
            "layer_norm_epsilon": 1.0e-12,
            "rotary_emb_base": 1000.0,
        });

        let kv = build_metadata(&cfg, 17 /* MostlyQ5_K_M */);

        // Check count + keyset (don't depend on insertion order).
        assert_eq!(kv.len(), 12, "NomicBert emits 12 KV pairs at v1");
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();

        assert_eq!(
            by_key["general.architecture"],
            MetaValue::String("nomic-bert".into())
        );
        assert_eq!(
            by_key["general.name"],
            MetaValue::String("nomic-ai/nomic-embed-text-v1.5".into())
        );
        assert_eq!(by_key["nomic-bert.context_length"], MetaValue::U32(2048));
        assert_eq!(by_key["nomic-bert.embedding_length"], MetaValue::U32(768));
        assert_eq!(by_key["nomic-bert.block_count"], MetaValue::U32(12));
        assert_eq!(
            by_key["nomic-bert.feed_forward_length"],
            MetaValue::U32(3072)
        );
        assert_eq!(
            by_key["nomic-bert.attention.head_count"],
            MetaValue::U32(12)
        );
        assert_eq!(
            by_key["nomic-bert.attention.head_count_kv"],
            MetaValue::U32(12),
            "NomicBert has no GQA — head_count_kv == head_count"
        );
        assert_eq!(
            by_key["nomic-bert.attention.layer_norm_epsilon"],
            MetaValue::F32(1.0e-12)
        );
        assert_eq!(
            by_key["nomic-bert.rope.freq_base"],
            MetaValue::F32(1000.0)
        );
        assert_eq!(
            by_key["nomic-bert.pooling_type"],
            MetaValue::U32(1),
            "Nomic embeddings are mean-pooled (LLAMA_POOLING_TYPE_MEAN = 1)"
        );
        assert_eq!(by_key["general.file_type"], MetaValue::U32(17));
    }

    /// Sibling — verify the BERT-style key fallback (`hidden_size`,
    /// `num_hidden_layers`, `num_attention_heads`) and the
    /// optional-key defaults all trigger when the HF config omits the
    /// GPT-Neo keys + the defaulted keys.
    #[test]
    fn nomic_bert_metadata_bert_style_keys_and_defaults() {
        let cfg = json!({
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            // n_inner omitted → defaults to 4 * hidden_size = 256
            "max_position_embeddings": 512,
            // layer_norm_epsilon omitted → defaults to 1.0e-12
            // rotary_emb_base omitted → defaults to 10000.0
            // _name_or_path omitted → defaults to "model"
        });
        let kv = build_metadata(&cfg, 0);
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();

        assert_eq!(
            by_key["general.name"],
            MetaValue::String("model".into()),
            "name defaults to 'model' when _name_or_path absent"
        );
        assert_eq!(by_key["nomic-bert.embedding_length"], MetaValue::U32(64));
        assert_eq!(by_key["nomic-bert.block_count"], MetaValue::U32(2));
        assert_eq!(by_key["nomic-bert.attention.head_count"], MetaValue::U32(4));
        assert_eq!(
            by_key["nomic-bert.feed_forward_length"],
            MetaValue::U32(256),
            "n_inner defaults to 4 * embedding_length"
        );
        assert_eq!(
            by_key["nomic-bert.attention.layer_norm_epsilon"],
            MetaValue::F32(1.0e-12),
            "layer_norm_epsilon defaults to 1.0e-12 per BertConfig"
        );
        assert_eq!(
            by_key["nomic-bert.rope.freq_base"],
            MetaValue::F32(10000.0),
            "rotary_emb_base defaults to 10000.0"
        );
        assert_eq!(by_key["nomic-bert.pooling_type"], MetaValue::U32(1));
    }
}
