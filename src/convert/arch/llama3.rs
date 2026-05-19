//! Llama-3 HF→GGUF tensor-name + metadata mapper.
//!
//! Port of `/opt/llama.cpp/conversion/llama.py::LlamaModel`'s name
//! mapping (transitively via `base.py::TextModel::modify_tensors` ->
//! `gguf-py/gguf/tensor_mapping.py`) and `set_gguf_parameters`. Strictly
//! the dense-decoder Llama-3 path — no Mistral / Llama4 / multimodal
//! special-cases.
//!
//! Per ADR-033 §P0 "Per-arch convert-side mapping": this is the
//! convert-side tensor-name + KV mapper for `LLM_ARCH_LLAMA`. The
//! inference-side mapping for Llama3 is out of v1 scope (Llama3 is
//! convert-only per the ADR convert matrix).
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
/// one of the 11 Llama-3 weight kinds.
///
/// Llama-3 weight kinds (one global trio + eight per-block):
///
/// | HF name                                                  | GGUF name                       |
/// |----------------------------------------------------------|----------------------------------|
/// | `model.embed_tokens.weight`                              | `token_embd.weight`              |
/// | `model.norm.weight`                                      | `output_norm.weight`             |
/// | `lm_head.weight`                                         | `output.weight`                  |
/// | `model.layers.<N>.input_layernorm.weight`                | `blk.<N>.attn_norm.weight`       |
/// | `model.layers.<N>.post_attention_layernorm.weight`       | `blk.<N>.ffn_norm.weight`        |
/// | `model.layers.<N>.self_attn.q_proj.weight`               | `blk.<N>.attn_q.weight`          |
/// | `model.layers.<N>.self_attn.k_proj.weight`               | `blk.<N>.attn_k.weight`          |
/// | `model.layers.<N>.self_attn.v_proj.weight`               | `blk.<N>.attn_v.weight`          |
/// | `model.layers.<N>.self_attn.o_proj.weight`               | `blk.<N>.attn_output.weight`     |
/// | `model.layers.<N>.mlp.gate_proj.weight`                  | `blk.<N>.ffn_gate.weight`        |
/// | `model.layers.<N>.mlp.up_proj.weight`                    | `blk.<N>.ffn_up.weight`          |
/// | `model.layers.<N>.mlp.down_proj.weight`                  | `blk.<N>.ffn_down.weight`        |
pub fn map_tensor_name(hf_name: &str) -> Option<String> {
    // ---- Globals ----------------------------------------------------------
    match hf_name {
        "model.embed_tokens.weight" => return Some("token_embd.weight".to_string()),
        "model.norm.weight" => return Some("output_norm.weight".to_string()),
        "lm_head.weight" => return Some("output.weight".to_string()),
        _ => {}
    }

    // ---- Per-block: `model.layers.<N>.<rest>` ----------------------------
    let stripped = hf_name.strip_prefix("model.layers.")?;
    let dot = stripped.find('.')?;
    let (layer_str, rest_with_dot) = stripped.split_at(dot);
    // Parse layer index (must be a bare non-negative integer; reject
    // leading zeros / signs to keep the mapper strict).
    let layer: usize = layer_str.parse().ok()?;
    if layer.to_string() != layer_str {
        // Reject `00`, `+0`, etc. — pedantic but cheap.
        return None;
    }
    let rest = &rest_with_dot[1..]; // skip the dot

    let suffix = match rest {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        _ => return None,
    };

    Some(format!("blk.{layer}.{suffix}"))
}

/// Build the GGUF metadata KV pairs for a Llama-3 model from its HF
/// `config.json`. Port of `conversion/llama.py::LlamaModel::set_gguf_parameters`
/// + the `TextModel::set_gguf_parameters` base it `super()`s into
/// (`/opt/llama.cpp/conversion/base.py:1111-1221`), restricted to the
/// keys every Llama-3-8B class model actually carries.
///
/// Required HF keys (mandatory; missing key → caller-side panic from
/// the `[]` indexing):
///   - `hidden_size`
///   - `num_hidden_layers`
///   - `intermediate_size`
///   - `num_attention_heads`
///   - `max_position_embeddings`
///   - `rms_norm_eps`
///
/// Optional HF keys (defaulted):
///   - `num_key_value_heads` — defaults to `num_attention_heads` (MHA;
///     present on every Llama-3 model the operator runs, but the field
///     IS optional in the HF schema — older Llama-2 architecture
///     configs may omit it. Default-to-num_heads mirrors HF's
///     `LlamaConfig.__init__` (`transformers/models/llama/configuration_llama.py`).
///   - `rope_theta` — defaults to `10000.0` per llama.cpp's base.py
///     read (it doesn't emit the KV if `rope_theta` is absent; we do,
///     because every Llama-3 we'll convert has it and the canonical
///     pipeline emits `10000.0` as the implicit C default in
///     `llama-hparams.cpp::llama_hparams_init`).
///   - `_name_or_path` — defaults to `"model"`.
///
/// `file_type` is the chosen `LlamaFtype` as a `u32` (matches
/// `gguf_writer.add_file_type(self.ftype)` at base.py:1220).
pub fn build_metadata(config: &serde_json::Value, file_type: u32) -> Vec<(String, MetaValue)> {
    let name = config
        .get("_name_or_path")
        .and_then(|v| v.as_str())
        .unwrap_or("model")
        .to_string();

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
        .expect("config.json missing required key `max_position_embeddings`") as u32;
    let rms_eps = config["rms_norm_eps"]
        .as_f64()
        .expect("config.json missing required key `rms_norm_eps`") as f32;

    // Optional with defaults — both keys can be absent on older Llama-2
    // class configs; Llama-3 always supplies them, but the mapper
    // tolerates absence to match HF + llama.cpp behavior.
    let n_head_kv = config
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32)
        .unwrap_or(n_head);
    let rope_theta = config
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0) as f32;

    vec![
        (
            "general.architecture".into(),
            MetaValue::String("llama".into()),
        ),
        ("general.name".into(), MetaValue::String(name)),
        ("llama.context_length".into(), MetaValue::U32(ctx_len)),
        ("llama.embedding_length".into(), MetaValue::U32(hidden_size)),
        ("llama.block_count".into(), MetaValue::U32(n_layers)),
        ("llama.feed_forward_length".into(), MetaValue::U32(ffn_len)),
        ("llama.attention.head_count".into(), MetaValue::U32(n_head)),
        ("llama.attention.head_count_kv".into(), MetaValue::U32(n_head_kv)),
        (
            "llama.attention.layer_norm_rms_epsilon".into(),
            MetaValue::F32(rms_eps),
        ),
        ("llama.rope.freq_base".into(), MetaValue::F32(rope_theta)),
        ("general.file_type".into(), MetaValue::U32(file_type)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Acceptance test 1 — round-trip the canonical table for every HF
    /// name kind. Asserts `map_tensor_name(hf) → Some(gguf)` with the
    /// exact pair from the ADR-033 §P0 table.
    #[test]
    fn llama3_tensor_name_round_trip() {
        let cases: &[(&str, &str)] = &[
            // Globals
            ("model.embed_tokens.weight", "token_embd.weight"),
            ("model.norm.weight", "output_norm.weight"),
            ("lm_head.weight", "output.weight"),
            // Per-block — sample at L=0, L=15, L=31 to cover edge / mid / depth.
            (
                "model.layers.0.input_layernorm.weight",
                "blk.0.attn_norm.weight",
            ),
            (
                "model.layers.15.input_layernorm.weight",
                "blk.15.attn_norm.weight",
            ),
            (
                "model.layers.31.input_layernorm.weight",
                "blk.31.attn_norm.weight",
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "blk.0.ffn_norm.weight",
            ),
            (
                "model.layers.7.self_attn.q_proj.weight",
                "blk.7.attn_q.weight",
            ),
            (
                "model.layers.7.self_attn.k_proj.weight",
                "blk.7.attn_k.weight",
            ),
            (
                "model.layers.7.self_attn.v_proj.weight",
                "blk.7.attn_v.weight",
            ),
            (
                "model.layers.7.self_attn.o_proj.weight",
                "blk.7.attn_output.weight",
            ),
            ("model.layers.3.mlp.gate_proj.weight", "blk.3.ffn_gate.weight"),
            ("model.layers.3.mlp.up_proj.weight", "blk.3.ffn_up.weight"),
            ("model.layers.3.mlp.down_proj.weight", "blk.3.ffn_down.weight"),
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

    /// Sibling — unknown names must surface as `None`. Per
    /// [[feedback-no-loop-suppression-2026-05-17]]: the caller is
    /// expected to error on this, never silently skip.
    #[test]
    fn llama3_tensor_name_rejects_unknown_kinds() {
        // Unknown global.
        assert_eq!(map_tensor_name("model.unknown.weight"), None);
        // Wrong prefix.
        assert_eq!(
            map_tensor_name("transformer.layers.0.attn.weight"),
            None
        );
        // Llama-3 has no biases on linear projections.
        assert_eq!(
            map_tensor_name("model.layers.0.self_attn.q_proj.bias"),
            None
        );
        // Malformed layer index (leading zero).
        assert_eq!(
            map_tensor_name("model.layers.01.self_attn.q_proj.weight"),
            None
        );
        // Empty layer index.
        assert_eq!(
            map_tensor_name("model.layers..self_attn.q_proj.weight"),
            None
        );
        // No layer prefix at all.
        assert_eq!(
            map_tensor_name("model.layers.self_attn.q_proj.weight"),
            None
        );
        // Unknown per-block suffix.
        assert_eq!(map_tensor_name("model.layers.0.unknown.weight"), None);
    }

    /// Acceptance test 2 — feed a minimal hand-written config.json and
    /// verify all 11 KV pairs come back with the right types + values.
    #[test]
    fn llama3_metadata_built_from_config() {
        let cfg = json!({
            "_name_or_path": "meta-llama/Llama-3-Tiny",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1.0e-5,
            "rope_theta": 500000.0,
        });

        let kv = build_metadata(&cfg, 17 /* MostlyQ5_K_M */);

        // Build a name -> value index for the asserts (don't depend on
        // insertion order — but we DO check count + keyset).
        assert_eq!(kv.len(), 11, "Llama3 emits 11 KV pairs at v1");
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();

        assert_eq!(
            by_key["general.architecture"],
            MetaValue::String("llama".into())
        );
        assert_eq!(
            by_key["general.name"],
            MetaValue::String("meta-llama/Llama-3-Tiny".into())
        );
        assert_eq!(by_key["llama.context_length"], MetaValue::U32(8192));
        assert_eq!(by_key["llama.embedding_length"], MetaValue::U32(32));
        assert_eq!(by_key["llama.block_count"], MetaValue::U32(2));
        assert_eq!(by_key["llama.feed_forward_length"], MetaValue::U32(64));
        assert_eq!(by_key["llama.attention.head_count"], MetaValue::U32(2));
        assert_eq!(by_key["llama.attention.head_count_kv"], MetaValue::U32(1));
        assert_eq!(
            by_key["llama.attention.layer_norm_rms_epsilon"],
            MetaValue::F32(1.0e-5)
        );
        assert_eq!(by_key["llama.rope.freq_base"], MetaValue::F32(500000.0));
        assert_eq!(by_key["general.file_type"], MetaValue::U32(17));
    }

    /// Sibling — verify the optional-key defaults trigger when the HF
    /// config omits `num_key_value_heads`, `rope_theta`, and
    /// `_name_or_path`. Matches HF's `LlamaConfig` defaults.
    #[test]
    fn llama3_metadata_optional_key_defaults() {
        let cfg = json!({
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            // num_key_value_heads omitted → defaults to num_attention_heads
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1.0e-6,
            // rope_theta omitted → defaults to 10000.0
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
        assert_eq!(
            by_key["llama.attention.head_count_kv"],
            MetaValue::U32(4),
            "num_key_value_heads defaults to num_attention_heads"
        );
        assert_eq!(
            by_key["llama.rope.freq_base"],
            MetaValue::F32(10000.0),
            "rope_theta defaults to 10000.0"
        );
    }
}
