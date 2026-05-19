//! Gemma-4 HF→GGUF tensor-name + metadata mapper.
//!
//! Port of `/opt/llama.cpp/conversion/gemma.py::Gemma4Model`
//! (`@ModelBase.register("Gemma4ForConditionalGeneration")`,
//! `model_arch = gguf.MODEL_ARCH.GEMMA4`) and its inherited
//! `Gemma3Model`/`TextModel` `set_gguf_parameters` chain, restricted to
//! the dense decoder text path. The MoE expert tensors, NVFP4 fold-in,
//! vision/audio mmproj, and shared-KV layer tracking are all out of v1
//! scope (per ADR-033 §P4 "Per-arch convert-side mapping" — convert-only
//! at this phase; inference-side mapping handled separately).
//!
//! Per the user's mission spec: Gemma-4 emits `general.architecture =
//! "gemma3"` in GGUF (Gemma-4 reuses the Gemma-3 architecture name on
//! the GGUF side for interop with the existing gemma3 loader). KV
//! prefix is `gemma3.*` accordingly.
//!
//! Per [[feedback-no-backwards-compat-2026-05-18]]: no fallback, no
//! aliasing — every HF name we recognize maps to exactly one GGUF name,
//! and every other name returns `None` (caller decides surface). Per
//! [[feedback-no-loop-suppression-2026-05-17]]: callers MUST NOT
//! silently skip a `None` — propagate as a typed error.
//!
//! Distinctive Gemma-4 quirks vs Llama-3 (worth documenting at the
//! port-site, because they are non-obvious from the architecture name
//! alone):
//!
//! 1. **Tied embeddings.** Gemma-4 ties `lm_head` to the input embedding
//!    matrix (HF stores it on `model.embed_tokens.weight` and the
//!    forward pass re-uses the tied weights). Production HF releases
//!    therefore typically OMIT `lm_head.weight` from the safetensors
//!    shards. The mapper does NOT contain a `lm_head.weight → output.weight`
//!    entry; if a checkpoint does ship `lm_head.weight` (e.g. an autoawq
//!    fork that duplicated the tie), the caller — `ConvertOrchestrator`
//!    — gets `None` and surfaces it as `UnmappedTensor`, which the
//!    user is expected to resolve by dropping the duplicate at the
//!    safetensors-load layer (see `Gemma4Model.filter_tensors` in
//!    `conversion/gemma.py`: lm_head is skipped via parent
//!    `Gemma3Model.filter_tensors` → inheritance chain).
//!
//! 2. **Four norms per block.** Llama-3 has two (`attn_norm`,
//!    `ffn_norm`). Gemma-4 has FOUR: input pre-attn, post-attn,
//!    pre-FFN, post-FFN. The HF names + GGUF tags are spelled out in
//!    the table on `map_tensor_name` below.
//!
//! 3. **Q/K head-dim norms.** Gemma-4 has per-head Q/K LayerNorms
//!    inside the self-attention block (`self_attn.q_norm`,
//!    `self_attn.k_norm`) that map to `attn_q_norm` / `attn_k_norm`
//!    GGUF tensors. Llama-3 has neither.
//!
//! 4. **Optional per-layer token embedding.** Gemma-4 retains the
//!    Gemma-3n "per-layer token embedding" tensor mechanism (its main
//!    use is multimodal token routing, but the dense text checkpoints
//!    can carry it). If present, the HF name
//!    `model.per_layer_embed_tokens.weight` maps to
//!    `per_layer_token_embd.weight` (a global tensor, NOT per-block).
//!    If absent, the mapper simply doesn't see the name and the
//!    feature is silently disabled at inference. This is the ONLY
//!    "optional global" in Gemma-4's text-side inventory.

use crate::backends::gguf::types::MetaValue;

/// Translate one HuggingFace tensor name (as seen in `model.safetensors`)
/// to its canonical GGUF tensor name for Gemma-4. Returns `None` if
/// `hf_name` is not one of the Gemma-4 text-decoder weight kinds.
///
/// Gemma-4 weight kinds (globals + per-block):
///
/// | HF name                                                  | GGUF name                          |
/// |----------------------------------------------------------|------------------------------------|
/// | `model.embed_tokens.weight`                              | `token_embd.weight`                |
/// | `model.norm.weight`                                      | `output_norm.weight`               |
/// | `model.per_layer_embed_tokens.weight` *(optional)*       | `per_layer_token_embd.weight`      |
/// | `model.layers.<N>.input_layernorm.weight`                | `blk.<N>.attn_norm.weight`         |
/// | `model.layers.<N>.post_attention_layernorm.weight`       | `blk.<N>.post_attention_norm.weight` |
/// | `model.layers.<N>.pre_feedforward_layernorm.weight`      | `blk.<N>.ffn_norm.weight`          |
/// | `model.layers.<N>.post_feedforward_layernorm.weight`     | `blk.<N>.post_ffw_norm.weight`     |
/// | `model.layers.<N>.self_attn.q_proj.weight`               | `blk.<N>.attn_q.weight`            |
/// | `model.layers.<N>.self_attn.k_proj.weight`               | `blk.<N>.attn_k.weight`            |
/// | `model.layers.<N>.self_attn.v_proj.weight`               | `blk.<N>.attn_v.weight`            |
/// | `model.layers.<N>.self_attn.o_proj.weight`               | `blk.<N>.attn_output.weight`       |
/// | `model.layers.<N>.self_attn.q_norm.weight`               | `blk.<N>.attn_q_norm.weight`       |
/// | `model.layers.<N>.self_attn.k_norm.weight`               | `blk.<N>.attn_k_norm.weight`       |
/// | `model.layers.<N>.mlp.gate_proj.weight`                  | `blk.<N>.ffn_gate.weight`          |
/// | `model.layers.<N>.mlp.up_proj.weight`                    | `blk.<N>.ffn_up.weight`            |
/// | `model.layers.<N>.mlp.down_proj.weight`                  | `blk.<N>.ffn_down.weight`          |
///
/// Explicitly NOT mapped (and `None` on encounter — caller must drop or
/// error per [[feedback-no-loop-suppression-2026-05-17]]):
///
/// - `lm_head.weight` — Gemma-4 ties embeddings, so the GGUF loader
///   re-uses `token_embd.weight`. Production HF releases omit it; if a
///   fork ships it the caller drops the duplicate.
/// - MoE expert tensors (`mlp.experts.<E>.*`, `router.*`,
///   `per_expert_scale`) — out of v1 scope; ADR-033 §P4 dense decoder
///   path only.
/// - Vision / audio mmproj tensors (`vision_tower.*`, `audio_tower.*`,
///   `multi_modal_projector.*`) — handled by a separate mapper.
pub fn map_tensor_name(hf_name: &str) -> Option<String> {
    // ---- Globals ----------------------------------------------------------
    match hf_name {
        "model.embed_tokens.weight" => return Some("token_embd.weight".to_string()),
        "model.norm.weight" => return Some("output_norm.weight".to_string()),
        "model.per_layer_embed_tokens.weight" => {
            return Some("per_layer_token_embd.weight".to_string());
        }
        _ => {}
    }

    // ---- Per-block: `model.layers.<N>.<rest>` ----------------------------
    let stripped = hf_name.strip_prefix("model.layers.")?;
    let dot = stripped.find('.')?;
    let (layer_str, rest_with_dot) = stripped.split_at(dot);
    // Parse layer index (must be a bare non-negative integer; reject
    // leading zeros / signs to keep the mapper strict — matches the
    // Llama-3 mapper's policy).
    let layer: usize = layer_str.parse().ok()?;
    if layer.to_string() != layer_str {
        return None;
    }
    let rest = &rest_with_dot[1..]; // skip the dot

    let suffix = match rest {
        // Four block-level norms (Gemma-4 quirk #2)
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "post_attention_norm.weight",
        "pre_feedforward_layernorm.weight" => "ffn_norm.weight",
        "post_feedforward_layernorm.weight" => "post_ffw_norm.weight",

        // Attention projections (same naming as Llama-3)
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",

        // Per-head Q/K norms (Gemma-4 quirk #3 — absent from Llama-3)
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",

        // FFN (SwiGLU, same as Llama-3)
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",

        _ => return None,
    };

    Some(format!("blk.{layer}.{suffix}"))
}

/// Build the GGUF metadata KV pairs for a Gemma-4 model from its HF
/// `config.json`. Port of `conversion/gemma.py::Gemma3Model::set_gguf_parameters`
/// + the `TextModel::set_gguf_parameters` base it `super()`s into,
/// restricted to the keys every Gemma-4 dense-text release carries.
///
/// Per the user's mission spec, `general.architecture = "gemma3"` (and
/// the prefix is `gemma3.*`) — Gemma-4 reuses the Gemma-3 architecture
/// name on the GGUF side.
///
/// Required HF keys (mandatory; missing → caller-side panic):
///   - `hidden_size`
///   - `num_hidden_layers`
///   - `intermediate_size`
///   - `num_attention_heads`
///   - `max_position_embeddings`
///   - `rms_norm_eps`
///   - `head_dim` (Gemma quirk: explicit, not derivable from
///     `hidden_size / num_attention_heads` because Gemma-3/4 use a
///     larger head dim than that ratio would suggest — `head_dim = 256`
///     on the 4B/12B/27B Gemma-3 releases for instance, despite
///     `hidden_size / n_head = 64`. The Python convert mapper defaults
///     it to 256 when absent, but every Gemma release in the wild
///     supplies it explicitly, so we require it.)
///
/// Optional HF keys (defaulted, mirroring `Gemma3Model.set_gguf_parameters`):
///   - `num_key_value_heads` — defaults to `num_attention_heads`
///   - `rope_theta` — defaults to `10000.0`
///   - `sliding_window` — defaults to `4096`. Emitted unconditionally
///     because Gemma-3/4 use SWA (sliding-window attention) on
///     alternating layers; the inference loader needs this even when
///     the model JSON omits it.
///   - `_name_or_path` — defaults to `"model"`
///
/// `file_type` is the chosen `LlamaFtype` as a `u32` (matches
/// `gguf_writer.add_file_type(self.ftype)` at base.py).
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
    let head_dim = config["head_dim"]
        .as_u64()
        .expect("config.json missing required key `head_dim`") as u32;

    // Optional with defaults — mirror Gemma3Model.set_gguf_parameters.
    let n_head_kv = config
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32)
        .unwrap_or(n_head);
    let rope_theta = config
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0) as f32;
    let sliding_window = config
        .get("sliding_window")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32)
        .unwrap_or(4096);

    vec![
        (
            "general.architecture".into(),
            MetaValue::String("gemma3".into()),
        ),
        ("general.name".into(), MetaValue::String(name)),
        ("gemma3.context_length".into(), MetaValue::U32(ctx_len)),
        (
            "gemma3.embedding_length".into(),
            MetaValue::U32(hidden_size),
        ),
        ("gemma3.block_count".into(), MetaValue::U32(n_layers)),
        (
            "gemma3.feed_forward_length".into(),
            MetaValue::U32(ffn_len),
        ),
        (
            "gemma3.attention.head_count".into(),
            MetaValue::U32(n_head),
        ),
        (
            "gemma3.attention.head_count_kv".into(),
            MetaValue::U32(n_head_kv),
        ),
        (
            "gemma3.attention.key_length".into(),
            MetaValue::U32(head_dim),
        ),
        (
            "gemma3.attention.value_length".into(),
            MetaValue::U32(head_dim),
        ),
        (
            "gemma3.attention.layer_norm_rms_epsilon".into(),
            MetaValue::F32(rms_eps),
        ),
        ("gemma3.rope.freq_base".into(), MetaValue::F32(rope_theta)),
        (
            "gemma3.attention.sliding_window".into(),
            MetaValue::U32(sliding_window),
        ),
        ("general.file_type".into(), MetaValue::U32(file_type)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Acceptance test 1 — round-trip the canonical table for every HF
    /// name kind. Asserts `map_tensor_name(hf) → Some(gguf)` with the
    /// exact pair from the Gemma-4 inventory documented above.
    #[test]
    fn gemma4_tensor_name_round_trip() {
        let cases: &[(&str, &str)] = &[
            // Globals
            ("model.embed_tokens.weight", "token_embd.weight"),
            ("model.norm.weight", "output_norm.weight"),
            (
                "model.per_layer_embed_tokens.weight",
                "per_layer_token_embd.weight",
            ),
            // Per-block — sample at L=0, L=15, L=47 to cover edge / mid / depth.
            // (Gemma-3 27B has 62 blocks; this triple covers low/mid/high.)
            (
                "model.layers.0.input_layernorm.weight",
                "blk.0.attn_norm.weight",
            ),
            (
                "model.layers.15.input_layernorm.weight",
                "blk.15.attn_norm.weight",
            ),
            (
                "model.layers.47.input_layernorm.weight",
                "blk.47.attn_norm.weight",
            ),
            // Four norm quartet
            (
                "model.layers.0.input_layernorm.weight",
                "blk.0.attn_norm.weight",
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "blk.0.post_attention_norm.weight",
            ),
            (
                "model.layers.0.pre_feedforward_layernorm.weight",
                "blk.0.ffn_norm.weight",
            ),
            (
                "model.layers.0.post_feedforward_layernorm.weight",
                "blk.0.post_ffw_norm.weight",
            ),
            // Attention projections + head-dim norms
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
            (
                "model.layers.7.self_attn.q_norm.weight",
                "blk.7.attn_q_norm.weight",
            ),
            (
                "model.layers.7.self_attn.k_norm.weight",
                "blk.7.attn_k_norm.weight",
            ),
            // SwiGLU FFN
            (
                "model.layers.3.mlp.gate_proj.weight",
                "blk.3.ffn_gate.weight",
            ),
            ("model.layers.3.mlp.up_proj.weight", "blk.3.ffn_up.weight"),
            (
                "model.layers.3.mlp.down_proj.weight",
                "blk.3.ffn_down.weight",
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

    /// Sibling — unknown names must surface as `None`. Per
    /// [[feedback-no-loop-suppression-2026-05-17]]: the caller is
    /// expected to error on this, never silently skip.
    #[test]
    fn gemma4_tensor_name_rejects_unknown_kinds() {
        // Tied embedding — Gemma-4 ties lm_head to token_embd, so the
        // safetensors should not carry it. If a fork DOES ship it, the
        // mapper returns None and the caller drops it explicitly (it is
        // a duplicate of token_embd; no inference value).
        assert_eq!(map_tensor_name("lm_head.weight"), None);
        // Unknown global.
        assert_eq!(map_tensor_name("model.unknown.weight"), None);
        // Wrong prefix.
        assert_eq!(map_tensor_name("transformer.layers.0.attn.weight"), None);
        // Gemma-4 has no biases on linear projections.
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
        // MoE expert (out of v1 scope — must NOT silently fall through
        // to a dense FFN name).
        assert_eq!(
            map_tensor_name("model.layers.0.mlp.experts.0.gate_proj.weight"),
            None
        );
        // Router scale (out of v1 scope).
        assert_eq!(map_tensor_name("model.layers.0.mlp.router.scale"), None);
        // Vision tower (handled by a separate mapper).
        assert_eq!(
            map_tensor_name("model.vision_tower.encoder.layer.0.attention.q_proj.weight"),
            None
        );
    }

    /// Acceptance test 2 — feed a minimal hand-written config.json and
    /// verify all 14 KV pairs come back with the right types + values.
    #[test]
    fn gemma4_metadata_built_from_config() {
        let cfg = json!({
            "_name_or_path": "google/gemma-4-27b-it",
            "hidden_size": 5376,
            "num_hidden_layers": 62,
            "intermediate_size": 21504,
            "num_attention_heads": 32,
            "num_key_value_heads": 16,
            "head_dim": 128,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1.0e-6,
            "rope_theta": 1_000_000.0,
            "sliding_window": 1024,
        });

        let kv = build_metadata(&cfg, 17 /* MostlyQ5_K_M */);

        // Build a name -> value index for the asserts (don't depend on
        // insertion order — but we DO check count + keyset).
        assert_eq!(kv.len(), 14, "Gemma4 emits 14 KV pairs at v1");
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();

        assert_eq!(
            by_key["general.architecture"],
            MetaValue::String("gemma3".into()),
            "Gemma-4 reuses the gemma3 architecture name in GGUF"
        );
        assert_eq!(
            by_key["general.name"],
            MetaValue::String("google/gemma-4-27b-it".into())
        );
        assert_eq!(by_key["gemma3.context_length"], MetaValue::U32(131072));
        assert_eq!(by_key["gemma3.embedding_length"], MetaValue::U32(5376));
        assert_eq!(by_key["gemma3.block_count"], MetaValue::U32(62));
        assert_eq!(by_key["gemma3.feed_forward_length"], MetaValue::U32(21504));
        assert_eq!(by_key["gemma3.attention.head_count"], MetaValue::U32(32));
        assert_eq!(by_key["gemma3.attention.head_count_kv"], MetaValue::U32(16));
        assert_eq!(by_key["gemma3.attention.key_length"], MetaValue::U32(128));
        assert_eq!(by_key["gemma3.attention.value_length"], MetaValue::U32(128));
        assert_eq!(
            by_key["gemma3.attention.layer_norm_rms_epsilon"],
            MetaValue::F32(1.0e-6)
        );
        assert_eq!(
            by_key["gemma3.rope.freq_base"],
            MetaValue::F32(1_000_000.0)
        );
        assert_eq!(by_key["gemma3.attention.sliding_window"], MetaValue::U32(1024));
        assert_eq!(by_key["general.file_type"], MetaValue::U32(17));
    }

    /// Sibling — verify the optional-key defaults trigger when the HF
    /// config omits `num_key_value_heads`, `rope_theta`,
    /// `sliding_window`, and `_name_or_path`. Mirrors
    /// `Gemma3Model.set_gguf_parameters`'s `.get(...)` defaults.
    #[test]
    fn gemma4_metadata_optional_key_defaults() {
        let cfg = json!({
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            // num_key_value_heads omitted → defaults to num_attention_heads
            "head_dim": 16,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1.0e-6,
            // rope_theta omitted → defaults to 10000.0
            // sliding_window omitted → defaults to 4096
            // _name_or_path omitted → defaults to "model"
        });
        let kv = build_metadata(&cfg, 0);
        assert_eq!(kv.len(), 14, "Gemma4 always emits 14 KV pairs");
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        assert_eq!(
            by_key["general.name"],
            MetaValue::String("model".into()),
            "name defaults to 'model' when _name_or_path absent"
        );
        assert_eq!(
            by_key["gemma3.attention.head_count_kv"],
            MetaValue::U32(4),
            "num_key_value_heads defaults to num_attention_heads"
        );
        assert_eq!(
            by_key["gemma3.rope.freq_base"],
            MetaValue::F32(10000.0),
            "rope_theta defaults to 10000.0"
        );
        assert_eq!(
            by_key["gemma3.attention.sliding_window"],
            MetaValue::U32(4096),
            "sliding_window defaults to 4096"
        );
        assert_eq!(
            by_key["general.architecture"],
            MetaValue::String("gemma3".into()),
        );
        assert_eq!(by_key["general.file_type"], MetaValue::U32(0));
    }

    /// Sibling — `file_type` round-trips as a plain u32 in
    /// `general.file_type` (matches `gguf_writer.add_file_type(self.ftype)`
    /// at base.py). Tests every nonzero ftype isn't dropped or coerced.
    #[test]
    fn gemma4_metadata_ftype_round_trips() {
        let cfg = json!({
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "head_dim": 16,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1.0e-6,
        });
        for &ftype in &[0u32, 1, 7, 15, 17, 23] {
            let kv = build_metadata(&cfg, ftype);
            let by_key: std::collections::HashMap<_, _> =
                kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
            assert_eq!(
                by_key["general.file_type"],
                MetaValue::U32(ftype),
                "file_type {ftype} must round-trip as MetaValue::U32"
            );
        }
    }
}
