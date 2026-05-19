//! Qwen3-VL **text-decoder** HF→GGUF tensor-name + metadata mapper.
//!
//! Port of `/opt/llama.cpp/conversion/qwen3vl.py::Qwen3VLTextModel`
//! (`@ModelBase.register("Qwen3VLForConditionalGeneration")`,
//! `model_arch = gguf.MODEL_ARCH.QWEN3VL`). `Qwen3VLTextModel` inherits
//! from `Qwen3Model` → `Qwen2Model` → `TextModel`, so the tensor-name
//! mapping is identical to dense Qwen3 (Llama3 + per-head Q/K RMS norms,
//! same as Gemma-4 quirk #3). The Qwen3-VL-specific bits are:
//!
//!   1. **`thinker.` prefix stripping** (`Qwen3VLTextModel.filter_tensors`
//!      at `conversion/qwen3vl.py:267`): the multimodal `thinker.*`
//!      wrapper is unwrapped before super().filter_tensors runs. The
//!      mapper accepts BOTH `thinker.model.embed_tokens.weight` AND bare
//!      `model.embed_tokens.weight`, since v1 HF shards have appeared in
//!      both shapes in the wild.
//!
//!   2. **`general.architecture = "qwen3vl"`** (per
//!      `gguf-py/gguf/constants.py:928`, `MODEL_ARCH_NAMES[QWEN3VL] =
//!      "qwen3vl"`). KV prefix is `qwen3vl.*` accordingly.
//!
//!   3. **`qwen3vl.n_deepstack_layers`** (uint32; `Keys.LLM.NUM_DEEPSTACK_LAYERS`
//!      at `gguf-py/gguf/constants.py:130`, `{arch}.n_deepstack_layers`).
//!      Counts the number of vision-side deepstack heads — the TEXT
//!      decoder needs the count so that the inference loader knows how
//!      many auxiliary residual injection points to wire. Sourced from
//!      `vision_config.deepstack_visual_indexes` (or
//!      `thinker_config.vision_config.deepstack_visual_indexes` on Omni
//!      configs). Defaults to 0 when vision_config is absent (text-only
//!      shard).
//!
//!   4. **`qwen3vl.rope.dimension_sections`** (array of uint32, padded to
//!      4 with trailing zeros — `Keys.Rope.DIMENSION_SECTIONS` at
//!      `gguf-py/gguf/constants.py:196`). Qwen3-VL uses **multi-modal
//!      RoPE (M-RoPE)** which splits the rope head-dim across [time,
//!      height, width, extra] axes. Sourced from
//!      `config["rope_scaling"]["mrope_section"]`; emitted only when
//!      present (text-only Qwen3 variants don't supply it).
//!
//!   5. **`qwen3vl.rope.freq_base` defaults to 1_000_000.0**, not the
//!      Llama3 / Gemma3 10_000.0. Every Qwen3-VL release the operator
//!      runs supplies `rope_theta` explicitly (typically `5_000_000.0`),
//!      so the default is only a defensive fallback.
//!
//! **Deepstack tensors live on the VISION (mmproj) side** — see
//! `/opt/hf2q/src/models/vit/convert.rs::hf_qwen3vl_deepstack_to_gguf`
//! and the test cases `wedge4f_qwen3vl_deepstack_*`. The text-side
//! mapper does NOT remap them; per
//! [[feedback-no-loop-suppression-2026-05-17]] we return `None` so the
//! caller can decide (drop for text-only convert, route to the mmproj
//! mapper for joint convert). Silent skip is forbidden.
//!
//! Per [[feedback-no-backwards-compat-2026-05-18]]: no fallback / no
//! aliasing — every HF name we recognize maps to exactly one GGUF name,
//! every other name returns `None`.

use crate::backends::gguf::types::MetaValue;

/// Translate one HuggingFace tensor name (as seen in `model.safetensors`)
/// to its canonical GGUF tensor name for the Qwen3-VL **text** decoder.
/// Returns `None` if `hf_name` is not one of the text-decoder weight
/// kinds.
///
/// Both bare (`model.embed_tokens.weight`) and `thinker.`-prefixed
/// (`thinker.model.embed_tokens.weight`) forms are accepted; the
/// prefix is stripped before matching per
/// `Qwen3VLTextModel.filter_tensors` at `conversion/qwen3vl.py:267`.
///
/// Qwen3-VL text-decoder weight kinds (globals + per-block):
///
/// | HF name                                                  | GGUF name                       |
/// |----------------------------------------------------------|----------------------------------|
/// | `model.embed_tokens.weight`                              | `token_embd.weight`              |
/// | `model.norm.weight`                                      | `output_norm.weight`             |
/// | `lm_head.weight` *(only if not tied)*                    | `output.weight`                  |
/// | `model.layers.<N>.input_layernorm.weight`                | `blk.<N>.attn_norm.weight`       |
/// | `model.layers.<N>.post_attention_layernorm.weight`       | `blk.<N>.ffn_norm.weight`        |
/// | `model.layers.<N>.self_attn.q_proj.weight`               | `blk.<N>.attn_q.weight`          |
/// | `model.layers.<N>.self_attn.k_proj.weight`               | `blk.<N>.attn_k.weight`          |
/// | `model.layers.<N>.self_attn.v_proj.weight`               | `blk.<N>.attn_v.weight`          |
/// | `model.layers.<N>.self_attn.o_proj.weight`               | `blk.<N>.attn_output.weight`     |
/// | `model.layers.<N>.self_attn.q_norm.weight`               | `blk.<N>.attn_q_norm.weight`     |
/// | `model.layers.<N>.self_attn.k_norm.weight`               | `blk.<N>.attn_k_norm.weight`     |
/// | `model.layers.<N>.mlp.gate_proj.weight`                  | `blk.<N>.ffn_gate.weight`        |
/// | `model.layers.<N>.mlp.up_proj.weight`                    | `blk.<N>.ffn_up.weight`           |
/// | `model.layers.<N>.mlp.down_proj.weight`                  | `blk.<N>.ffn_down.weight`        |
///
/// Explicitly NOT mapped (and `None` on encounter — caller MUST surface
/// per [[feedback-no-loop-suppression-2026-05-17]]):
///
/// - **Vision tensors** (`visual.*`, `model.visual.*`) — handled by the
///   vit/mmproj mapper at `src/models/vit/convert.rs`.
/// - **Deepstack merger tensors** (`visual.deepstack_merger_list.<N>.*`,
///   `model.visual.deepstack_merger_list.<N>.*`) — these are the
///   per-deepstack-head FC + norm tensors that live in the mmproj
///   GGUF, NOT the text decoder. Qwen3-VL injects vision features into
///   the text residual stream at certain text-layer indices, but the
///   weights for that injection ride in the mmproj file
///   (`v.deepstack.<abs_idx>.{norm,fc1,fc2}.{weight,bias}`). The text
///   decoder itself carries no deepstack-specific tensors.
/// - **MTP (multi-token-prediction) tensors** (`mtp.*`) — out of v1
///   scope.
/// - **Audio tower tensors** (`audio_tower.*`, `thinker.audio_tower.*`)
///   — Qwen3-Omni audio side, handled by a separate mapper.
pub fn map_tensor_name(hf_name: &str) -> Option<String> {
    // Strip Qwen3-VL's `thinker.` multimodal-wrapper prefix if present
    // (matches `Qwen3VLTextModel.filter_tensors` at
    // `conversion/qwen3vl.py:265`: `name = name.replace("thinker.", "")`).
    // We use a strict prefix strip (only at index 0), not a global
    // replace, because the canonical filter only touches the leading
    // wrapper — a mid-name `thinker.` would be a corrupted shard and
    // should fall through to the unmapped path.
    let name = hf_name.strip_prefix("thinker.").unwrap_or(hf_name);

    // ---- Reject vision-side tensors explicitly (deepstack + visual
    // tower live in mmproj, NOT in the text decoder). ----------------
    if name.starts_with("visual.") || name.starts_with("model.visual.") {
        return None;
    }
    // MTP and audio-tower are also not text-decoder tensors.
    if name.starts_with("mtp.") || name.starts_with("audio_tower.") {
        return None;
    }

    // ---- Globals ----------------------------------------------------------
    match name {
        "model.embed_tokens.weight" => return Some("token_embd.weight".to_string()),
        "model.norm.weight" => return Some("output_norm.weight".to_string()),
        "lm_head.weight" => return Some("output.weight".to_string()),
        _ => {}
    }

    // ---- Per-block: `model.layers.<N>.<rest>` ----------------------------
    let stripped = name.strip_prefix("model.layers.")?;
    let dot = stripped.find('.')?;
    let (layer_str, rest_with_dot) = stripped.split_at(dot);
    // Parse layer index (must be a bare non-negative integer; reject
    // leading zeros / signs to keep the mapper strict — matches Llama-3
    // and Gemma-4 policy).
    let layer: usize = layer_str.parse().ok()?;
    if layer.to_string() != layer_str {
        return None;
    }
    let rest = &rest_with_dot[1..]; // skip the dot

    let suffix = match rest {
        // Block-level norms (Llama-style: 2 norms per block)
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",

        // Attention projections (Qwen3 has no biases)
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",

        // Per-head Q/K RMSNorms (Qwen3 family quirk — same as Gemma-4)
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",

        // FFN (SwiGLU)
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",

        _ => return None,
    };

    Some(format!("blk.{layer}.{suffix}"))
}

/// Count the number of deepstack vision injection points for a given
/// Qwen3-VL HF config. Returns the length of `deepstack_visual_indexes`
/// (looked up at `vision_config.deepstack_visual_indexes` OR
/// `thinker_config.vision_config.deepstack_visual_indexes` for the Omni
/// shape). Returns `0` when no vision_config is present (text-only
/// shard).
///
/// Per `Qwen3VLTextModel.set_gguf_parameters` at
/// `conversion/qwen3vl.py:252-259`:
///
/// ```python
/// if "thinker_config" in self.hparams:
///     vision_config = self.hparams["thinker_config"].get("vision_config", {})
/// else:
///     vision_config = self.hparams.get("vision_config", {})
/// deepstack_layer_num = len(vision_config.get("deepstack_visual_indexes", []))
/// ```
fn count_deepstack_layers(config: &serde_json::Value) -> u32 {
    let vc = config
        .get("thinker_config")
        .and_then(|tc| tc.get("vision_config"))
        .or_else(|| config.get("vision_config"));
    vc.and_then(|v| v.get("deepstack_visual_indexes"))
        .and_then(|a| a.as_array())
        .map(|a| a.len() as u32)
        .unwrap_or(0)
}

/// Build the GGUF metadata KV pairs for a Qwen3-VL **text decoder** from
/// its HF `config.json`. Port of
/// `conversion/qwen3vl.py::Qwen3VLTextModel::set_gguf_parameters` +
/// the chain it inherits (`Qwen3Model` → `Qwen2Model` → `TextModel`),
/// restricted to the keys every Qwen3-VL dense-text release carries.
///
/// Required HF keys (mandatory; missing → caller-side panic):
///   - `hidden_size`
///   - `num_hidden_layers`
///   - `intermediate_size`
///   - `num_attention_heads`
///   - `max_position_embeddings`
///   - `rms_norm_eps`
///
/// Optional HF keys (defaulted):
///   - `num_key_value_heads` — defaults to `num_attention_heads`
///   - `head_dim` — emitted as `qwen3vl.attention.{key,value}_length`
///     when present; omitted entirely when absent (matches
///     `base.py:1216` `if (head_dim := self.hparams.get("head_dim")) is
///     not None`).
///   - `rope_theta` — defaults to `1_000_000.0` (Qwen3-VL convention;
///     every Qwen3-VL release supplies it explicitly so the default is
///     a defensive fallback only).
///   - `rope_scaling.mrope_section` — when present, emitted as
///     `qwen3vl.rope.dimension_sections` (array of 4 uint32, padded
///     with trailing zeros per `base.py:1174-1180`).
///   - `_name_or_path` — defaults to `"model"`.
///   - `vision_config.deepstack_visual_indexes` /
///     `thinker_config.vision_config.deepstack_visual_indexes` —
///     length used to compute `qwen3vl.n_deepstack_layers`; defaults
///     to 0 when absent (text-only shard).
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

    // Optional with defaults — mirror Qwen3 / TextModel base behavior.
    let n_head_kv = config
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32)
        .unwrap_or(n_head);
    let rope_theta = config
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(1_000_000.0) as f32;
    let head_dim = config
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32);

    // M-RoPE (multi-modal RoPE) section split, optional. Per
    // `base.py:1174-1180`: pad to 4 entries with trailing zeros, take
    // first 4. Present on every Qwen3-VL release; absent on text-only
    // Qwen3 variants.
    let mrope_section: Option<Vec<u32>> = config
        .get("rope_scaling")
        .and_then(|rs| rs.get("mrope_section"))
        .and_then(|m| m.as_array())
        .map(|arr| {
            let mut out: Vec<u32> = arr
                .iter()
                .filter_map(|v| v.as_u64().map(|x| x as u32))
                .collect();
            while out.len() < 4 {
                out.push(0);
            }
            out.truncate(4);
            out
        });

    let n_deepstack = count_deepstack_layers(config);

    let mut kv: Vec<(String, MetaValue)> = Vec::with_capacity(16);
    kv.push((
        "general.architecture".into(),
        MetaValue::String("qwen3vl".into()),
    ));
    kv.push(("general.name".into(), MetaValue::String(name)));
    kv.push(("qwen3vl.context_length".into(), MetaValue::U32(ctx_len)));
    kv.push((
        "qwen3vl.embedding_length".into(),
        MetaValue::U32(hidden_size),
    ));
    kv.push(("qwen3vl.block_count".into(), MetaValue::U32(n_layers)));
    kv.push((
        "qwen3vl.feed_forward_length".into(),
        MetaValue::U32(ffn_len),
    ));
    kv.push((
        "qwen3vl.attention.head_count".into(),
        MetaValue::U32(n_head),
    ));
    kv.push((
        "qwen3vl.attention.head_count_kv".into(),
        MetaValue::U32(n_head_kv),
    ));
    if let Some(hd) = head_dim {
        kv.push(("qwen3vl.attention.key_length".into(), MetaValue::U32(hd)));
        kv.push(("qwen3vl.attention.value_length".into(), MetaValue::U32(hd)));
    }
    kv.push((
        "qwen3vl.attention.layer_norm_rms_epsilon".into(),
        MetaValue::F32(rms_eps),
    ));
    kv.push(("qwen3vl.rope.freq_base".into(), MetaValue::F32(rope_theta)));
    if let Some(sections) = mrope_section {
        kv.push((
            "qwen3vl.rope.dimension_sections".into(),
            MetaValue::ArrayU32(sections),
        ));
    }
    kv.push((
        "qwen3vl.n_deepstack_layers".into(),
        MetaValue::U32(n_deepstack),
    ));
    kv.push(("general.file_type".into(), MetaValue::U32(file_type)));

    kv
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Acceptance test 1 — round-trip the canonical table for every HF
    /// name kind, including the `thinker.`-prefixed mirror.
    #[test]
    fn qwen3vl_text_tensor_name_round_trip() {
        let cases: &[(&str, &str)] = &[
            // Globals (bare)
            ("model.embed_tokens.weight", "token_embd.weight"),
            ("model.norm.weight", "output_norm.weight"),
            ("lm_head.weight", "output.weight"),
            // Globals via `thinker.` wrapper (Qwen3-VL multimodal shape)
            ("thinker.model.embed_tokens.weight", "token_embd.weight"),
            ("thinker.model.norm.weight", "output_norm.weight"),
            ("thinker.lm_head.weight", "output.weight"),
            // Per-block: sample at L=0, L=15, L=35 (Qwen3-VL-2B has 28
            // layers, Qwen3-VL-7B has 28+, 30B+ has 36+ — cover low/mid/depth).
            (
                "model.layers.0.input_layernorm.weight",
                "blk.0.attn_norm.weight",
            ),
            (
                "model.layers.15.input_layernorm.weight",
                "blk.15.attn_norm.weight",
            ),
            (
                "model.layers.35.input_layernorm.weight",
                "blk.35.attn_norm.weight",
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "blk.0.ffn_norm.weight",
            ),
            // Attention projections + per-head Q/K norms
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
            // `thinker.` prefix on per-block tensor too
            (
                "thinker.model.layers.5.self_attn.q_norm.weight",
                "blk.5.attn_q_norm.weight",
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

    /// Sibling — vision-side tensors (visual tower + DeepStack merger)
    /// MUST surface as `None` so the caller can route them to the
    /// mmproj mapper (or drop for text-only convert). Per
    /// [[feedback-no-loop-suppression-2026-05-17]]: never silently
    /// skip — the caller decides.
    ///
    /// The DeepStack names tested here mirror the vit-side test fixtures
    /// at `src/models/vit/convert.rs::wedge4f_qwen3vl_deepstack_*`.
    #[test]
    fn qwen3vl_text_rejects_vision_and_deepstack_tensors() {
        // Visual tower
        assert_eq!(
            map_tensor_name("visual.patch_embed.proj.weight"),
            None,
            "vision-side patch_embed lives in mmproj, not text decoder"
        );
        assert_eq!(
            map_tensor_name("model.visual.blocks.5.attn.proj.weight"),
            None,
            "vision-side block tensor lives in mmproj"
        );
        // DeepStack merger heads (per
        // `Qwen3VLVisionModel.modify_tensors` at qwen3vl.py:84-105).
        // Both bare and `model.`-prefixed forms must reject.
        assert_eq!(
            map_tensor_name("visual.deepstack_merger_list.0.norm.weight"),
            None,
            "deepstack merger norm is mmproj-side"
        );
        assert_eq!(
            map_tensor_name("visual.deepstack_merger_list.1.linear_fc1.weight"),
            None,
            "deepstack merger fc1 is mmproj-side"
        );
        assert_eq!(
            map_tensor_name("visual.deepstack_merger_list.2.linear_fc2.bias"),
            None,
            "deepstack merger fc2 is mmproj-side"
        );
        assert_eq!(
            map_tensor_name("model.visual.deepstack_merger_list.0.norm.weight"),
            None,
            "deepstack with model. prefix is mmproj-side"
        );
        // MTP head (Qwen3-VL multi-token-prediction; out of v1 scope)
        assert_eq!(map_tensor_name("mtp.embed_tokens.weight"), None);
        // Audio tower (Qwen3-Omni)
        assert_eq!(map_tensor_name("audio_tower.encoder.layer.0.conv1.weight"), None);
    }

    /// Sibling — unknown / malformed names must surface as `None`.
    #[test]
    fn qwen3vl_text_rejects_unknown_kinds() {
        // Unknown global.
        assert_eq!(map_tensor_name("model.unknown.weight"), None);
        // Wrong prefix.
        assert_eq!(map_tensor_name("transformer.layers.0.attn.weight"), None);
        // Qwen3 has no biases on linear projections.
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
        // Unknown per-block suffix.
        assert_eq!(map_tensor_name("model.layers.0.unknown.weight"), None);
        // MoE expert (Qwen3-VL MoE = Qwen3VLMoeForConditionalGeneration —
        // distinct arch QWEN3VLMOE; out of THIS mapper's v1 scope).
        assert_eq!(
            map_tensor_name("model.layers.0.mlp.experts.0.gate_proj.weight"),
            None
        );
        // Mid-name `thinker.` is corrupt (only leading-prefix strip is
        // canonical) — must NOT silently map.
        assert_eq!(
            map_tensor_name("model.thinker.layers.0.input_layernorm.weight"),
            None
        );
    }

    /// Acceptance test 2 — feed a representative Qwen3-VL-2B-Instruct
    /// config.json shape and verify all expected KV pairs come back
    /// with the right types + values, including the deepstack-count
    /// and M-RoPE dimension-sections.
    #[test]
    fn qwen3vl_text_metadata_built_from_config() {
        let cfg = json!({
            "_name_or_path": "Qwen/Qwen3-VL-2B-Instruct",
            "hidden_size": 2048,
            "num_hidden_layers": 28,
            "intermediate_size": 6144,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "max_position_embeddings": 128000,
            "rms_norm_eps": 1.0e-6,
            "rope_theta": 5_000_000.0,
            "rope_scaling": {
                "rope_type": "mrope",
                "mrope_section": [24, 20, 20]  // 3 entries → padded to 4
            },
            "vision_config": {
                "deepstack_visual_indexes": [5, 11, 17]
            },
        });

        let kv = build_metadata(&cfg, 17 /* MostlyQ5_K_M */);

        // 16 KV pairs when head_dim + mrope_section + deepstack are all present:
        //   architecture, name, context_length, embedding_length, block_count,
        //   feed_forward_length, head_count, head_count_kv, key_length,
        //   value_length, rms_eps, rope_freq_base, rope.dimension_sections,
        //   n_deepstack_layers, file_type  → 15? Let's count by index:
        //   0 general.architecture, 1 general.name, 2 context, 3 embedding,
        //   4 block, 5 ffn, 6 head_count, 7 head_count_kv, 8 key_length,
        //   9 value_length, 10 rms_eps, 11 rope_freq, 12 mrope_sections,
        //   13 n_deepstack, 14 file_type  → 15 entries.
        assert_eq!(
            kv.len(),
            15,
            "Qwen3VL emits 15 KV pairs when head_dim, mrope_section, and \
             deepstack are all present"
        );
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();

        assert_eq!(
            by_key["general.architecture"],
            MetaValue::String("qwen3vl".into())
        );
        assert_eq!(
            by_key["general.name"],
            MetaValue::String("Qwen/Qwen3-VL-2B-Instruct".into())
        );
        assert_eq!(by_key["qwen3vl.context_length"], MetaValue::U32(128000));
        assert_eq!(by_key["qwen3vl.embedding_length"], MetaValue::U32(2048));
        assert_eq!(by_key["qwen3vl.block_count"], MetaValue::U32(28));
        assert_eq!(by_key["qwen3vl.feed_forward_length"], MetaValue::U32(6144));
        assert_eq!(by_key["qwen3vl.attention.head_count"], MetaValue::U32(16));
        assert_eq!(by_key["qwen3vl.attention.head_count_kv"], MetaValue::U32(8));
        assert_eq!(by_key["qwen3vl.attention.key_length"], MetaValue::U32(128));
        assert_eq!(by_key["qwen3vl.attention.value_length"], MetaValue::U32(128));
        assert_eq!(
            by_key["qwen3vl.attention.layer_norm_rms_epsilon"],
            MetaValue::F32(1.0e-6)
        );
        assert_eq!(
            by_key["qwen3vl.rope.freq_base"],
            MetaValue::F32(5_000_000.0)
        );
        // M-RoPE: 3 entries [24, 20, 20] padded with trailing 0 → 4 entries.
        assert_eq!(
            by_key["qwen3vl.rope.dimension_sections"],
            MetaValue::ArrayU32(vec![24, 20, 20, 0])
        );
        assert_eq!(
            by_key["qwen3vl.n_deepstack_layers"],
            MetaValue::U32(3),
            "deepstack count = len(deepstack_visual_indexes)"
        );
        assert_eq!(by_key["general.file_type"], MetaValue::U32(17));
    }

    /// Sibling — verify the optional-key defaults trigger correctly:
    ///   - `_name_or_path` omitted → "model"
    ///   - `num_key_value_heads` omitted → n_head
    ///   - `head_dim` omitted → key_length / value_length KVs are NOT
    ///     emitted (matches `base.py:1216` conditional)
    ///   - `rope_theta` omitted → 1_000_000.0 (Qwen3-VL default)
    ///   - `rope_scaling.mrope_section` omitted → no
    ///     `qwen3vl.rope.dimension_sections` KV
    ///   - no `vision_config` → `n_deepstack_layers = 0`
    #[test]
    fn qwen3vl_text_metadata_optional_key_defaults() {
        let cfg = json!({
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            // num_key_value_heads omitted → defaults to num_attention_heads
            // head_dim omitted → key_length/value_length NOT emitted
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1.0e-6,
            // rope_theta omitted → defaults to 1_000_000.0 (Qwen3-VL)
            // rope_scaling.mrope_section omitted → no mrope_section KV
            // vision_config omitted → n_deepstack_layers = 0
            // _name_or_path omitted → defaults to "model"
        });
        let kv = build_metadata(&cfg, 0);
        // With head_dim absent and mrope absent: 15 - 2 (key+value length) - 1
        // (mrope_sections) = 12 entries.
        assert_eq!(kv.len(), 12, "12 KVs when head_dim + mrope absent");
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();

        assert_eq!(
            by_key["general.architecture"],
            MetaValue::String("qwen3vl".into())
        );
        assert_eq!(
            by_key["general.name"],
            MetaValue::String("model".into()),
            "name defaults to 'model' when _name_or_path absent"
        );
        assert_eq!(
            by_key["qwen3vl.attention.head_count_kv"],
            MetaValue::U32(4),
            "num_key_value_heads defaults to num_attention_heads"
        );
        assert!(
            !by_key.contains_key("qwen3vl.attention.key_length"),
            "key_length omitted when head_dim absent"
        );
        assert!(
            !by_key.contains_key("qwen3vl.attention.value_length"),
            "value_length omitted when head_dim absent"
        );
        assert_eq!(
            by_key["qwen3vl.rope.freq_base"],
            MetaValue::F32(1_000_000.0),
            "rope_theta defaults to 1_000_000.0 (Qwen3-VL convention)"
        );
        assert!(
            !by_key.contains_key("qwen3vl.rope.dimension_sections"),
            "mrope sections omitted when rope_scaling.mrope_section absent"
        );
        assert_eq!(
            by_key["qwen3vl.n_deepstack_layers"],
            MetaValue::U32(0),
            "deepstack count defaults to 0 when no vision_config"
        );
        assert_eq!(by_key["general.file_type"], MetaValue::U32(0));
    }

    /// Sibling — Qwen3-VL Omni / multimodal shape stores vision config
    /// under `thinker_config.vision_config` instead of bare
    /// `vision_config`. The deepstack count must read through the
    /// thinker_config wrapper. Mirrors
    /// `Qwen3VLTextModel.set_gguf_parameters` at qwen3vl.py:254-258.
    #[test]
    fn qwen3vl_text_metadata_thinker_config_deepstack_count() {
        let cfg = json!({
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1.0e-6,
            "thinker_config": {
                "vision_config": {
                    "deepstack_visual_indexes": [3, 7, 11, 15, 19]
                }
            },
        });
        let kv = build_metadata(&cfg, 0);
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        assert_eq!(
            by_key["qwen3vl.n_deepstack_layers"],
            MetaValue::U32(5),
            "thinker_config.vision_config.deepstack_visual_indexes \
             must be reachable; len = 5"
        );
    }

    /// Sibling — `file_type` round-trips as a plain u32 in
    /// `general.file_type` across the full LlamaFtype enum range.
    #[test]
    fn qwen3vl_text_metadata_ftype_round_trips() {
        let cfg = json!({
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 4,
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
