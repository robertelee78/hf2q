//! Qwen-3.5 / Qwen-3.6 MoE-A3B HF→GGUF tensor-name + metadata mapper.
//!
//! Port of `/opt/llama.cpp/conversion/qwen.py::Qwen3MoeModel`
//! (`@ModelBase.register("Qwen3MoeForCausalLM")`, `model_arch =
//! gguf.MODEL_ARCH.QWEN3MOE`). Qwen3MoE inherits its tensor-mapping
//! logic from `Qwen2MoeModel.modify_tensors` (no overrides except
//! InternS1 vocab handling, which is out of v1 scope here), so this
//! mapper transitively follows the Qwen2MoE expert-fusion contract.
//!
//! Per ADR-033 §P0 "Per-arch convert-side mapping": this is the
//! convert-side tensor-name + KV mapper for `LLM_ARCH_QWEN3MOE`. The
//! inference-side mapping (Qwen3MoE *is* a runtime target in hf2q's
//! Qwen 3.6 production path) is handled separately by mlx-native's
//! qwen3moe loader.
//!
//! Per [[feedback-no-backwards-compat-2026-05-18]]: no fallback, no
//! aliasing — every HF name we recognize maps to exactly one outcome
//! (`Direct` 1:1 rename, `ExpertGroup` accumulation, or `Drop` for
//! known-discardable tensors); every other name returns `None`. Per
//! [[feedback-no-loop-suppression-2026-05-17]]: callers MUST NOT
//! silently skip a `None` — propagate as a typed error.
//!
//! # Distinctive Qwen-3.5 MoE quirks vs Llama-3 / Gemma-4
//!
//! 1. **MoE expert FUSION.** Llama-3 / Gemma-4 (dense) have one
//!    `mlp.gate_proj` / `mlp.up_proj` / `mlp.down_proj` per layer.
//!    Qwen3MoE has `n_experts` (commonly 128) copies of each per
//!    layer, named `model.layers.<L>.mlp.experts.<E>.{gate,up,down}_proj.weight`.
//!    The GGUF format fuses these into a SINGLE 3-D tensor per layer +
//!    kind: `blk.<L>.ffn_{gate,up,down}_exps.weight` with shape
//!    `[hidden, ffn_exp, n_experts]` for gate/up and `[ffn_exp, hidden,
//!    n_experts]` for down (GGUF innermost-first order; see
//!    `qwen.py::Qwen2MoeModel.modify_tensors:90-93` for the
//!    PyTorch↔GGUF axis derivation). Fusion is `torch.stack(datas,
//!    dim=0)` in PyTorch axis order, which becomes a leading
//!    `n_experts` axis in HF shape (`[n_experts, ffn, hidden]` for
//!    gate/up), then reversed by GGUF to put `n_experts` at the
//!    outermost slot. The mapper itself does NOT execute the
//!    stack — it returns an [`ExpertGroup`] discriminant that the
//!    caller (orchestrator / a thin Qwen35Moe-specific staging driver)
//!    accumulates and emits as a single 3-D tensor once all
//!    `n_experts` experts of a given layer + kind have been seen.
//!
//! 2. **Router gate vs per-expert gate.** Qwen3MoE has TWO completely
//!    different tensors that both contain the word "gate":
//!      - `model.layers.<L>.mlp.gate.weight` — the ROUTER projection
//!        (`[n_experts, hidden]`), maps to `blk.<L>.ffn_gate_inp.weight`.
//!      - `model.layers.<L>.mlp.experts.<E>.gate_proj.weight` — the
//!        SwiGLU per-expert gate projection, maps via fusion to
//!        `blk.<L>.ffn_gate_exps.weight`.
//!    The mapper disambiguates by looking for `.experts.<E>.` AFTER
//!    `mlp.` — if absent, it's the router gate; if present, it's the
//!    per-expert gate that needs fusion. (Llama-3 and Gemma-4 have no
//!    router; this disambiguation is a Qwen-MoE-specific concern.)
//!
//! 3. **Per-head Q/K LayerNorms.** Like Gemma-4 (but unlike Llama-3),
//!    Qwen3 has per-head Q/K norms: `self_attn.q_norm.weight` /
//!    `self_attn.k_norm.weight` → `attn_q_norm` / `attn_k_norm`.
//!    Inherited from `Qwen3Model` (the dense parent of `Qwen3MoeModel`).
//!
//! 4. **No shared experts.** Qwen2MoE had shared-expert tensors
//!    (`shared_expert.{gate,up,down}_proj`, `shared_expert_gate`).
//!    `MODEL_ARCH.QWEN3MOE` in `gguf-py/gguf/constants.py:2012-2028`
//!    does NOT include `FFN_*_SHEXP` / `FFN_GATE_INP_SHEXP` — Qwen3MoE
//!    dropped shared experts. If a checkpoint contains
//!    `mlp.shared_expert.*` tensors, this mapper returns `None` (the
//!    caller surfaces it; do not silently drop). The Qwen3Next variant
//!    re-adds shared experts but that is a separate arch.
//!
//! 5. **No `lm_head` tie by default.** Qwen3MoE production checkpoints
//!    DO ship `lm_head.weight` (no tied embeddings, unlike Gemma-4).
//!    The `lm_head.weight` → `output.weight` map is present.
//!
//! 6. **`feed_forward_length` vs `expert_feed_forward_length`.** Qwen3MoE
//!    has TWO ffn dim KVs: `qwen3moe.feed_forward_length` (the dense
//!    fallback — used by some loaders for shared-expert sizing; on
//!    Qwen3MoE without shared experts this is the per-expert dim) and
//!    `qwen3moe.expert_feed_forward_length` (per-expert). Both come
//!    from `config["moe_intermediate_size"]` per the mission spec —
//!    NOT from `intermediate_size`, which on Qwen3MoE is unused / set
//!    to a dense-equivalent number for HF compat. Mirrors
//!    `Qwen2MoeModel.set_gguf_parameters:78-80`.

use crate::backends::gguf::types::MetaValue;

// ---------------------------------------------------------------------------
// Public API: MappedTensor + ExpertKind
// ---------------------------------------------------------------------------
//
// These types live in this module today because Qwen35MoE is the first
// arch in the convert pipeline that needs expert fusion. When MiniMaxM2
// (also MoE, ADR-033 §P4 convert matrix row) is ported, the cleanest
// move is to hoist these into `crate::convert::arch::mod` (or a sibling
// `arch::mapped` submodule) and have both per-arch mappers re-export.
// At that point Llama3 / Gemma4 / Bert / NomicBert do NOT need to
// change — their `map_tensor_name -> Option<String>` signature is
// strictly more constrained than `Option<MappedTensor>` and the
// orchestrator dispatches per arch anyway.

/// What the convert orchestrator should do with one HF tensor name
/// once the per-arch mapper has classified it.
///
/// Per the no-fallback contract, the mapper MUST return `Some(...)` for
/// every name it recognizes and `None` for every name it does not. The
/// `Drop` variant is reserved for tensors that the upstream Python
/// converter *explicitly* discards (e.g. precomputed `inv_freq` rope
/// buffers, MTP head residuals). Unknown / unmapped names are NOT
/// `Drop` — they are `None`, and the caller is expected to error out
/// per [[feedback-no-loop-suppression-2026-05-17]].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MappedTensor {
    /// One HF tensor → one GGUF tensor, 1:1 rename.
    Direct(String),

    /// One HF tensor is one slice of a fused GGUF expert tensor. The
    /// caller is expected to accumulate every expert of a given
    /// `(layer, kind)` group and emit a single 3-D GGUF tensor named
    /// `gguf_name` once `expert_index` covers `[0, n_experts)`.
    ///
    /// Fields:
    ///  - `gguf_name`: the fused GGUF tensor name
    ///    (e.g. `blk.7.ffn_gate_exps.weight`).
    ///  - `layer`: the block index (`<L>` from
    ///    `model.layers.<L>.mlp.experts.<E>.<kind>_proj.weight`).
    ///  - `expert_index`: `<E>` — the per-expert axis index this slice
    ///    will occupy in the fused tensor.
    ///  - `kind`: which of the three SwiGLU projections this slice is.
    ExpertGroup {
        gguf_name: String,
        layer: usize,
        expert_index: usize,
        kind: ExpertKind,
    },

    /// Known-discardable HF tensor — the upstream Python converter
    /// explicitly skips this name. Reserved for cases like cached rope
    /// inv-freq buffers (`model.layers.<L>.self_attn.rotary_emb.inv_freq`)
    /// that are recomputed at load time. The Qwen3MoE upstream does
    /// not currently emit any such tensor in production checkpoints,
    /// so this variant is unused by `map_tensor_name` today; it
    /// remains in the enum so future Qwen MoE variants (and the eventual
    /// MiniMaxM2 port) can express explicit drops without re-introducing
    /// silent skips.
    Drop,
}

/// Which SwiGLU projection a [`MappedTensor::ExpertGroup`] slice
/// belongs to. Determines both the GGUF tensor name suffix
/// (`ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`) and the
/// PyTorch→GGUF axis order the caller must apply during fusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertKind {
    /// SwiGLU gate projection. PyTorch per-expert shape `[ffn_exp,
    /// hidden]`. Fused GGUF shape `[hidden, ffn_exp, n_experts]`.
    Gate,
    /// SwiGLU up projection. PyTorch per-expert shape `[ffn_exp,
    /// hidden]`. Fused GGUF shape `[hidden, ffn_exp, n_experts]`.
    Up,
    /// SwiGLU down projection. PyTorch per-expert shape `[hidden,
    /// ffn_exp]`. Fused GGUF shape `[ffn_exp, hidden, n_experts]`.
    Down,
}

/// Translate one HuggingFace tensor name (as seen in `model.safetensors`)
/// to its [`MappedTensor`] outcome for Qwen3MoE. Returns `None` if
/// `hf_name` is not one of the Qwen3MoE text-decoder weight kinds.
///
/// Qwen3MoE weight kinds:
///
/// | HF name                                                            | GGUF name / action                                |
/// |--------------------------------------------------------------------|---------------------------------------------------|
/// | `model.embed_tokens.weight`                                        | `Direct("token_embd.weight")`                     |
/// | `model.norm.weight`                                                | `Direct("output_norm.weight")`                    |
/// | `lm_head.weight`                                                   | `Direct("output.weight")`                         |
/// | `model.layers.<L>.input_layernorm.weight`                          | `Direct("blk.<L>.attn_norm.weight")`              |
/// | `model.layers.<L>.post_attention_layernorm.weight`                 | `Direct("blk.<L>.ffn_norm.weight")`               |
/// | `model.layers.<L>.self_attn.q_proj.weight`                         | `Direct("blk.<L>.attn_q.weight")`                 |
/// | `model.layers.<L>.self_attn.k_proj.weight`                         | `Direct("blk.<L>.attn_k.weight")`                 |
/// | `model.layers.<L>.self_attn.v_proj.weight`                         | `Direct("blk.<L>.attn_v.weight")`                 |
/// | `model.layers.<L>.self_attn.o_proj.weight`                         | `Direct("blk.<L>.attn_output.weight")`            |
/// | `model.layers.<L>.self_attn.q_norm.weight`                         | `Direct("blk.<L>.attn_q_norm.weight")`            |
/// | `model.layers.<L>.self_attn.k_norm.weight`                         | `Direct("blk.<L>.attn_k_norm.weight")`            |
/// | `model.layers.<L>.mlp.gate.weight` *(router)*                      | `Direct("blk.<L>.ffn_gate_inp.weight")`           |
/// | `model.layers.<L>.mlp.experts.<E>.gate_proj.weight`                | `ExpertGroup{ blk.<L>.ffn_gate_exps.weight, <E>, Gate }` |
/// | `model.layers.<L>.mlp.experts.<E>.up_proj.weight`                  | `ExpertGroup{ blk.<L>.ffn_up_exps.weight,   <E>, Up   }` |
/// | `model.layers.<L>.mlp.experts.<E>.down_proj.weight`                | `ExpertGroup{ blk.<L>.ffn_down_exps.weight, <E>, Down }` |
///
/// Names returning `None` (caller is expected to error out — never
/// silently skip per [[feedback-no-loop-suppression-2026-05-17]]):
///   - Qwen2MoE shared-expert tensors (`mlp.shared_expert.*`,
///     `mlp.shared_expert_gate.weight`) — Qwen3MoE dropped shared experts.
///   - Linear biases (`q_proj.bias`, etc.) — Qwen3MoE has none.
///   - Per-block layer prefixes with malformed integer indices
///     (leading zeros, signs).
pub fn map_tensor_name(hf_name: &str) -> Option<MappedTensor> {
    // ---- Globals ----------------------------------------------------------
    match hf_name {
        "model.embed_tokens.weight" => {
            return Some(MappedTensor::Direct("token_embd.weight".to_string()));
        }
        "model.norm.weight" => {
            return Some(MappedTensor::Direct("output_norm.weight".to_string()));
        }
        "lm_head.weight" => {
            return Some(MappedTensor::Direct("output.weight".to_string()));
        }
        _ => {}
    }

    // ---- Per-block: `model.layers.<L>.<rest>` ----------------------------
    let stripped = hf_name.strip_prefix("model.layers.")?;
    let dot = stripped.find('.')?;
    let (layer_str, rest_with_dot) = stripped.split_at(dot);
    // Parse layer index — strict (no leading zeros / signs). Mirrors
    // the Llama-3 / Gemma-4 mapper policy.
    let layer: usize = layer_str.parse().ok()?;
    if layer.to_string() != layer_str {
        return None;
    }
    let rest = &rest_with_dot[1..]; // skip the dot

    // ---- Dense-side per-block tensors (no fusion) -----------------------
    let direct_suffix = match rest {
        // Norms
        "input_layernorm.weight" => Some("attn_norm.weight"),
        "post_attention_layernorm.weight" => Some("ffn_norm.weight"),

        // Attention projections
        "self_attn.q_proj.weight" => Some("attn_q.weight"),
        "self_attn.k_proj.weight" => Some("attn_k.weight"),
        "self_attn.v_proj.weight" => Some("attn_v.weight"),
        "self_attn.o_proj.weight" => Some("attn_output.weight"),

        // Per-head Q/K norms (Qwen3 quirk inherited from Qwen3Model)
        "self_attn.q_norm.weight" => Some("attn_q_norm.weight"),
        "self_attn.k_norm.weight" => Some("attn_k_norm.weight"),

        // Router gate (NOT per-expert gate — disambiguated by absence
        // of `.experts.<E>.`). Single 2-D tensor `[n_experts, hidden]`.
        "mlp.gate.weight" => Some("ffn_gate_inp.weight"),

        _ => None,
    };
    if let Some(suffix) = direct_suffix {
        return Some(MappedTensor::Direct(format!("blk.{layer}.{suffix}")));
    }

    // ---- Per-expert FFN tensors (need fusion) ---------------------------
    // Pattern: `mlp.experts.<E>.<kind>_proj.weight`.
    let expert_rest = rest.strip_prefix("mlp.experts.")?;
    let dot2 = expert_rest.find('.')?;
    let (expert_str, kind_with_dot) = expert_rest.split_at(dot2);
    let expert_index: usize = expert_str.parse().ok()?;
    if expert_index.to_string() != expert_str {
        return None;
    }
    let kind_tail = &kind_with_dot[1..];

    let (kind, gguf_suffix) = match kind_tail {
        "gate_proj.weight" => (ExpertKind::Gate, "ffn_gate_exps.weight"),
        "up_proj.weight" => (ExpertKind::Up, "ffn_up_exps.weight"),
        "down_proj.weight" => (ExpertKind::Down, "ffn_down_exps.weight"),
        _ => return None,
    };

    Some(MappedTensor::ExpertGroup {
        gguf_name: format!("blk.{layer}.{gguf_suffix}"),
        layer,
        expert_index,
        kind,
    })
}

/// Build the GGUF metadata KV pairs for a Qwen3MoE model from its HF
/// `config.json`. Port of `conversion/qwen.py::Qwen2MoeModel::set_gguf_parameters`
/// (inherited by `Qwen3MoeModel`) + the `TextModel::set_gguf_parameters`
/// base it `super()`s into (`/opt/llama.cpp/conversion/base.py:1111-1221`),
/// restricted to the keys every Qwen3MoE production release carries.
///
/// `general.architecture` = `"qwen3moe"`. Prefix is `qwen3moe.*`.
///
/// Required HF keys (mandatory; missing → caller-side panic from `[]`
/// indexing — mirrors the Llama3 / Gemma4 mapper policy):
///   - `hidden_size`
///   - `num_hidden_layers`
///   - `num_attention_heads`
///   - `num_key_value_heads`  (Qwen3MoE always GQA; no MHA fallback in HF)
///   - `max_position_embeddings`
///   - `rms_norm_eps`
///   - `moe_intermediate_size`  (per-expert ffn dim — NOT
///     `intermediate_size`, which on Qwen3MoE is the unused
///     dense-equivalent dim. Mirrors `Qwen2MoeModel.set_gguf_parameters`.)
///   - `num_experts`
///   - `num_experts_per_tok`
///
/// Optional HF keys (defaulted):
///   - `rope_theta` — defaults to `10000.0`. Production Qwen3MoE
///     uses `1_000_000.0`, but the field IS optional in the
///     HF schema and `base.py` only emits it when present; the
///     mapper supplies the schema default to keep the loader's
///     KV reads non-optional.
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
    let n_head = config["num_attention_heads"]
        .as_u64()
        .expect("config.json missing required key `num_attention_heads`") as u32;
    let n_head_kv = config["num_key_value_heads"]
        .as_u64()
        .expect("config.json missing required key `num_key_value_heads`") as u32;
    let ctx_len = config["max_position_embeddings"]
        .as_u64()
        .expect("config.json missing required key `max_position_embeddings`") as u32;
    let rms_eps = config["rms_norm_eps"]
        .as_f64()
        .expect("config.json missing required key `rms_norm_eps`") as f32;
    let moe_ffn = config["moe_intermediate_size"]
        .as_u64()
        .expect("config.json missing required key `moe_intermediate_size`") as u32;
    let n_experts = config["num_experts"]
        .as_u64()
        .expect("config.json missing required key `num_experts`") as u32;
    let n_experts_used = config["num_experts_per_tok"]
        .as_u64()
        .expect("config.json missing required key `num_experts_per_tok`") as u32;

    // Optional with defaults.
    let rope_theta = config
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0) as f32;

    vec![
        (
            "general.architecture".into(),
            MetaValue::String("qwen3moe".into()),
        ),
        ("general.name".into(), MetaValue::String(name)),
        ("qwen3moe.context_length".into(), MetaValue::U32(ctx_len)),
        (
            "qwen3moe.embedding_length".into(),
            MetaValue::U32(hidden_size),
        ),
        ("qwen3moe.block_count".into(), MetaValue::U32(n_layers)),
        // Per the mission spec: `feed_forward_length` is the per-expert
        // ffn dim (`moe_intermediate_size`), NOT the unused `intermediate_size`.
        (
            "qwen3moe.feed_forward_length".into(),
            MetaValue::U32(moe_ffn),
        ),
        (
            "qwen3moe.attention.head_count".into(),
            MetaValue::U32(n_head),
        ),
        (
            "qwen3moe.attention.head_count_kv".into(),
            MetaValue::U32(n_head_kv),
        ),
        (
            "qwen3moe.attention.layer_norm_rms_epsilon".into(),
            MetaValue::F32(rms_eps),
        ),
        (
            "qwen3moe.rope.freq_base".into(),
            MetaValue::F32(rope_theta),
        ),
        ("qwen3moe.expert_count".into(), MetaValue::U32(n_experts)),
        (
            "qwen3moe.expert_used_count".into(),
            MetaValue::U32(n_experts_used),
        ),
        (
            "qwen3moe.expert_feed_forward_length".into(),
            MetaValue::U32(moe_ffn),
        ),
        ("general.file_type".into(), MetaValue::U32(file_type)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Test 1 — Dense-side per-block + global tensors map via `Direct`.
    /// Covers every 1:1 rename in the Qwen3MoE inventory.
    #[test]
    fn qwen35moe_direct_tensor_names_round_trip() {
        let cases: &[(&str, &str)] = &[
            // Globals
            ("model.embed_tokens.weight", "token_embd.weight"),
            ("model.norm.weight", "output_norm.weight"),
            ("lm_head.weight", "output.weight"),
            // Per-block norms (sample at L=0, L=23, L=47 — Qwen3MoE-30B-A3B
            // is 48 layers; this covers edge/mid/depth).
            (
                "model.layers.0.input_layernorm.weight",
                "blk.0.attn_norm.weight",
            ),
            (
                "model.layers.23.input_layernorm.weight",
                "blk.23.attn_norm.weight",
            ),
            (
                "model.layers.47.input_layernorm.weight",
                "blk.47.attn_norm.weight",
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "blk.0.ffn_norm.weight",
            ),
            // Attention projections
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
            // Per-head Q/K norms (Qwen3 quirk)
            (
                "model.layers.7.self_attn.q_norm.weight",
                "blk.7.attn_q_norm.weight",
            ),
            (
                "model.layers.7.self_attn.k_norm.weight",
                "blk.7.attn_k_norm.weight",
            ),
            // Router gate — NOT per-expert. Maps to ffn_gate_inp.
            ("model.layers.3.mlp.gate.weight", "blk.3.ffn_gate_inp.weight"),
        ];

        for &(hf, expected_gguf) in cases {
            let got = map_tensor_name(hf);
            assert_eq!(
                got,
                Some(MappedTensor::Direct(expected_gguf.to_string())),
                "map_tensor_name({hf:?}) = {got:?}, want Direct({expected_gguf:?})"
            );
        }
    }

    /// Test 2 — Per-expert tensors map via `ExpertGroup` with the
    /// correct `(layer, expert_index, kind, gguf_name)` tuple. Covers
    /// all three SwiGLU projections + a sample of expert indices that
    /// exercise multi-digit parsing.
    #[test]
    fn qwen35moe_expert_tensor_fusion_classified() {
        let cases: &[(&str, usize, usize, ExpertKind, &str)] = &[
            // gate, layer=0, expert=0
            (
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                0,
                0,
                ExpertKind::Gate,
                "blk.0.ffn_gate_exps.weight",
            ),
            // up, layer=12, expert=63
            (
                "model.layers.12.mlp.experts.63.up_proj.weight",
                12,
                63,
                ExpertKind::Up,
                "blk.12.ffn_up_exps.weight",
            ),
            // down, layer=47, expert=127 (max for 128-expert Qwen3-MoE-30B-A3B)
            (
                "model.layers.47.mlp.experts.127.down_proj.weight",
                47,
                127,
                ExpertKind::Down,
                "blk.47.ffn_down_exps.weight",
            ),
            // gate, layer=15, expert=1 (single-digit expert + double-digit layer)
            (
                "model.layers.15.mlp.experts.1.gate_proj.weight",
                15,
                1,
                ExpertKind::Gate,
                "blk.15.ffn_gate_exps.weight",
            ),
        ];

        for &(hf, want_layer, want_expert, want_kind, want_gguf) in cases {
            let got = map_tensor_name(hf);
            match got {
                Some(MappedTensor::ExpertGroup {
                    gguf_name,
                    layer,
                    expert_index,
                    kind,
                }) => {
                    assert_eq!(layer, want_layer, "layer for {hf:?}");
                    assert_eq!(expert_index, want_expert, "expert_index for {hf:?}");
                    assert_eq!(kind, want_kind, "kind for {hf:?}");
                    assert_eq!(gguf_name, want_gguf, "gguf_name for {hf:?}");
                }
                other => panic!("map_tensor_name({hf:?}) = {other:?}, want ExpertGroup"),
            }
        }
    }

    /// Test 3 — Router gate vs per-expert gate disambiguation. The
    /// mapper must NOT confuse `mlp.gate.weight` (router) with
    /// `mlp.experts.<E>.gate_proj.weight` (per-expert SwiGLU gate).
    #[test]
    fn qwen35moe_router_vs_expert_gate_disambiguated() {
        // Router gate — 2-D, single tensor per layer.
        assert_eq!(
            map_tensor_name("model.layers.5.mlp.gate.weight"),
            Some(MappedTensor::Direct("blk.5.ffn_gate_inp.weight".into())),
            "mlp.gate.weight must map to ffn_gate_inp (router), NOT ffn_gate_exps"
        );

        // Per-expert gate — one per expert; needs fusion.
        let expert = map_tensor_name("model.layers.5.mlp.experts.0.gate_proj.weight");
        assert!(
            matches!(
                expert,
                Some(MappedTensor::ExpertGroup {
                    kind: ExpertKind::Gate,
                    ..
                })
            ),
            "experts.<E>.gate_proj.weight must map to ExpertGroup(Gate), got {expert:?}"
        );

        // The two names must NOT produce the same GGUF tensor name —
        // the disambiguation is load-bearing for correctness.
        let router = match map_tensor_name("model.layers.5.mlp.gate.weight") {
            Some(MappedTensor::Direct(n)) => n,
            _ => panic!(),
        };
        let per_expert = match map_tensor_name("model.layers.5.mlp.experts.0.gate_proj.weight") {
            Some(MappedTensor::ExpertGroup { gguf_name, .. }) => gguf_name,
            _ => panic!(),
        };
        assert_ne!(
            router, per_expert,
            "router gate and per-expert gate MUST map to distinct GGUF tensors"
        );
    }

    /// Test 4 — Unknown / out-of-scope names surface as `None`. Per
    /// [[feedback-no-loop-suppression-2026-05-17]]: callers are
    /// expected to error on this, never silently skip.
    #[test]
    fn qwen35moe_tensor_name_rejects_unknown_kinds() {
        // Unknown global.
        assert_eq!(map_tensor_name("model.unknown.weight"), None);
        // Wrong prefix.
        assert_eq!(
            map_tensor_name("transformer.layers.0.attn.weight"),
            None
        );
        // Qwen3MoE has no biases on linear projections.
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
        // Empty expert index.
        assert_eq!(
            map_tensor_name("model.layers.0.mlp.experts..gate_proj.weight"),
            None
        );
        // Malformed expert index (leading zero).
        assert_eq!(
            map_tensor_name("model.layers.0.mlp.experts.00.gate_proj.weight"),
            None
        );
        // Unknown per-expert projection kind.
        assert_eq!(
            map_tensor_name("model.layers.0.mlp.experts.0.unknown_proj.weight"),
            None
        );
        // Qwen2MoE-only shared-expert tensors — Qwen3MoE dropped them,
        // so they MUST surface as None (not be silently mapped to
        // something else).
        assert_eq!(
            map_tensor_name("model.layers.0.mlp.shared_expert.gate_proj.weight"),
            None
        );
        assert_eq!(
            map_tensor_name("model.layers.0.mlp.shared_expert.up_proj.weight"),
            None
        );
        assert_eq!(
            map_tensor_name("model.layers.0.mlp.shared_expert.down_proj.weight"),
            None
        );
        assert_eq!(
            map_tensor_name("model.layers.0.mlp.shared_expert_gate.weight"),
            None
        );
        // Cached rope inv-freq (recomputed at load time; upstream
        // Python discards). We surface as None, NOT MappedTensor::Drop —
        // production Qwen3MoE checkpoints do not ship this tensor, so
        // returning Drop would be reinterpreting silence as policy.
        assert_eq!(
            map_tensor_name("model.layers.0.self_attn.rotary_emb.inv_freq"),
            None
        );
    }

    /// Test 5 — Metadata round-trip with a minimal hand-written config.
    /// All 14 required KV pairs come back with the right types + values.
    #[test]
    fn qwen35moe_metadata_built_from_config() {
        let cfg = json!({
            "_name_or_path": "Qwen/Qwen3-30B-A3B",
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "intermediate_size": 6144,
            "moe_intermediate_size": 768,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1.0e-6,
            "rope_theta": 1_000_000.0,
            "num_experts": 128,
            "num_experts_per_tok": 8,
        });

        let kv = build_metadata(&cfg, 17 /* MostlyQ5_K_M */);

        assert_eq!(kv.len(), 14, "Qwen3MoE emits 14 KV pairs at v1");
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();

        assert_eq!(
            by_key["general.architecture"],
            MetaValue::String("qwen3moe".into())
        );
        assert_eq!(
            by_key["general.name"],
            MetaValue::String("Qwen/Qwen3-30B-A3B".into())
        );
        assert_eq!(by_key["qwen3moe.context_length"], MetaValue::U32(32768));
        assert_eq!(by_key["qwen3moe.embedding_length"], MetaValue::U32(2048));
        assert_eq!(by_key["qwen3moe.block_count"], MetaValue::U32(48));
        // feed_forward_length comes from moe_intermediate_size, NOT
        // intermediate_size — explicit per mission spec.
        assert_eq!(
            by_key["qwen3moe.feed_forward_length"],
            MetaValue::U32(768),
            "feed_forward_length must come from moe_intermediate_size, not intermediate_size"
        );
        assert_eq!(by_key["qwen3moe.attention.head_count"], MetaValue::U32(32));
        assert_eq!(
            by_key["qwen3moe.attention.head_count_kv"],
            MetaValue::U32(4)
        );
        assert_eq!(
            by_key["qwen3moe.attention.layer_norm_rms_epsilon"],
            MetaValue::F32(1.0e-6)
        );
        assert_eq!(
            by_key["qwen3moe.rope.freq_base"],
            MetaValue::F32(1_000_000.0)
        );
        assert_eq!(by_key["qwen3moe.expert_count"], MetaValue::U32(128));
        assert_eq!(by_key["qwen3moe.expert_used_count"], MetaValue::U32(8));
        assert_eq!(
            by_key["qwen3moe.expert_feed_forward_length"],
            MetaValue::U32(768)
        );
        assert_eq!(by_key["general.file_type"], MetaValue::U32(17));
    }

    /// Test 6 — Optional-key defaults. With `rope_theta` and
    /// `_name_or_path` omitted, the mapper supplies the canonical
    /// defaults (`10000.0` and `"model"`).
    #[test]
    fn qwen35moe_metadata_optional_key_defaults() {
        let cfg = json!({
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "intermediate_size": 256,
            "moe_intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1.0e-6,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            // rope_theta omitted → defaults to 10000.0
            // _name_or_path omitted → defaults to "model"
        });
        let kv = build_metadata(&cfg, 0);
        assert_eq!(kv.len(), 14, "Qwen3MoE always emits 14 KV pairs");
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        assert_eq!(
            by_key["general.name"],
            MetaValue::String("model".into()),
            "name defaults to 'model' when _name_or_path absent"
        );
        assert_eq!(
            by_key["qwen3moe.rope.freq_base"],
            MetaValue::F32(10000.0),
            "rope_theta defaults to 10000.0"
        );
        assert_eq!(
            by_key["general.architecture"],
            MetaValue::String("qwen3moe".into())
        );
        assert_eq!(by_key["general.file_type"], MetaValue::U32(0));
    }
}
