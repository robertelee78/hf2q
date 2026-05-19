//! MiniMax-M2.7 HF→GGUF tensor-name + metadata mapper.
//!
//! Port of `/opt/llama.cpp/conversion/minimax.py::MiniMaxM2Model`
//! (`@ModelBase.register("MiniMaxM2ForCausalLM")`,
//! `model_arch = gguf.MODEL_ARCH.MINIMAXM2`) and its inherited
//! `TextModel::set_gguf_parameters` chain plus MiniMax-specific
//! overlay (`minimax.py:18-22`):
//!
//! ```python
//! self.gguf_writer.add_expert_feed_forward_length(self.find_hparam(["intermediate_size"]))
//! self.gguf_writer.add_rope_dimension_count(self.find_hparam(["rotary_dim"]))
//! ```
//!
//! MiniMax-M2.7 is an MoE model published in **FP8** source format
//! (`config.json::quantization_config.quant_method = "fp8"`). The
//! source-side load + dequant happens in
//! [`crate::convert::source_reader`] + [`crate::convert::source_dtype::fp8`];
//! by the time this mapper runs, every tensor is already F32. The
//! mapper itself is dtype-agnostic — it only re-names HF tensor paths
//! to GGUF tensor paths and emits the metadata KV pairs.
//!
//! Per ADR-033 §P0 "Per-arch convert-side mapping": convert-side mapper
//! for `LLM_ARCH_MINIMAXM2` (`gguf-py/gguf/constants.py:497`,
//! `:1012`). The GGUF architecture name is `"minimax-m2"`.
//!
//! Per [[feedback-no-backwards-compat-2026-05-18]]: every recognized
//! HF name maps to exactly one GGUF name; every other returns
//! `None` (caller surfaces as a typed error). Per
//! [[feedback-no-loop-suppression-2026-05-17]]: callers MUST NOT
//! silently skip a `None`.
//!
//! MiniMax-M2.7 architecture features (vs Llama-3):
//!
//! 1. **MoE FFN.** Each block has `num_local_experts` experts (typically
//!    32-256) with three weights per expert: `w1` (gate), `w3` (up),
//!    `w2` (down). HF stores them as separate per-expert tensors:
//!    `model.layers.<N>.block_sparse_moe.experts.<E>.w[1-3].weight`.
//!    The GGUF convention is to STACK them along a new expert axis,
//!    producing one tensor per (block, role) named
//!    `blk.<N>.ffn_{gate,up,down}_exps.weight` of shape
//!    `[n_experts, role_out, role_in]`. This mapper emits per-expert
//!    GGUF names ([`MappedTensor::ExpertWeight`]) and the orchestrator
//!    (or a downstream stacker) is responsible for the stack; this
//!    mirrors the Python `MiniMaxM2Model.modify_tensors` cache+stack
//!    pattern.
//!
//! 2. **Router.** Each block has one router gate
//!    `model.layers.<N>.block_sparse_moe.gate.weight` → `blk.<N>.ffn_gate_inp.weight`.
//!
//! 3. **Per-head Q/K LayerNorms** like Gemma-4 / Qwen3:
//!    `self_attn.q_norm.weight` / `self_attn.k_norm.weight` →
//!    `attn_q_norm.weight` / `attn_k_norm.weight`.
//!
//! 4. **Optional expert score bias.** Some MiniMax-M2.7 releases ship
//!    an expert-selection bias at
//!    `model.layers.<N>.block_sparse_moe.e_score_correction` →
//!    `blk.<N>.exp_probs_b.weight` (`MODEL_TENSOR.FFN_EXP_PROBS_B`,
//!    `gguf-py/gguf/tensor_mapping.py:464`).
//!
//! 5. **Rotary dim is explicit.** `rotary_dim` is required in
//!    `config.json` (the MiniMax convention; mirrors the Python
//!    `add_rope_dimension_count`).

use crate::backends::gguf::types::MetaValue;

// ============================================================================
// MappedTensor — MoE-aware mapper output
// ============================================================================

/// One mapped tensor. `Dense` is a 1:1 HF→GGUF name pair; `ExpertWeight`
/// carries the per-expert (block, expert_id, role) triple so the
/// downstream stacker can assemble the GGUF expert-stacked tensor.
/// `Router` is an alias for `Dense` with a tag — it lets a caller
/// special-case routers if needed (e.g. to avoid quantizing them) but
/// otherwise behaves identically.
///
/// **NOTE on cross-worker coordination.** This enum is the minimal
/// MoE-aware mapper output for MiniMax-M2.7. The parallel Qwen35Moe
/// worker is designing the same shape; once both land, the
/// downstream consolidator (P4+ convert orchestrator wiring) is
/// expected to merge them into a single canonical
/// `crate::convert::arch::MappedTensor` enum. The names + semantics
/// chosen here are intended to compose:
///
/// - `Dense { hf, gguf }` — both arches.
/// - `Router { hf, gguf }` — both arches; MiniMax `ffn_gate_inp`,
///   Qwen3moe `ffn_gate_inp`.
/// - `ExpertWeight { hf, layer, expert, role: ExpertRole, gguf_stacked }`
///   — both arches; `gguf_stacked` is the post-stack tensor name
///   (`blk.<N>.ffn_{gate,up,down}_exps.weight`) the stacker will
///   produce. Qwen35Moe's HF layout is the same `mlp.experts.<E>.{gate,up,down}_proj`
///   pattern (vs MiniMax's `block_sparse_moe.experts.<E>.w[1-3]`); the
///   `role` field abstracts that.
///
/// If the Qwen35Moe worker ships a different enum shape (e.g. flat
/// rather than tagged stack name), the consolidator can adapt — the
/// `role` + `expert` + `layer` triple is the load-bearing info.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MappedTensor {
    /// Standard 1:1 HF→GGUF tensor (embeddings, norms, attention
    /// projections, dense FFN, lm_head).
    Dense {
        hf: String,
        gguf: String,
    },
    /// Per-expert weight that will be stacked along an expert axis
    /// before being written to GGUF. The stacker concatenates all
    /// `n_experts` instances at the same `(layer, role)` into one
    /// tensor named `gguf_stacked`.
    ExpertWeight {
        /// Original HF tensor name (per-expert).
        hf: String,
        /// Decoder block index.
        layer: u32,
        /// Expert index within the block.
        expert: u32,
        /// Which of the three SwiGLU weights this is.
        role: ExpertRole,
        /// Post-stack canonical GGUF name. Identical across every
        /// expert at the same `(layer, role)` — the stacker uses this
        /// as the grouping key.
        gguf_stacked: String,
    },
    /// Router gate / expert-selection bias — emitted as a single
    /// tensor per block; semantically distinct from `Dense` so a
    /// caller can apply a different quant policy if needed.
    Router {
        hf: String,
        gguf: String,
    },
}

/// SwiGLU role for an expert weight. MiniMax follows the Mixtral
/// convention (`w1=gate, w2=down, w3=up`) on disk; the mapper
/// translates to GGUF's role-named tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertRole {
    Gate, // HF `w1` → GGUF `ffn_gate_exps`
    Up,   // HF `w3` → GGUF `ffn_up_exps`
    Down, // HF `w2` → GGUF `ffn_down_exps`
}

impl ExpertRole {
    /// Lowercase GGUF role suffix (without the `_exps.weight` tail).
    pub const fn gguf_prefix(self) -> &'static str {
        match self {
            ExpertRole::Gate => "ffn_gate",
            ExpertRole::Up => "ffn_up",
            ExpertRole::Down => "ffn_down",
        }
    }
}

// ============================================================================
// map_tensor_name — the canonical entry point
// ============================================================================

/// Translate one HuggingFace tensor name to its canonical GGUF mapping.
///
/// Returns `None` if `hf_name` is not one of the MiniMax-M2.7 weight
/// kinds. Per the no-loop-suppression rule, callers must surface
/// `None` as a typed error.
///
/// Inventory:
///
/// | HF name                                                                       | Output                                          |
/// |-------------------------------------------------------------------------------|-------------------------------------------------|
/// | `model.embed_tokens.weight`                                                   | `Dense` → `token_embd.weight`                   |
/// | `model.norm.weight`                                                           | `Dense` → `output_norm.weight`                  |
/// | `lm_head.weight`                                                              | `Dense` → `output.weight`                       |
/// | `model.layers.<N>.input_layernorm.weight`                                     | `Dense` → `blk.<N>.attn_norm.weight`            |
/// | `model.layers.<N>.post_attention_layernorm.weight`                            | `Dense` → `blk.<N>.ffn_norm.weight`             |
/// | `model.layers.<N>.self_attn.q_proj.weight`                                    | `Dense` → `blk.<N>.attn_q.weight`               |
/// | `model.layers.<N>.self_attn.k_proj.weight`                                    | `Dense` → `blk.<N>.attn_k.weight`               |
/// | `model.layers.<N>.self_attn.v_proj.weight`                                    | `Dense` → `blk.<N>.attn_v.weight`               |
/// | `model.layers.<N>.self_attn.o_proj.weight`                                    | `Dense` → `blk.<N>.attn_output.weight`          |
/// | `model.layers.<N>.self_attn.q_norm.weight`                                    | `Dense` → `blk.<N>.attn_q_norm.weight`          |
/// | `model.layers.<N>.self_attn.k_norm.weight`                                    | `Dense` → `blk.<N>.attn_k_norm.weight`          |
/// | `model.layers.<N>.block_sparse_moe.gate.weight`                               | `Router` → `blk.<N>.ffn_gate_inp.weight`        |
/// | `model.layers.<N>.block_sparse_moe.e_score_correction`                        | `Router` → `blk.<N>.exp_probs_b.weight`         |
/// | `model.layers.<N>.block_sparse_moe.experts.<E>.w1.weight`                     | `ExpertWeight` → `blk.<N>.ffn_gate_exps.weight` |
/// | `model.layers.<N>.block_sparse_moe.experts.<E>.w2.weight`                     | `ExpertWeight` → `blk.<N>.ffn_down_exps.weight` |
/// | `model.layers.<N>.block_sparse_moe.experts.<E>.w3.weight`                     | `ExpertWeight` → `blk.<N>.ffn_up_exps.weight`   |
pub fn map_tensor_name(hf_name: &str) -> Option<MappedTensor> {
    // ---- Globals ----------------------------------------------------------
    match hf_name {
        "model.embed_tokens.weight" => {
            return Some(MappedTensor::Dense {
                hf: hf_name.to_string(),
                gguf: "token_embd.weight".to_string(),
            })
        }
        "model.norm.weight" => {
            return Some(MappedTensor::Dense {
                hf: hf_name.to_string(),
                gguf: "output_norm.weight".to_string(),
            })
        }
        "lm_head.weight" => {
            return Some(MappedTensor::Dense {
                hf: hf_name.to_string(),
                gguf: "output.weight".to_string(),
            })
        }
        _ => {}
    }

    // ---- Per-block: `model.layers.<N>.<rest>` -----------------------------
    let stripped = hf_name.strip_prefix("model.layers.")?;
    let dot = stripped.find('.')?;
    let (layer_str, rest_with_dot) = stripped.split_at(dot);
    let layer: u32 = layer_str.parse().ok()?;
    if layer.to_string() != layer_str {
        return None; // reject leading zeros / signs
    }
    let rest = &rest_with_dot[1..]; // skip the dot

    // ---- Per-block dense kinds -------------------------------------------
    if let Some(suffix) = match_dense_suffix(rest) {
        return Some(MappedTensor::Dense {
            hf: hf_name.to_string(),
            gguf: format!("blk.{layer}.{suffix}"),
        });
    }

    // ---- Per-block router ------------------------------------------------
    match rest {
        "block_sparse_moe.gate.weight" => {
            return Some(MappedTensor::Router {
                hf: hf_name.to_string(),
                gguf: format!("blk.{layer}.ffn_gate_inp.weight"),
            });
        }
        "block_sparse_moe.e_score_correction" => {
            return Some(MappedTensor::Router {
                hf: hf_name.to_string(),
                gguf: format!("blk.{layer}.exp_probs_b.weight"),
            });
        }
        _ => {}
    }

    // ---- Per-block, per-expert MoE weights -------------------------------
    // Format: `block_sparse_moe.experts.<E>.w[1-3].weight`
    if let Some(rest2) = rest.strip_prefix("block_sparse_moe.experts.") {
        // Split into `<E>` + `.w[1-3].weight`.
        let dot2 = rest2.find('.')?;
        let (eid_str, after) = rest2.split_at(dot2);
        let expert: u32 = eid_str.parse().ok()?;
        if expert.to_string() != eid_str {
            return None;
        }
        let role = match after {
            ".w1.weight" => ExpertRole::Gate,
            ".w2.weight" => ExpertRole::Down,
            ".w3.weight" => ExpertRole::Up,
            _ => return None,
        };
        let gguf_stacked = format!("blk.{layer}.{}_exps.weight", role.gguf_prefix());
        return Some(MappedTensor::ExpertWeight {
            hf: hf_name.to_string(),
            layer,
            expert,
            role,
            gguf_stacked,
        });
    }

    None
}

/// Match the dense per-block suffixes (no MoE / router). Returns the
/// GGUF tensor-name suffix (without the `blk.<N>.` prefix) on hit.
fn match_dense_suffix(rest: &str) -> Option<&'static str> {
    Some(match rest {
        // Norms
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        // Attention projections
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        // Per-head Q/K norms
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        _ => return None,
    })
}

// ============================================================================
// build_metadata — config.json → GGUF KV pairs
// ============================================================================

/// Build the GGUF metadata KV pairs for a MiniMax-M2.7 model from its
/// HF `config.json`. Mirrors `MiniMaxM2Model.set_gguf_parameters` +
/// the inherited `TextModel::set_gguf_parameters` chain.
///
/// Required HF keys:
///   - `hidden_size`
///   - `num_hidden_layers`
///   - `num_attention_heads`
///   - `max_position_embeddings`
///   - `rms_norm_eps`
///   - `intermediate_size` (used for `expert_feed_forward_length` per
///     `minimax.py:21` AND as the model's `feed_forward_length`)
///   - `rotary_dim` (MiniMax-specific; emitted as
///     `minimax-m2.rope.dimension_count` per `minimax.py:22`)
///
/// Optional HF keys (defaulted):
///   - `num_key_value_heads` — defaults to `num_attention_heads`
///   - `rope_theta` — defaults to `10000.0`
///   - `num_experts` / `num_local_experts` — expert count; emitted as
///     `minimax-m2.expert_count`. Per `TextModel::set_gguf_parameters`
///     line 1194: the loader checks both names in order, taking the
///     first found.
///   - `num_experts_per_tok` — emitted as
///     `minimax-m2.expert_used_count`. (`base.py:1197`)
///   - `_name_or_path` — defaults to `"model"`
///
/// `file_type` is the chosen `LlamaFtype` as a `u32`.
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
    let ctx_len = config["max_position_embeddings"]
        .as_u64()
        .expect("config.json missing required key `max_position_embeddings`") as u32;
    let rms_eps = config["rms_norm_eps"]
        .as_f64()
        .expect("config.json missing required key `rms_norm_eps`") as f32;

    // `feed_forward_length` — MiniMax uses the same `intermediate_size`
    // for both the (theoretical) dense path AND each expert's FFN. The
    // Python writer emits it twice (`add_feed_forward_length` via base,
    // then `add_expert_feed_forward_length` overlay at minimax.py:21).
    let ffn_len = config["intermediate_size"]
        .as_u64()
        .expect("config.json missing required key `intermediate_size`") as u32;

    let rotary_dim = config["rotary_dim"]
        .as_u64()
        .expect("config.json missing required key `rotary_dim`") as u32;

    // ---- Optional with defaults ------------------------------------------
    let n_head_kv = config
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32)
        .unwrap_or(n_head);
    let rope_theta = config
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0) as f32;

    // Expert metadata. Per base.py:1194-1199, the loader tries
    // `num_local_experts` first, then `num_experts`. We mirror that
    // priority order. Missing field → key not emitted (the loader will
    // treat the model as dense).
    let n_experts = config
        .get("num_local_experts")
        .and_then(|v| v.as_u64())
        .or_else(|| config.get("num_experts").and_then(|v| v.as_u64()))
        .map(|x| x as u32);
    let n_experts_used = config
        .get("num_experts_per_tok")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32);

    let mut kv: Vec<(String, MetaValue)> = vec![
        (
            "general.architecture".into(),
            MetaValue::String("minimax-m2".into()),
        ),
        ("general.name".into(), MetaValue::String(name)),
        ("minimax-m2.context_length".into(), MetaValue::U32(ctx_len)),
        (
            "minimax-m2.embedding_length".into(),
            MetaValue::U32(hidden_size),
        ),
        ("minimax-m2.block_count".into(), MetaValue::U32(n_layers)),
        (
            "minimax-m2.feed_forward_length".into(),
            MetaValue::U32(ffn_len),
        ),
        (
            "minimax-m2.expert_feed_forward_length".into(),
            MetaValue::U32(ffn_len),
        ),
        (
            "minimax-m2.attention.head_count".into(),
            MetaValue::U32(n_head),
        ),
        (
            "minimax-m2.attention.head_count_kv".into(),
            MetaValue::U32(n_head_kv),
        ),
        (
            "minimax-m2.attention.layer_norm_rms_epsilon".into(),
            MetaValue::F32(rms_eps),
        ),
        (
            "minimax-m2.rope.freq_base".into(),
            MetaValue::F32(rope_theta),
        ),
        (
            "minimax-m2.rope.dimension_count".into(),
            MetaValue::U32(rotary_dim),
        ),
        ("general.file_type".into(), MetaValue::U32(file_type)),
    ];

    if let Some(n) = n_experts {
        kv.push(("minimax-m2.expert_count".into(), MetaValue::U32(n)));
    }
    if let Some(n) = n_experts_used {
        kv.push(("minimax-m2.expert_used_count".into(), MetaValue::U32(n)));
    }

    kv
}

// ============================================================================
// tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Acceptance test 1 — dense tensors round-trip with the right
    /// GGUF names. Sample at L=0, L=15, L=31.
    #[test]
    fn minimax_m2_dense_tensor_name_round_trip() {
        let cases: &[(&str, &str)] = &[
            // Globals
            ("model.embed_tokens.weight", "token_embd.weight"),
            ("model.norm.weight", "output_norm.weight"),
            ("lm_head.weight", "output.weight"),
            // Per-block — sample
            (
                "model.layers.0.input_layernorm.weight",
                "blk.0.attn_norm.weight",
            ),
            (
                "model.layers.15.post_attention_layernorm.weight",
                "blk.15.ffn_norm.weight",
            ),
            (
                "model.layers.31.self_attn.q_proj.weight",
                "blk.31.attn_q.weight",
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
                "model.layers.3.self_attn.q_norm.weight",
                "blk.3.attn_q_norm.weight",
            ),
            (
                "model.layers.3.self_attn.k_norm.weight",
                "blk.3.attn_k_norm.weight",
            ),
        ];

        for &(hf, expected_gguf) in cases {
            match map_tensor_name(hf) {
                Some(MappedTensor::Dense { hf: h, gguf }) => {
                    assert_eq!(h, hf);
                    assert_eq!(gguf, expected_gguf);
                }
                other => panic!("map_tensor_name({hf:?}) = {other:?}, want Dense({expected_gguf:?})"),
            }
        }
    }

    /// Acceptance test 2 — expert tensors return `ExpertWeight` with
    /// the right `(layer, expert, role)` triple + stacked-name.
    #[test]
    fn minimax_m2_expert_tensor_mapping() {
        let cases: &[(&str, u32, u32, ExpertRole, &str)] = &[
            (
                "model.layers.0.block_sparse_moe.experts.0.w1.weight",
                0,
                0,
                ExpertRole::Gate,
                "blk.0.ffn_gate_exps.weight",
            ),
            (
                "model.layers.0.block_sparse_moe.experts.0.w2.weight",
                0,
                0,
                ExpertRole::Down,
                "blk.0.ffn_down_exps.weight",
            ),
            (
                "model.layers.0.block_sparse_moe.experts.0.w3.weight",
                0,
                0,
                ExpertRole::Up,
                "blk.0.ffn_up_exps.weight",
            ),
            // Mid-block, mid-expert
            (
                "model.layers.15.block_sparse_moe.experts.7.w1.weight",
                15,
                7,
                ExpertRole::Gate,
                "blk.15.ffn_gate_exps.weight",
            ),
            // High-block, high-expert
            (
                "model.layers.31.block_sparse_moe.experts.255.w3.weight",
                31,
                255,
                ExpertRole::Up,
                "blk.31.ffn_up_exps.weight",
            ),
        ];

        for &(hf, exp_layer, exp_expert, exp_role, exp_stacked) in cases {
            match map_tensor_name(hf) {
                Some(MappedTensor::ExpertWeight {
                    hf: h,
                    layer,
                    expert,
                    role,
                    gguf_stacked,
                }) => {
                    assert_eq!(h, hf);
                    assert_eq!(layer, exp_layer);
                    assert_eq!(expert, exp_expert);
                    assert_eq!(role, exp_role);
                    assert_eq!(gguf_stacked, exp_stacked);
                }
                other => panic!(
                    "map_tensor_name({hf:?}) = {other:?}, want ExpertWeight({exp_stacked:?})"
                ),
            }
        }
    }

    /// Acceptance test 3 — router gate + e_score_correction return
    /// `Router` with the right GGUF name.
    #[test]
    fn minimax_m2_router_tensor_mapping() {
        match map_tensor_name("model.layers.0.block_sparse_moe.gate.weight") {
            Some(MappedTensor::Router { gguf, .. }) => {
                assert_eq!(gguf, "blk.0.ffn_gate_inp.weight");
            }
            other => panic!("router gate: got {other:?}"),
        }
        match map_tensor_name("model.layers.5.block_sparse_moe.e_score_correction") {
            Some(MappedTensor::Router { gguf, .. }) => {
                assert_eq!(gguf, "blk.5.exp_probs_b.weight");
            }
            other => panic!("e_score_correction: got {other:?}"),
        }
    }

    /// Sibling — unknown / malformed names return `None`. Per
    /// [[feedback-no-loop-suppression-2026-05-17]]: callers MUST NOT
    /// silently skip these.
    #[test]
    fn minimax_m2_tensor_name_rejects_unknown_kinds() {
        // Unknown global
        assert!(map_tensor_name("model.unknown.weight").is_none());
        // Wrong prefix
        assert!(map_tensor_name("transformer.layers.0.attn.weight").is_none());
        // Bias rejected (MiniMax-M2 has no biases on linear projections)
        assert!(map_tensor_name("model.layers.0.self_attn.q_proj.bias").is_none());
        // Malformed layer index
        assert!(map_tensor_name("model.layers.01.input_layernorm.weight").is_none());
        // Malformed expert id
        assert!(
            map_tensor_name("model.layers.0.block_sparse_moe.experts.01.w1.weight").is_none()
        );
        // Unknown expert role
        assert!(
            map_tensor_name("model.layers.0.block_sparse_moe.experts.0.w4.weight").is_none()
        );
        // Dense Mixtral-style FFN names — NOT used by MiniMax-M2 (it's
        // pure MoE, every block goes through the router).
        assert!(map_tensor_name("model.layers.0.mlp.gate_proj.weight").is_none());
        assert!(map_tensor_name("model.layers.0.mlp.up_proj.weight").is_none());
        assert!(map_tensor_name("model.layers.0.mlp.down_proj.weight").is_none());
        // Unknown per-block suffix
        assert!(map_tensor_name("model.layers.0.unknown.weight").is_none());
    }

    /// Acceptance test 4 — metadata builds from a hand-written
    /// config.json. Verifies count + types of every emitted KV pair.
    #[test]
    fn minimax_m2_metadata_built_from_config() {
        let cfg = json!({
            "_name_or_path": "MiniMaxAI/MiniMax-M2",
            "hidden_size": 6144,
            "num_hidden_layers": 80,
            "intermediate_size": 9216,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "max_position_embeddings": 196608,
            "rms_norm_eps": 1.0e-6,
            "rope_theta": 5_000_000.0,
            "rotary_dim": 64,
            "num_experts_per_tok": 8,
            "num_local_experts": 256,
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
        });

        let kv = build_metadata(&cfg, 17 /* MostlyQ5_K_M */);
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();

        // Always-emitted: 13 (architecture, name, ctx, embed, blocks,
        // feed_forward, expert_feed_forward, head_count, head_count_kv,
        // rms_eps, rope_freq_base, rope_dim, file_type).
        // Plus expert_count + expert_used_count (both present here) = 15.
        assert_eq!(kv.len(), 15);

        assert_eq!(
            by_key["general.architecture"],
            MetaValue::String("minimax-m2".into())
        );
        assert_eq!(
            by_key["general.name"],
            MetaValue::String("MiniMaxAI/MiniMax-M2".into())
        );
        assert_eq!(by_key["minimax-m2.context_length"], MetaValue::U32(196608));
        assert_eq!(by_key["minimax-m2.embedding_length"], MetaValue::U32(6144));
        assert_eq!(by_key["minimax-m2.block_count"], MetaValue::U32(80));
        assert_eq!(
            by_key["minimax-m2.feed_forward_length"],
            MetaValue::U32(9216)
        );
        assert_eq!(
            by_key["minimax-m2.expert_feed_forward_length"],
            MetaValue::U32(9216)
        );
        assert_eq!(by_key["minimax-m2.attention.head_count"], MetaValue::U32(64));
        assert_eq!(
            by_key["minimax-m2.attention.head_count_kv"],
            MetaValue::U32(8)
        );
        assert_eq!(
            by_key["minimax-m2.attention.layer_norm_rms_epsilon"],
            MetaValue::F32(1.0e-6)
        );
        assert_eq!(
            by_key["minimax-m2.rope.freq_base"],
            MetaValue::F32(5_000_000.0)
        );
        assert_eq!(
            by_key["minimax-m2.rope.dimension_count"],
            MetaValue::U32(64)
        );
        assert_eq!(by_key["minimax-m2.expert_count"], MetaValue::U32(256));
        assert_eq!(by_key["minimax-m2.expert_used_count"], MetaValue::U32(8));
        assert_eq!(by_key["general.file_type"], MetaValue::U32(17));
    }

    /// Sibling — verify optional-key defaults:
    /// - `num_experts` (alias) preferred when `num_local_experts` absent.
    /// - `num_key_value_heads` defaults to `num_attention_heads`.
    /// - `rope_theta` defaults to `10000.0`.
    /// - Without ANY expert count, the `expert_count` KV is omitted.
    #[test]
    fn minimax_m2_metadata_optional_key_defaults() {
        let cfg = json!({
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            // num_key_value_heads omitted → defaults to num_attention_heads
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1.0e-5,
            "rotary_dim": 8,
            // rope_theta omitted → defaults to 10000.0
            "num_experts": 4, // legacy alias for num_local_experts
            // num_experts_per_tok omitted
        });
        let kv = build_metadata(&cfg, 0);
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();

        assert_eq!(
            by_key["general.name"],
            MetaValue::String("model".into()),
            "name defaults to 'model'"
        );
        assert_eq!(
            by_key["minimax-m2.attention.head_count_kv"],
            MetaValue::U32(4),
            "head_count_kv defaults to head_count"
        );
        assert_eq!(
            by_key["minimax-m2.rope.freq_base"],
            MetaValue::F32(10000.0),
            "rope_theta defaults to 10000.0"
        );
        assert_eq!(
            by_key["minimax-m2.expert_count"],
            MetaValue::U32(4),
            "num_experts (alias) accepted when num_local_experts absent"
        );
        assert!(
            !by_key.contains_key("minimax-m2.expert_used_count"),
            "expert_used_count omitted when num_experts_per_tok absent"
        );
    }

    /// Sibling — FP8 source path is invisible at the mapper layer. The
    /// mapper has no opinion on dtype; it just renames. This test
    /// documents that the FP8 dispatch lives in the source_reader, not
    /// here, by exercising the same mapper against the same name
    /// regardless of source dtype.
    #[test]
    fn minimax_m2_mapper_is_dtype_agnostic() {
        // Whether the underlying safetensors entry was F8_E4M3 or
        // F32/BF16 (modules_to_not_convert path), the mapper output
        // is identical — it operates on names only.
        let name = "model.layers.0.block_sparse_moe.experts.5.w1.weight";
        match map_tensor_name(name) {
            Some(MappedTensor::ExpertWeight {
                layer,
                expert,
                role,
                ..
            }) => {
                assert_eq!(layer, 0);
                assert_eq!(expert, 5);
                assert_eq!(role, ExpertRole::Gate);
            }
            other => panic!("expert mapping: got {other:?}"),
        }
    }
}
