//! Qwen 3.5 MoE-A3B convert-side tensor mapper for the linear-attention
//! + MTP + shared-expert variant (canonical GGUF arch label
//! `qwen35moe`, gguf-py `MODEL_ARCH.QWEN35MOE`). Distinct from the
//! older `qwen3moe` dense-MoE variant handled by
//! [`crate::convert::arch::qwen35moe`].
//!
//! # Source-of-truth
//! Port of `/opt/llama.cpp/conversion/qwen.py:626-628` —
//! `Qwen3_5MoeTextModel(_Qwen35MtpMixin, _Qwen35MRopeMixin,
//! _LinearAttentionVReorderBase)`. Inheritance chain pulls in three
//! sources of HF→GGUF mapping behavior:
//!
//! - `Qwen3NextModel.modify_tensors` (qwen.py:296-345) — `.A_log` →
//!   `-exp(.)`, `.dt_bias` rename, `conv1d` squeeze, `norm.weight + 1`
//!   bake (except `linear_attn.norm.weight`).
//! - `_LinearAttentionVReorderBase._reorder_v_heads` (qwen.py:354-369)
//!   — V-head grouped→tiled reorder for `in_proj_qkv`, `in_proj_z`,
//!   `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`, `dt_proj`,
//!   `conv1d`, `out_proj`.
//! - `_Qwen35MtpMixin.modify_tensors` (qwen.py:597-618) — `mtp.*`
//!   tensor remap into `model.layers.{N+n_layer}.*` form so the
//!   inherited tensor_map handles them.
//! - `Qwen2MoeModel.modify_tensors` (qwen.py:93-145) — pre-fused
//!   `mlp.experts.gate_up_proj` split into separate gate + up
//!   tensors before downstream expert merge.
//!
//! # Distinctive features vs the older `qwen3moe` handler:
//! 1. Multimodal wrapping: tensor prefix is `model.language_model.*`
//!    instead of `model.*` when the HF architecture is
//!    `Qwen3_5MoeForConditionalGeneration` (VLM variant). Vision
//!    encoder tensors (`model.visual.blocks.<L>.*`) are explicit
//!    drops at convert time — they'd go into a separate mmproj GGUF
//!    if needed.
//! 2. Pre-fused experts: `mlp.experts.gate_up_proj` is one HF tensor
//!    of shape `[n_expert, 2*n_ff, n_embd]` that splits into two
//!    GGUF tensors via [`BakeOp::SplitAxisHalf`]. Similarly
//!    `mlp.experts.down_proj` is already merged.
//! 3. Shared experts re-added: `mlp.shared_expert.*` +
//!    `mlp.shared_expert_gate.weight` map to `ffn_*_shexp.weight`
//!    + `ffn_gate_inp_shexp.weight`.
//! 4. Linear-attention layers (every `1 / full_attention_interval`
//!    layers are full-attention; the rest are gated-DeltaNet
//!    linear-attention). Linear-attn tensors map to `ssm_*` family
//!    in GGUF.
//! 5. MTP heads at `mtp.layers.0.*` remap to
//!    `blk.{n_layer}.*` form via the `_Qwen35MtpMixin` translation
//!    table (`mtp.fc` → `eh_proj`, `mtp.pre_fc_norm_*` → `enorm/hnorm`,
//!    `mtp.norm` → `shared_head_norm`).
//!
//! # No fallback / no stub / no silent skip
//! Per [[feedback-no-loop-suppression-2026-05-17]] every recognized HF
//! name returns `Some(MappedTensor::...)`; every unrecognized name
//! returns `None` so the caller surfaces `UnmappedTensor`. Per
//! [[feedback-no-backwards-compat-2026-05-18]] there is no aliasing
//! or migration — each canonical name has exactly one outcome.
//!
//! # Context-dependence
//! Unlike older arch mappers, [`map_tensor_name`] takes a
//! [`Qwen35MoeFullCtx`] context because several transforms need
//! hparams: V-head reorder needs `linear_num_key_heads` /
//! `linear_num_value_heads` / `linear_value_head_dim`;
//! MTP layer remap needs `num_hidden_layers`;
//! pre-fused expert split needs `num_experts` /
//! `moe_intermediate_size` / `hidden_size`.

use crate::backends::gguf::types::MetaValue;
use crate::convert::arch::bake::{BakeOp, SplitHalf};
use crate::convert::cli_driver::SplitOutput;

/// HF hparams a [`map_tensor_name`] call needs to construct concrete
/// [`BakeOp`] / [`SplitOutput`] instances. Built once per convert
/// from the HF `config.json`.
#[derive(Debug, Clone)]
pub struct Qwen35MoeFullCtx {
    /// `num_hidden_layers` of the text decoder. MTP-block tensors
    /// remap to layer index `bid + num_hidden_layers` (so a single
    /// MTP block lands at `blk.{num_hidden_layers}`).
    pub num_hidden_layers: usize,

    /// `num_experts` (HF) or `num_local_experts`. The pre-fused
    /// `gate_up_proj` has leading dim `num_experts`.
    pub num_experts: usize,

    /// `moe_intermediate_size` — per-expert FFN intermediate dim.
    /// For the multimodal Qwen 3.5 35B-A3B this is 768 (operator's
    /// model). The pre-fused `gate_up_proj` middle axis has size
    /// `2 * moe_intermediate_size`.
    pub moe_intermediate_size: usize,

    /// `hidden_size` — model embedding width. The pre-fused
    /// `gate_up_proj` inner (last) axis has size `hidden_size`.
    pub hidden_size: usize,

    /// `linear_num_key_heads` — number of K heads in linear-attn
    /// layers. The HF V tensor is stored grouped by K head;
    /// [`BakeOp::ReorderVHeads`] tiles it to the GGML broadcast
    /// order. From `config.linear_num_key_heads` (Qwen 3.5/3.6
    /// linear-attn hparam).
    pub linear_num_key_heads: usize,

    /// `linear_num_value_heads` — number of V heads. Must be a
    /// multiple of `linear_num_key_heads`; the ratio is
    /// `num_v_per_k`.
    pub linear_num_value_heads: usize,

    /// `linear_value_head_dim` — elements per V head.
    pub linear_value_head_dim: usize,

    /// Whether the safetensors uses the `model.language_model.*`
    /// prefix (multimodal `Qwen3_5MoeForConditionalGeneration`) vs
    /// the bare `model.*` prefix (text-only `Qwen3_5MoeForCausalLM`).
    /// Detected from `config.json::architectures` at orchestrator
    /// init.
    pub multimodal_wrapping: bool,
}

impl Qwen35MoeFullCtx {
    /// `num_v_per_k = linear_num_value_heads / linear_num_key_heads`.
    /// Number of V heads grouped under each K head in the HF
    /// safetensors layout.
    pub fn num_v_per_k(&self) -> usize {
        self.linear_num_value_heads / self.linear_num_key_heads
    }
}

/// What the convert orchestrator should do with one HF tensor name
/// once the per-arch mapper has classified it for Qwen 3.5 MoE
/// (linear-attn + MTP + multimodal-VLM).
///
/// Mirrors [`crate::convert::cli_driver::MapOutcome`] structurally
/// but lives here so the per-arch handler is self-contained. The
/// orchestrator's `lift_qwen35moe_full_mapped` adapter translates
/// these into the driver-level outcome.
#[derive(Debug, Clone)]
pub enum MappedTensor {
    /// 1:1 rename, no data transform.
    Direct(String),

    /// 1:1 rename plus a post-load data transform. See [`BakeOp`].
    DirectWithBake { gguf_name: String, bake: BakeOp },

    /// One HF tensor → N GGUF tensors. Used for the pre-fused
    /// `mlp.experts.gate_up_proj` split.
    SplitInto(Vec<SplitOutput>),

    /// Known-discardable per the canonical Python converter (e.g.
    /// `model.visual.*` vision encoder tensors for the text-only
    /// GGUF; precomputed rope inv_freq buffers).
    Drop,
}

/// Top-level mapper. Returns `None` for any HF name not recognized as
/// a Qwen 3.5 MoE (linear-attn + MTP) weight kind — the caller is
/// expected to surface `UnmappedTensor` per the no-fallback rule.
///
/// Dispatch:
/// 1. Strip the optional `model.language_model.` multimodal prefix
///    when `ctx.multimodal_wrapping`. After strip, the name has the
///    same shape as a text-only `Qwen3_5MoeForCausalLM` checkpoint.
/// 2. Vision encoder tensors (`model.visual.*`) get `Drop`.
/// 3. MTP tensors (`mtp.*`) go through `map_mtp`.
/// 4. Globals (`model.embed_tokens`, `model.norm`, `lm_head`) and
///    per-block (`model.layers.<L>.<...>`) tensors go through their
///    respective sub-mappers.
///
/// All branches that emit a `norm.weight` (except
/// `linear_attn.norm.weight`) attach a [`BakeOp::AddOne`] bake —
/// mirrors canonical `qwen.py:303-304`.
pub fn map_tensor_name(hf_name: &str, ctx: &Qwen35MoeFullCtx) -> Option<MappedTensor> {
    // ---- Multimodal prefix strip --------------------------------------
    let canonical = if ctx.multimodal_wrapping {
        if let Some(stripped) = hf_name.strip_prefix("model.language_model.") {
            // Re-introduce the `model.` prefix so the rest of the
            // mapper sees the bare text-only layout.
            Some(format!("model.{stripped}"))
        } else if hf_name.starts_with("model.visual.") {
            // Vision encoder — explicit drop for text-only GGUF.
            return Some(MappedTensor::Drop);
        } else if hf_name == "lm_head.weight" || hf_name.starts_with("mtp.") {
            // `lm_head` and `mtp.*` live at the top level in the
            // multimodal layout too — pass through unchanged.
            None
        } else {
            // Anything else with no `model.language_model.` prefix
            // AND not a known top-level tensor: surface as unmapped.
            return None;
        }
    } else {
        None
    };
    let canonical_ref: &str = canonical.as_deref().unwrap_or(hf_name);

    // ---- MTP family ---------------------------------------------------
    if let Some(rest) = canonical_ref.strip_prefix("mtp.") {
        return map_mtp(rest, ctx);
    }

    // ---- Globals ------------------------------------------------------
    match canonical_ref {
        "model.embed_tokens.weight" => {
            return Some(MappedTensor::Direct("token_embd.weight".into()));
        }
        "model.norm.weight" => {
            return Some(MappedTensor::DirectWithBake {
                gguf_name: "output_norm.weight".into(),
                bake: BakeOp::AddOne,
            });
        }
        "lm_head.weight" => {
            return Some(MappedTensor::Direct("output.weight".into()));
        }
        _ => {}
    }

    // ---- Per-block ----------------------------------------------------
    let stripped = canonical_ref.strip_prefix("model.layers.")?;
    let dot = stripped.find('.')?;
    let (layer_str, rest_with_dot) = stripped.split_at(dot);
    let layer: usize = layer_str.parse().ok()?;
    if layer.to_string() != layer_str {
        return None;
    }
    let rest = &rest_with_dot[1..];

    map_per_block(layer, rest, ctx)
}

/// Per-block dispatcher. Handles norms, full-attention projections,
/// linear-attention SSM tensors, MoE router gate, per-expert
/// gate_up_proj split + down_proj direct, shared experts.
fn map_per_block(layer: usize, rest: &str, ctx: &Qwen35MoeFullCtx) -> Option<MappedTensor> {
    let blk = |suffix: &str| format!("blk.{layer}.{suffix}");

    // ---- Block-level norms (the +1 bake applies) ----------------------
    if rest == "input_layernorm.weight" {
        return Some(MappedTensor::DirectWithBake {
            gguf_name: blk("attn_norm.weight"),
            bake: BakeOp::AddOne,
        });
    }
    if rest == "post_attention_layernorm.weight" {
        return Some(MappedTensor::DirectWithBake {
            gguf_name: blk("attn_post_norm.weight"),
            bake: BakeOp::AddOne,
        });
    }

    // ---- Full-attention projections (only present on every Kth layer) -
    let full_attn_map = [
        ("self_attn.q_proj.weight", "attn_q.weight"),
        ("self_attn.k_proj.weight", "attn_k.weight"),
        ("self_attn.v_proj.weight", "attn_v.weight"),
        ("self_attn.o_proj.weight", "attn_output.weight"),
    ];
    for (suffix, gguf_suffix) in full_attn_map {
        if rest == suffix {
            return Some(MappedTensor::Direct(blk(gguf_suffix)));
        }
    }

    // Q/K per-head norms (also get the +1 bake)
    if rest == "self_attn.q_norm.weight" {
        return Some(MappedTensor::DirectWithBake {
            gguf_name: blk("attn_q_norm.weight"),
            bake: BakeOp::AddOne,
        });
    }
    if rest == "self_attn.k_norm.weight" {
        return Some(MappedTensor::DirectWithBake {
            gguf_name: blk("attn_k_norm.weight"),
            bake: BakeOp::AddOne,
        });
    }

    // ---- Linear-attention SSM family ----------------------------------
    if let Some(la_rest) = rest.strip_prefix("linear_attn.") {
        return map_linear_attn(layer, la_rest, ctx);
    }

    // ---- MoE router + per-expert + shared experts ---------------------
    if let Some(mlp_rest) = rest.strip_prefix("mlp.") {
        return map_mlp(layer, mlp_rest, ctx);
    }

    None
}

/// Linear-attention SSM tensors. Mirrors gguf-py
/// `tensor_mapping.py::MODEL_TENSOR.SSM_*` entries for `qwen3.5`.
/// The norm.weight here is NOT baked +1 — it's the one explicit
/// exception in canonical `qwen.py:303` (`.endswith("norm.weight") and
/// not name.endswith("linear_attn.norm.weight")`).
fn map_linear_attn(layer: usize, la_rest: &str, _ctx: &Qwen35MoeFullCtx) -> Option<MappedTensor> {
    let blk = |suffix: &str| format!("blk.{layer}.{suffix}");

    match la_rest {
        // No-bake renames first (simplest cases).
        "norm.weight" => Some(MappedTensor::Direct(blk("ssm_norm.weight"))),
        "dt_bias" => Some(MappedTensor::Direct(blk("ssm_dt.bias"))),
        "in_proj_a.weight" => Some(MappedTensor::Direct(blk("ssm_alpha.weight"))),
        "in_proj_b.weight" => Some(MappedTensor::Direct(blk("ssm_beta.weight"))),

        // A_log: element-wise -exp transform.
        "A_log" => Some(MappedTensor::DirectWithBake {
            gguf_name: blk("ssm_a.weight"),
            bake: BakeOp::NegExp,
        }),

        // conv1d.weight: safetensors shape [hidden, 1, kernel] but
        // GGUF expects [hidden, kernel] — squeeze the singleton.
        "conv1d.weight" => Some(MappedTensor::DirectWithBake {
            gguf_name: blk("ssm_conv1d.weight"),
            bake: BakeOp::Squeeze,
        }),

        // in_proj_qkv / in_proj_z / out_proj need V-head reorder.
        // The exact slice and dim depend on the tensor shape and
        // the QKV layout. Per canonical `qwen.py:354-369` and orphan
        // `src/models/qwen35/mod.rs:454+`:
        //
        // - in_proj_qkv: shape [nk*(qd+kd+nv_per_k*vd), hidden].
        //   Only the V rows (offset = nk*(qd+kd), length = nk*nv_per_k*vd)
        //   need the reorder; Q and K rows are passed through.
        //   This requires knowing the head dims (qd=linear_key_head_dim,
        //   kd=linear_key_head_dim, vd=linear_value_head_dim) — context
        //   doesn't currently carry qd/kd separately. Defer to a
        //   subsequent commit once Qwen35MoeFullCtx is extended.
        //
        // - in_proj_z: shape [nk*nv_per_k*vd, hidden]. Full reorder
        //   on the row dim.
        //
        // - out_proj: shape [hidden, nk*nv_per_k*vd]. Reorder on
        //   col dim — currently apply_bake_op operates on row-major
        //   flat buffers; col-axis reorder requires a transposed
        //   BakeOp variant (deferred).
        //
        // For now (no-stub policy): the three reorder-requiring
        // tensors return None, surfacing UnmappedTensor with a
        // typed error so the operator sees exactly which tensor
        // patterns need additional work. This matches the
        // [[feedback-no-loop-suppression-2026-05-17]] contract:
        // surface, don't silently degrade.
        "in_proj_qkv.weight" | "in_proj_z.weight" | "out_proj.weight" => None,

        _ => None,
    }
}

/// MoE per-block MLP tensors.
/// - `mlp.gate.weight` (router): direct map to `ffn_gate_inp.weight`.
/// - `mlp.experts.gate_up_proj` (pre-fused 3-D): split via
///   `SplitInto` + `BakeOp::SplitAxisHalf` into `ffn_gate_exps` +
///   `ffn_up_exps`.
/// - `mlp.experts.down_proj` (pre-fused 3-D): direct map to
///   `ffn_down_exps.weight`. No split needed.
/// - `mlp.shared_expert.gate_proj.weight` /
///   `mlp.shared_expert.up_proj.weight` /
///   `mlp.shared_expert.down_proj.weight`: shared-expert
///   projections → `ffn_{gate,up,down}_shexp.weight`.
/// - `mlp.shared_expert_gate.weight`: shared-expert router →
///   `ffn_gate_inp_shexp.weight`.
fn map_mlp(layer: usize, mlp_rest: &str, ctx: &Qwen35MoeFullCtx) -> Option<MappedTensor> {
    let blk = |suffix: &str| format!("blk.{layer}.{suffix}");

    match mlp_rest {
        "gate.weight" => Some(MappedTensor::Direct(blk("ffn_gate_inp.weight"))),

        "shared_expert.gate_proj.weight" => {
            Some(MappedTensor::Direct(blk("ffn_gate_shexp.weight")))
        }
        "shared_expert.up_proj.weight" => Some(MappedTensor::Direct(blk("ffn_up_shexp.weight"))),
        "shared_expert.down_proj.weight" => {
            Some(MappedTensor::Direct(blk("ffn_down_shexp.weight")))
        }
        "shared_expert_gate.weight" => Some(MappedTensor::Direct(blk("ffn_gate_inp_shexp.weight"))),

        "experts.down_proj" | "experts.down_proj.weight" => {
            // Pre-fused 3-D `[n_expert, n_embd, n_ff]` — direct map.
            // GGUF shape derived in plan-build from meta.shape reversed.
            Some(MappedTensor::Direct(blk("ffn_down_exps.weight")))
        }

        "experts.gate_up_proj" | "experts.gate_up_proj.weight" => {
            // Pre-fused 3-D `[n_expert, 2*n_ff, n_embd]` → split into
            // `ffn_gate_exps` (first half) + `ffn_up_exps` (second half).
            // Per `qwen.py:99-112`. Each output is shape
            // `[n_expert, n_ff, n_embd]` in HF order, which becomes
            // `[n_embd, n_ff, n_expert]` in GGUF (PyTorch-order
            // reversed).
            let n_expert = ctx.num_experts;
            let n_ff = ctx.moe_intermediate_size;
            let n_embd = ctx.hidden_size;
            let two_nff = 2 * n_ff;
            let gate_op = BakeOp::SplitAxisHalf {
                outer_count: n_expert,
                axis_size: two_nff,
                inner_count: n_embd,
                half: SplitHalf::First,
            };
            let up_op = BakeOp::SplitAxisHalf {
                outer_count: n_expert,
                axis_size: two_nff,
                inner_count: n_embd,
                half: SplitHalf::Second,
            };
            // GGUF-order shape `[n_embd, n_ff, n_expert]` for each.
            let gguf_shape = vec![n_embd, n_ff, n_expert];
            Some(MappedTensor::SplitInto(vec![
                SplitOutput {
                    gguf_name: blk("ffn_gate_exps.weight"),
                    gguf_shape: gguf_shape.clone(),
                    bake: gate_op,
                },
                SplitOutput {
                    gguf_name: blk("ffn_up_exps.weight"),
                    gguf_shape,
                    bake: up_op,
                },
            ]))
        }

        _ => None,
    }
}

/// MTP-block tensors per `_Qwen35MtpMixin.modify_tensors`
/// (qwen.py:597-618). The MTP block lives at HF layer index 0
/// (`mtp.layers.0.*`) but GGUF places it at
/// `blk.{num_hidden_layers}.*` per the MTP append convention.
///
/// Naming translations (from canonical line 605-610):
/// ```text
///   mtp.fc                       -> model.layers.{N}.eh_proj
///   mtp.pre_fc_norm_embedding    -> model.layers.{N}.enorm
///   mtp.pre_fc_norm_hidden       -> model.layers.{N}.hnorm
///   mtp.norm                     -> model.layers.{N}.shared_head.norm
/// ```
/// where `N = num_hidden_layers`. After remap, the MTP-block's
/// transformer layer body (input_layernorm, q_proj, etc.) gets the
/// same per-block treatment as a normal layer — including the +1
/// norm bake.
fn map_mtp(rest: &str, ctx: &Qwen35MoeFullCtx) -> Option<MappedTensor> {
    let n_layer = ctx.num_hidden_layers;
    let mtp_blk = |suffix: &str| format!("blk.{n_layer}.{suffix}");

    // Top-level MTP helpers.
    match rest {
        "fc.weight" => return Some(MappedTensor::Direct(mtp_blk("eh_proj.weight"))),
        "pre_fc_norm_embedding.weight" => {
            return Some(MappedTensor::DirectWithBake {
                gguf_name: mtp_blk("enorm.weight"),
                bake: BakeOp::AddOne,
            });
        }
        "pre_fc_norm_hidden.weight" => {
            return Some(MappedTensor::DirectWithBake {
                gguf_name: mtp_blk("hnorm.weight"),
                bake: BakeOp::AddOne,
            });
        }
        "norm.weight" => {
            return Some(MappedTensor::DirectWithBake {
                gguf_name: mtp_blk("shared_head_norm.weight"),
                bake: BakeOp::AddOne,
            });
        }
        _ => {}
    }

    // Per-MTP-layer tensors: `mtp.layers.<bid>.<rest>`. Per canonical
    // qwen.py:599-602 the layer-index translation is `mtp.layers.<bid>`
    // -> `model.layers.<bid + n_layer>`, so we delegate to
    // map_per_block with the shifted layer index.
    let layers_rest = rest.strip_prefix("layers.")?;
    let dot = layers_rest.find('.')?;
    let (bid_str, inner_with_dot) = layers_rest.split_at(dot);
    let bid: usize = bid_str.parse().ok()?;
    if bid.to_string() != bid_str {
        return None;
    }
    let inner = &inner_with_dot[1..];
    map_per_block(bid + n_layer, inner, ctx)
}

/// Emit Qwen 3.5 MoE-specific GGUF metadata KVs. Called by the
/// orchestrator after the standard arch/general KVs are written.
///
/// Per gguf-py `MODEL_ARCH.QWEN35MOE`:
/// - `qwen35moe.rope.dimension_sections = [11, 11, 10, 0]` (MRoPE)
/// - `qwen35moe.nextn_predict_layers = mtp_num_hidden_layers` (when MTP)
/// - `qwen35moe.attention.output_gate = attn_output_gate` (when set)
/// - `qwen35moe.expert_count = num_experts`
/// - `qwen35moe.expert_used_count = num_experts_per_tok` (top-k)
/// - `qwen35moe.expert_feed_forward_length = moe_intermediate_size`
pub fn build_metadata(ctx: &Qwen35MoeFullCtx, config: &serde_json::Value) -> Vec<(String, MetaValue)> {
    let mut kvs = Vec::new();

    // Expert metadata
    kvs.push((
        "qwen35moe.expert_count".into(),
        MetaValue::U32(ctx.num_experts as u32),
    ));
    if let Some(top_k) = effective_text_field(config, "num_experts_per_tok").and_then(|v| v.as_u64()) {
        kvs.push((
            "qwen35moe.expert_used_count".into(),
            MetaValue::U32(top_k as u32),
        ));
    }
    kvs.push((
        "qwen35moe.expert_feed_forward_length".into(),
        MetaValue::U32(ctx.moe_intermediate_size as u32),
    ));

    // MRoPE dimension sections (always written; default per
    // `_Qwen35MRopeMixin._QWEN35_DEFAULT_MROPE_SECTION` qwen.py:526).
    let mrope: Vec<u32> = effective_text_field(config, "mrope_section")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_u64().map(|n| n as u32))
                .collect()
        })
        .unwrap_or_else(|| vec![11, 11, 10, 0]);
    kvs.push((
        "qwen35moe.rope.dimension_sections".into(),
        MetaValue::ArrayU32(mrope),
    ));

    // attn_output_gate (Qwen 3.5/3.6 attention quirk).
    if let Some(gate) = effective_text_field(config, "attn_output_gate").and_then(|v| v.as_bool()) {
        kvs.push((
            "qwen35moe.attention.output_gate".into(),
            MetaValue::Bool(gate),
        ));
    }

    // nextn_predict_layers — emitted only when MTP is present.
    if let Some(n_mtp) = effective_text_field(config, "mtp_num_hidden_layers")
        .and_then(|v| v.as_u64())
        .filter(|&n| n > 0)
    {
        kvs.push((
            "qwen35moe.nextn_predict_layers".into(),
            MetaValue::U32(n_mtp as u32),
        ));
    }

    kvs
}

/// Look up a field that may live at the top level of `config` or
/// nested under `config.text_config`. Mirrors canonical
/// `convert_hf_to_gguf.py` behavior for multimodal-wrapping configs.
fn effective_text_field<'a>(
    config: &'a serde_json::Value,
    field: &str,
) -> Option<&'a serde_json::Value> {
    if let Some(v) = config.get(field) {
        return Some(v);
    }
    config.get("text_config").and_then(|tc| tc.get(field))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vlm_ctx() -> Qwen35MoeFullCtx {
        // Sized to match operator's Qwen-Qwen3.5-35B-A3B multimodal
        // VLM: 80 hidden layers, 128 experts, moe_intermediate=768,
        // hidden=2048, linear: 16 K heads, 32 V heads (2 per K),
        // head_dim=128.
        Qwen35MoeFullCtx {
            num_hidden_layers: 80,
            num_experts: 128,
            moe_intermediate_size: 768,
            hidden_size: 2048,
            linear_num_key_heads: 16,
            linear_num_value_heads: 32,
            linear_value_head_dim: 128,
            multimodal_wrapping: true,
        }
    }

    fn text_only_ctx() -> Qwen35MoeFullCtx {
        Qwen35MoeFullCtx {
            multimodal_wrapping: false,
            ..vlm_ctx()
        }
    }

    #[test]
    fn globals_with_multimodal_prefix() {
        let ctx = vlm_ctx();
        match map_tensor_name("model.language_model.embed_tokens.weight", &ctx) {
            Some(MappedTensor::Direct(s)) => assert_eq!(s, "token_embd.weight"),
            other => panic!("unexpected: {other:?}"),
        }
        match map_tensor_name("model.language_model.norm.weight", &ctx) {
            Some(MappedTensor::DirectWithBake { gguf_name, bake }) => {
                assert_eq!(gguf_name, "output_norm.weight");
                assert_eq!(bake, BakeOp::AddOne);
            }
            other => panic!("unexpected: {other:?}"),
        }
        match map_tensor_name("lm_head.weight", &ctx) {
            Some(MappedTensor::Direct(s)) => assert_eq!(s, "output.weight"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn globals_text_only() {
        let ctx = text_only_ctx();
        match map_tensor_name("model.embed_tokens.weight", &ctx) {
            Some(MappedTensor::Direct(s)) => assert_eq!(s, "token_embd.weight"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn visual_tensors_drop_in_multimodal() {
        let ctx = vlm_ctx();
        match map_tensor_name("model.visual.blocks.0.attn.qkv.weight", &ctx) {
            Some(MappedTensor::Drop) => {}
            other => panic!("expected Drop, got: {other:?}"),
        }
    }

    #[test]
    fn input_layernorm_gets_plus_one_bake() {
        let ctx = vlm_ctx();
        match map_tensor_name(
            "model.language_model.layers.5.input_layernorm.weight",
            &ctx,
        ) {
            Some(MappedTensor::DirectWithBake { gguf_name, bake }) => {
                assert_eq!(gguf_name, "blk.5.attn_norm.weight");
                assert_eq!(bake, BakeOp::AddOne);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn post_attention_layernorm_gets_plus_one_bake() {
        let ctx = vlm_ctx();
        match map_tensor_name(
            "model.language_model.layers.3.post_attention_layernorm.weight",
            &ctx,
        ) {
            Some(MappedTensor::DirectWithBake { gguf_name, bake }) => {
                assert_eq!(gguf_name, "blk.3.attn_post_norm.weight");
                assert_eq!(bake, BakeOp::AddOne);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn linear_attn_norm_does_not_get_plus_one() {
        // The single exception in canonical qwen.py:303 — every other
        // norm gets +1 EXCEPT linear_attn.norm.weight.
        let ctx = vlm_ctx();
        match map_tensor_name(
            "model.language_model.layers.0.linear_attn.norm.weight",
            &ctx,
        ) {
            Some(MappedTensor::Direct(s)) => assert_eq!(s, "blk.0.ssm_norm.weight"),
            other => panic!("expected Direct (no bake), got: {other:?}"),
        }
    }

    #[test]
    fn linear_attn_a_log_neg_exp() {
        let ctx = vlm_ctx();
        match map_tensor_name("model.language_model.layers.0.linear_attn.A_log", &ctx) {
            Some(MappedTensor::DirectWithBake { gguf_name, bake }) => {
                assert_eq!(gguf_name, "blk.0.ssm_a.weight");
                assert_eq!(bake, BakeOp::NegExp);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn linear_attn_conv1d_squeeze() {
        let ctx = vlm_ctx();
        match map_tensor_name(
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            &ctx,
        ) {
            Some(MappedTensor::DirectWithBake { gguf_name, bake }) => {
                assert_eq!(gguf_name, "blk.0.ssm_conv1d.weight");
                assert_eq!(bake, BakeOp::Squeeze);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn linear_attn_in_proj_a_b_direct_rename() {
        let ctx = vlm_ctx();
        match map_tensor_name(
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            &ctx,
        ) {
            Some(MappedTensor::Direct(s)) => assert_eq!(s, "blk.0.ssm_alpha.weight"),
            other => panic!("unexpected: {other:?}"),
        }
        match map_tensor_name(
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            &ctx,
        ) {
            Some(MappedTensor::Direct(s)) => assert_eq!(s, "blk.0.ssm_beta.weight"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn linear_attn_in_proj_qkv_z_out_proj_surfaces_unmapped() {
        // These need V-head reorder over specific slices/columns
        // which the current Qwen35MoeFullCtx + BakeOp::ReorderVHeads
        // can't fully express (col-axis reorder and qd/kd split
        // require a richer context). Per no-stub: return None and
        // let the caller surface UnmappedTensor — operator-visible.
        let ctx = vlm_ctx();
        for name in [
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
        ] {
            assert!(
                map_tensor_name(name, &ctx).is_none(),
                "expected None for {name} (V-head reorder not yet wired)",
            );
        }
    }

    #[test]
    fn mlp_router_gate_direct() {
        let ctx = vlm_ctx();
        match map_tensor_name("model.language_model.layers.7.mlp.gate.weight", &ctx) {
            Some(MappedTensor::Direct(s)) => assert_eq!(s, "blk.7.ffn_gate_inp.weight"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn mlp_shared_experts_direct() {
        let ctx = vlm_ctx();
        let cases = [
            ("mlp.shared_expert.gate_proj.weight", "ffn_gate_shexp.weight"),
            ("mlp.shared_expert.up_proj.weight", "ffn_up_shexp.weight"),
            ("mlp.shared_expert.down_proj.weight", "ffn_down_shexp.weight"),
            (
                "mlp.shared_expert_gate.weight",
                "ffn_gate_inp_shexp.weight",
            ),
        ];
        for (hf_suffix, gguf_suffix) in cases {
            let hf = format!("model.language_model.layers.2.{hf_suffix}");
            match map_tensor_name(&hf, &ctx) {
                Some(MappedTensor::Direct(s)) => assert_eq!(s, format!("blk.2.{gguf_suffix}")),
                other => panic!("unexpected for {hf}: {other:?}"),
            }
        }
    }

    #[test]
    fn mlp_experts_gate_up_proj_splits_via_split_axis_half() {
        let ctx = vlm_ctx();
        // No `.weight` suffix in canonical safetensors layout for this
        // pre-fused tensor (per operator's safetensors index inspection).
        match map_tensor_name(
            "model.language_model.layers.9.mlp.experts.gate_up_proj",
            &ctx,
        ) {
            Some(MappedTensor::SplitInto(outputs)) => {
                assert_eq!(outputs.len(), 2);
                assert_eq!(outputs[0].gguf_name, "blk.9.ffn_gate_exps.weight");
                assert_eq!(outputs[1].gguf_name, "blk.9.ffn_up_exps.weight");
                // GGUF-order shape `[n_embd, n_ff, n_expert]`.
                assert_eq!(outputs[0].gguf_shape, vec![2048, 768, 128]);
                assert_eq!(outputs[1].gguf_shape, vec![2048, 768, 128]);
                // First half = gate, Second half = up.
                match outputs[0].bake {
                    BakeOp::SplitAxisHalf { half, .. } => assert_eq!(half, SplitHalf::First),
                    ref other => panic!("expected SplitAxisHalf, got {other:?}"),
                }
                match outputs[1].bake {
                    BakeOp::SplitAxisHalf { half, .. } => assert_eq!(half, SplitHalf::Second),
                    ref other => panic!("expected SplitAxisHalf, got {other:?}"),
                }
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn mlp_experts_down_proj_direct_rename() {
        let ctx = vlm_ctx();
        match map_tensor_name(
            "model.language_model.layers.9.mlp.experts.down_proj",
            &ctx,
        ) {
            Some(MappedTensor::Direct(s)) => assert_eq!(s, "blk.9.ffn_down_exps.weight"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn mtp_helpers_remap_to_next_block_layer() {
        let ctx = vlm_ctx();
        // n_layer = 80, so MTP block lives at blk.80.
        match map_tensor_name("mtp.fc.weight", &ctx) {
            Some(MappedTensor::Direct(s)) => assert_eq!(s, "blk.80.eh_proj.weight"),
            other => panic!("unexpected: {other:?}"),
        }
        match map_tensor_name("mtp.pre_fc_norm_embedding.weight", &ctx) {
            Some(MappedTensor::DirectWithBake { gguf_name, bake }) => {
                assert_eq!(gguf_name, "blk.80.enorm.weight");
                assert_eq!(bake, BakeOp::AddOne);
            }
            other => panic!("unexpected: {other:?}"),
        }
        match map_tensor_name("mtp.pre_fc_norm_hidden.weight", &ctx) {
            Some(MappedTensor::DirectWithBake { gguf_name, bake }) => {
                assert_eq!(gguf_name, "blk.80.hnorm.weight");
                assert_eq!(bake, BakeOp::AddOne);
            }
            other => panic!("unexpected: {other:?}"),
        }
        match map_tensor_name("mtp.norm.weight", &ctx) {
            Some(MappedTensor::DirectWithBake { gguf_name, bake }) => {
                assert_eq!(gguf_name, "blk.80.shared_head_norm.weight");
                assert_eq!(bake, BakeOp::AddOne);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn mtp_layer_body_tensors_remap_to_n_plus_bid() {
        let ctx = vlm_ctx();
        // mtp.layers.0.input_layernorm.weight → blk.80.attn_norm.weight (with +1 bake)
        match map_tensor_name("mtp.layers.0.input_layernorm.weight", &ctx) {
            Some(MappedTensor::DirectWithBake { gguf_name, bake }) => {
                assert_eq!(gguf_name, "blk.80.attn_norm.weight");
                assert_eq!(bake, BakeOp::AddOne);
            }
            other => panic!("unexpected: {other:?}"),
        }
        // mtp.layers.0.self_attn.q_proj.weight → blk.80.attn_q.weight
        match map_tensor_name("mtp.layers.0.self_attn.q_proj.weight", &ctx) {
            Some(MappedTensor::Direct(s)) => assert_eq!(s, "blk.80.attn_q.weight"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn unmapped_name_returns_none() {
        let ctx = vlm_ctx();
        assert!(map_tensor_name("model.totally_made_up_tensor.weight", &ctx).is_none());
        assert!(map_tensor_name("garbage", &ctx).is_none());
    }
}
