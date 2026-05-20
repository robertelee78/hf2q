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
use crate::convert::arch::qwen35moe::ExpertKind;
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

    /// `linear_key_head_dim` — elements per K (and Q) head in
    /// linear-attn layers. Used by `in_proj_qkv` to compute the
    /// byte offset of the V-row slice (Q rows + K rows pass through
    /// unchanged; only V rows get the reorder).
    pub linear_key_head_dim: usize,

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

    /// One HF tensor is one slice of a fused GGUF expert tensor. Used
    /// for the per-expert layout in the MTP block (mtp.layers.0.mlp.
    /// experts.<E>.<kind>_proj.weight) — the MTP block stores experts
    /// per-expert even though the main model layers pre-fuse them.
    /// The caller accumulates every expert of a given `(layer, kind)`
    /// group and emits a single 3-D GGUF tensor named `gguf_name`
    /// once `expert_index` covers `[0, n_experts)`. Mirrors the
    /// older `qwen35moe::MappedTensor::ExpertGroup` semantics.
    ExpertGroup {
        gguf_name: String,
        layer: usize,
        expert_index: usize,
        kind: ExpertKind,
    },

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
pub fn map_tensor_name(
    hf_name: &str,
    hf_shape: &[usize],
    ctx: &Qwen35MoeFullCtx,
) -> Option<MappedTensor> {
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
        return map_mtp(rest, hf_shape, ctx);
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

    map_per_block(layer, rest, hf_shape, ctx)
}

/// Per-block dispatcher. Handles norms, full-attention projections,
/// linear-attention SSM tensors, MoE router gate, per-expert
/// gate_up_proj split + down_proj direct, shared experts.
fn map_per_block(
    layer: usize,
    rest: &str,
    hf_shape: &[usize],
    ctx: &Qwen35MoeFullCtx,
) -> Option<MappedTensor> {
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
        return map_linear_attn(layer, la_rest, hf_shape, ctx);
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
fn map_linear_attn(
    layer: usize,
    la_rest: &str,
    hf_shape: &[usize],
    ctx: &Qwen35MoeFullCtx,
) -> Option<MappedTensor> {
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

        // in_proj_qkv: HF shape [nk*(qd+kd) + nk*nv_per_k*vd, cols]
        // — Q rows + K rows + V rows packed along axis 0. Q and K
        // rows pass through unchanged; ONLY the V rows get the
        // grouped→tiled head reorder. Per canonical
        // `qwen.py:5384-5392` + orphan `mod.rs:552-616`.
        //
        // The flat F32 buffer is `total_rows * cols` elements, where
        // total_rows = shape[0] and cols = shape[1] (if 2-D).
        // V slice start = (qd+kd) * nk * cols (Q+K rows × cols),
        // V slice length = nv_per_k * vd * nk * cols.
        // Each V "head" treats `vd * cols` scalars as one head_dim
        // block; reorder swaps the outer [nk, nv_per_k] axes.
        "in_proj_qkv.weight" => {
            let cols = if hf_shape.len() >= 2 { hf_shape[1] } else { 1 };
            let nk = ctx.linear_num_key_heads;
            let nv_per_k = ctx.num_v_per_k();
            let qd = ctx.linear_key_head_dim;
            let kd = ctx.linear_key_head_dim;
            let vd = ctx.linear_value_head_dim;
            let v_slice_start = (qd + kd) * nk * cols;
            let v_slice_len = nv_per_k * vd * nk * cols;
            Some(MappedTensor::DirectWithBake {
                gguf_name: blk("ssm_in_proj_qkv.weight"),
                bake: BakeOp::ReorderVHeads {
                    num_k_heads: nk,
                    num_v_per_k: nv_per_k,
                    head_dim: vd * cols,
                    slice: Some(v_slice_start..(v_slice_start + v_slice_len)),
                },
            })
        }

        // in_proj_z: HF shape [nk * nv_per_k * vd, cols]. ALL rows
        // get reorder (no Q/K to skip). Treat each "head" as
        // `vd * cols` scalars laid out as [nk, nv_per_k, vd*cols] →
        // swap outer two axes. Per canonical `qwen.py:5394-5396` +
        // orphan `mod.rs:624-637`.
        "in_proj_z.weight" => {
            let cols = if hf_shape.len() >= 2 { hf_shape[1] } else { 1 };
            let nk = ctx.linear_num_key_heads;
            let nv_per_k = ctx.num_v_per_k();
            let vd = ctx.linear_value_head_dim;
            Some(MappedTensor::DirectWithBake {
                gguf_name: blk("ssm_in_proj_z.weight"),
                bake: BakeOp::ReorderVHeads {
                    num_k_heads: nk,
                    num_v_per_k: nv_per_k,
                    head_dim: vd * cols,
                    slice: None,
                },
            })
        }

        // out_proj: HF shape [hidden, nk * nv_per_k * vd]. PER-ROW
        // reorder of the column axis: each row's cols elements are
        // laid out as [nk, nv_per_k, vd] → swap [nk, nv_per_k]. Per
        // canonical `qwen.py:5402-5408` + orphan `mod.rs:670-705`.
        "out_proj.weight" => {
            if hf_shape.len() < 2 {
                return None;
            }
            let rows = hf_shape[0];
            let nk = ctx.linear_num_key_heads;
            let nv_per_k = ctx.num_v_per_k();
            let vd = ctx.linear_value_head_dim;
            // Sanity: each row must equal nk * nv_per_k * vd.
            if hf_shape[1] != nk * nv_per_k * vd {
                return None;
            }
            Some(MappedTensor::DirectWithBake {
                gguf_name: blk("ssm_out.weight"),
                bake: BakeOp::ReorderVHeadsPerRow {
                    row_count: rows,
                    num_k_heads: nk,
                    num_v_per_k: nv_per_k,
                    head_dim_in_row: vd,
                },
            })
        }

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

        // Per-expert layout (MTP block uses this even though the main
        // model layers pre-fuse experts into one tensor — verified
        // 2026-05-19 on operator's Qwen-Qwen3.5-35B-A3B safetensors
        // which has 256 mtp.layers.0.mlp.experts.<E>.{gate,up,down}_proj.weight
        // tensors). Routes through ExpertGroup so the orchestrator
        // accumulates and fuses at plan-build time.
        rest if rest.starts_with("experts.") && rest != "experts.gate_up_proj"
            && rest != "experts.gate_up_proj.weight"
            && rest != "experts.down_proj"
            && rest != "experts.down_proj.weight" =>
        {
            let expert_rest = rest.strip_prefix("experts.")?;
            let dot = expert_rest.find('.')?;
            let (expert_str, kind_with_dot) = expert_rest.split_at(dot);
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
                gguf_name: blk(gguf_suffix),
                layer,
                expert_index,
                kind,
            })
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
fn map_mtp(rest: &str, hf_shape: &[usize], ctx: &Qwen35MoeFullCtx) -> Option<MappedTensor> {
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
    map_per_block(bid + n_layer, inner, hf_shape, ctx)
}

/// Emit the full set of `qwen35moe.*` + `general.*` GGUF metadata KVs.
/// Mirrors gguf-py `MODEL_ARCH.QWEN35MOE` and the canonical
/// `Qwen3_5MoeTextModel.set_gguf_parameters` chain (qwen.py:522-616
/// for `_Qwen35MRopeMixin` + `_Qwen35MtpMixin` + inherited Qwen3MoE
/// hparams).
pub fn build_metadata(
    ctx: &Qwen35MoeFullCtx,
    config: &serde_json::Value,
    file_type: u32,
) -> Vec<(String, MetaValue)> {
    let text = effective_text_config(config);
    let name = config
        .get("_name_or_path")
        .and_then(|v| v.as_str())
        .unwrap_or("model")
        .to_string();

    let hidden_size = text
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .expect("config missing hidden_size") as u32;
    let n_layers_base = text
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .expect("config missing num_hidden_layers") as u32;
    let n_mtp = text
        .get("mtp_num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let n_layers = n_layers_base + n_mtp;
    let n_head = text
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .expect("config missing num_attention_heads") as u32;
    let n_head_kv = text
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32)
        .unwrap_or(n_head);
    let ctx_len = text
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .expect("config missing max_position_embeddings") as u32;
    let rms_eps = text
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .expect("config missing rms_norm_eps") as f32;
    let moe_ffn = text
        .get("moe_intermediate_size")
        .and_then(|v| v.as_u64())
        .expect("config missing moe_intermediate_size") as u32;
    let n_experts = text
        .get("num_experts")
        .or_else(|| text.get("num_local_experts"))
        .and_then(|v| v.as_u64())
        .expect("config missing num_experts") as u32;
    let n_experts_used = text
        .get("num_experts_per_tok")
        .and_then(|v| v.as_u64())
        .expect("config missing num_experts_per_tok") as u32;
    let rope_theta = text
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0) as f32;

    let mut kvs: Vec<(String, MetaValue)> = vec![
        (
            "general.architecture".into(),
            MetaValue::String("qwen35moe".into()),
        ),
        ("general.name".into(), MetaValue::String(name)),
        ("qwen35moe.context_length".into(), MetaValue::U32(ctx_len)),
        (
            "qwen35moe.embedding_length".into(),
            MetaValue::U32(hidden_size),
        ),
        ("qwen35moe.block_count".into(), MetaValue::U32(n_layers)),
        (
            "qwen35moe.feed_forward_length".into(),
            MetaValue::U32(moe_ffn),
        ),
        (
            "qwen35moe.attention.head_count".into(),
            MetaValue::U32(n_head),
        ),
        (
            "qwen35moe.attention.head_count_kv".into(),
            MetaValue::U32(n_head_kv),
        ),
        (
            "qwen35moe.attention.layer_norm_rms_epsilon".into(),
            MetaValue::F32(rms_eps),
        ),
        (
            "qwen35moe.rope.freq_base".into(),
            MetaValue::F32(rope_theta),
        ),
        (
            "qwen35moe.expert_count".into(),
            MetaValue::U32(n_experts),
        ),
        (
            "qwen35moe.expert_used_count".into(),
            MetaValue::U32(n_experts_used),
        ),
        (
            "qwen35moe.expert_feed_forward_length".into(),
            MetaValue::U32(moe_ffn),
        ),
        ("general.file_type".into(), MetaValue::U32(file_type)),
    ];

    // MRoPE dimension sections (always written; default per
    // `_Qwen35MRopeMixin._QWEN35_DEFAULT_MROPE_SECTION` qwen.py:526).
    let mrope: Vec<u32> = text
        .get("rope_parameters")
        .and_then(|rp| rp.get("mrope_section"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_u64().map(|n| n as u32))
                .collect()
        })
        .or_else(|| {
            text.get("mrope_section").and_then(|v| v.as_array()).map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as u32))
                    .collect()
            })
        })
        .unwrap_or_else(|| vec![11, 11, 10, 0]);
    kvs.push((
        "qwen35moe.rope.dimension_sections".into(),
        MetaValue::ArrayU32(mrope),
    ));

    // attn_output_gate (Qwen 3.5 attention quirk).
    if let Some(gate) = text.get("attn_output_gate").and_then(|v| v.as_bool()) {
        kvs.push((
            "qwen35moe.attention.output_gate".into(),
            MetaValue::Bool(gate),
        ));
    }

    // nextn_predict_layers — emitted only when MTP is present.
    if n_mtp > 0 {
        kvs.push((
            "qwen35moe.nextn_predict_layers".into(),
            MetaValue::U32(n_mtp),
        ));
    }

    let _ = ctx; // future per-arch metadata may need ctx fields
    kvs
}

/// Resolve the effective text-decoder config: nested `text_config`
/// for multimodal-wrapping `ConditionalGeneration` configs, otherwise
/// the root.
fn effective_text_config(config: &serde_json::Value) -> &serde_json::Value {
    config.get("text_config").unwrap_or(config)
}


#[cfg(test)]
mod tests {
    use super::*;

    fn vlm_ctx() -> Qwen35MoeFullCtx {
        // Sized to match operator's Qwen-Qwen3.5-35B-A3B multimodal
        // VLM: 80 hidden layers, 128 experts, moe_intermediate=768,
        // hidden=2048, linear: 16 K heads, 32 V heads (2 per K),
        // key_head_dim=128, value_head_dim=128.
        Qwen35MoeFullCtx {
            num_hidden_layers: 80,
            num_experts: 128,
            moe_intermediate_size: 768,
            hidden_size: 2048,
            linear_num_key_heads: 16,
            linear_num_value_heads: 32,
            linear_key_head_dim: 128,
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
        match map_tensor_name("model.language_model.embed_tokens.weight", &[], &ctx) {
            Some(MappedTensor::Direct(s)) => assert_eq!(s, "token_embd.weight"),
            other => panic!("unexpected: {other:?}"),
        }
        match map_tensor_name("model.language_model.norm.weight", &[], &ctx) {
            Some(MappedTensor::DirectWithBake { gguf_name, bake }) => {
                assert_eq!(gguf_name, "output_norm.weight");
                assert_eq!(bake, BakeOp::AddOne);
            }
            other => panic!("unexpected: {other:?}"),
        }
        match map_tensor_name("lm_head.weight", &[], &ctx) {
            Some(MappedTensor::Direct(s)) => assert_eq!(s, "output.weight"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn globals_text_only() {
        let ctx = text_only_ctx();
        match map_tensor_name("model.embed_tokens.weight", &[], &ctx) {
            Some(MappedTensor::Direct(s)) => assert_eq!(s, "token_embd.weight"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn visual_tensors_drop_in_multimodal() {
        let ctx = vlm_ctx();
        match map_tensor_name("model.visual.blocks.0.attn.qkv.weight", &[], &ctx) {
            Some(MappedTensor::Drop) => {}
            other => panic!("expected Drop, got: {other:?}"),
        }
    }

    #[test]
    fn input_layernorm_gets_plus_one_bake() {
        let ctx = vlm_ctx();
        match map_tensor_name("model.language_model.layers.5.input_layernorm.weight", &[], &ctx) {
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
        match map_tensor_name("model.language_model.layers.3.post_attention_layernorm.weight", &[], &ctx) {
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
        match map_tensor_name("model.language_model.layers.0.linear_attn.norm.weight", &[], &ctx) {
            Some(MappedTensor::Direct(s)) => assert_eq!(s, "blk.0.ssm_norm.weight"),
            other => panic!("expected Direct (no bake), got: {other:?}"),
        }
    }

    #[test]
    fn linear_attn_a_log_neg_exp() {
        let ctx = vlm_ctx();
        match map_tensor_name("model.language_model.layers.0.linear_attn.A_log", &[], &ctx) {
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
        match map_tensor_name("model.language_model.layers.0.linear_attn.conv1d.weight", &[], &ctx) {
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
        match map_tensor_name("model.language_model.layers.0.linear_attn.in_proj_a.weight", &[], &ctx) {
            Some(MappedTensor::Direct(s)) => assert_eq!(s, "blk.0.ssm_alpha.weight"),
            other => panic!("unexpected: {other:?}"),
        }
        match map_tensor_name("model.language_model.layers.0.linear_attn.in_proj_b.weight", &[], &ctx) {
            Some(MappedTensor::Direct(s)) => assert_eq!(s, "blk.0.ssm_beta.weight"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn linear_attn_in_proj_qkv_v_only_reorder_with_slice() {
        let ctx = vlm_ctx();
        // operator's config: nk=16, qd=kd=vd=128, nv_per_k=2, cols=2048.
        // shape[0] = nk*(qd+kd) + nk*nv_per_k*vd = 16*256 + 16*2*128 = 4096+4096 = 8192
        // shape[1] = cols = 2048
        let hf_shape = [8192_usize, 2048];
        match map_tensor_name(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            &hf_shape,
            &ctx,
        ) {
            Some(MappedTensor::DirectWithBake { gguf_name, bake }) => {
                assert_eq!(gguf_name, "blk.0.ssm_in_proj_qkv.weight");
                match bake {
                    BakeOp::ReorderVHeads {
                        num_k_heads,
                        num_v_per_k,
                        head_dim,
                        slice,
                    } => {
                        assert_eq!(num_k_heads, 16);
                        assert_eq!(num_v_per_k, 2);
                        assert_eq!(head_dim, 128 * 2048); // vd * cols
                        // V slice starts after Q+K rows:
                        // (qd+kd)*nk*cols = (128+128)*16*2048
                        let expect_start = (128 + 128) * 16 * 2048;
                        let expect_len = 2 * 128 * 16 * 2048;
                        let r = slice.expect("expected sliced reorder");
                        assert_eq!(r.start, expect_start);
                        assert_eq!(r.end - r.start, expect_len);
                    }
                    other => panic!("expected ReorderVHeads, got {other:?}"),
                }
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn linear_attn_in_proj_z_full_reorder_no_slice() {
        let ctx = vlm_ctx();
        // shape[0] = nk * nv_per_k * vd = 16*2*128 = 4096
        // shape[1] = cols = 2048
        let hf_shape = [4096_usize, 2048];
        match map_tensor_name(
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            &hf_shape,
            &ctx,
        ) {
            Some(MappedTensor::DirectWithBake { gguf_name, bake }) => {
                assert_eq!(gguf_name, "blk.0.ssm_in_proj_z.weight");
                match bake {
                    BakeOp::ReorderVHeads {
                        num_k_heads,
                        num_v_per_k,
                        head_dim,
                        slice,
                    } => {
                        assert_eq!(num_k_heads, 16);
                        assert_eq!(num_v_per_k, 2);
                        assert_eq!(head_dim, 128 * 2048);
                        assert!(slice.is_none(), "in_proj_z reorders full buffer");
                    }
                    other => panic!("expected ReorderVHeads, got {other:?}"),
                }
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn linear_attn_out_proj_per_row_col_reorder() {
        let ctx = vlm_ctx();
        // shape: [hidden, nk*nv_per_k*vd] = [2048, 4096]
        let hf_shape = [2048_usize, 4096];
        match map_tensor_name(
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            &hf_shape,
            &ctx,
        ) {
            Some(MappedTensor::DirectWithBake { gguf_name, bake }) => {
                assert_eq!(gguf_name, "blk.0.ssm_out.weight");
                match bake {
                    BakeOp::ReorderVHeadsPerRow {
                        row_count,
                        num_k_heads,
                        num_v_per_k,
                        head_dim_in_row,
                    } => {
                        assert_eq!(row_count, 2048);
                        assert_eq!(num_k_heads, 16);
                        assert_eq!(num_v_per_k, 2);
                        assert_eq!(head_dim_in_row, 128);
                    }
                    other => panic!("expected ReorderVHeadsPerRow, got {other:?}"),
                }
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn linear_attn_out_proj_rejects_mismatched_cols() {
        let ctx = vlm_ctx();
        // shape[1] != nk * nv_per_k * vd (16*2*128=4096); use bogus.
        let hf_shape = [2048_usize, 3000];
        assert!(map_tensor_name(
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            &hf_shape,
            &ctx,
        )
        .is_none());
    }

    #[test]
    fn linear_attn_v_reorder_arms_no_longer_surface_unmapped_with_shape() {
        // Regression: the three tensors that previously returned None
        // (in_proj_qkv, in_proj_z, out_proj — pre-`ReorderVHeadsPerRow`
        // + pre-`linear_key_head_dim`) now route through the right
        // bake op when shape is supplied. This test pins that fact.
        let ctx = vlm_ctx();
        // in_proj_qkv shape: [8192, 2048]
        assert!(map_tensor_name(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            &[8192, 2048],
            &ctx,
        )
        .is_some());
        // in_proj_z shape: [4096, 2048]
        assert!(map_tensor_name(
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            &[4096, 2048],
            &ctx,
        )
        .is_some());
        // out_proj shape: [2048, 4096]
        assert!(map_tensor_name(
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            &[2048, 4096],
            &ctx,
        )
        .is_some());
    }

    #[test]
    fn mlp_router_gate_direct() {
        let ctx = vlm_ctx();
        match map_tensor_name("model.language_model.layers.7.mlp.gate.weight", &[], &ctx) {
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
            match map_tensor_name(&hf, &[], &ctx) {
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
        match map_tensor_name("model.language_model.layers.9.mlp.experts.gate_up_proj", &[], &ctx) {
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
        match map_tensor_name("model.language_model.layers.9.mlp.experts.down_proj", &[], &ctx) {
            Some(MappedTensor::Direct(s)) => assert_eq!(s, "blk.9.ffn_down_exps.weight"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn mtp_helpers_remap_to_next_block_layer() {
        let ctx = vlm_ctx();
        // n_layer = 80, so MTP block lives at blk.80.
        match map_tensor_name("mtp.fc.weight", &[], &ctx) {
            Some(MappedTensor::Direct(s)) => assert_eq!(s, "blk.80.eh_proj.weight"),
            other => panic!("unexpected: {other:?}"),
        }
        match map_tensor_name("mtp.pre_fc_norm_embedding.weight", &[], &ctx) {
            Some(MappedTensor::DirectWithBake { gguf_name, bake }) => {
                assert_eq!(gguf_name, "blk.80.enorm.weight");
                assert_eq!(bake, BakeOp::AddOne);
            }
            other => panic!("unexpected: {other:?}"),
        }
        match map_tensor_name("mtp.pre_fc_norm_hidden.weight", &[], &ctx) {
            Some(MappedTensor::DirectWithBake { gguf_name, bake }) => {
                assert_eq!(gguf_name, "blk.80.hnorm.weight");
                assert_eq!(bake, BakeOp::AddOne);
            }
            other => panic!("unexpected: {other:?}"),
        }
        match map_tensor_name("mtp.norm.weight", &[], &ctx) {
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
        match map_tensor_name("mtp.layers.0.input_layernorm.weight", &[], &ctx) {
            Some(MappedTensor::DirectWithBake { gguf_name, bake }) => {
                assert_eq!(gguf_name, "blk.80.attn_norm.weight");
                assert_eq!(bake, BakeOp::AddOne);
            }
            other => panic!("unexpected: {other:?}"),
        }
        // mtp.layers.0.self_attn.q_proj.weight → blk.80.attn_q.weight
        match map_tensor_name("mtp.layers.0.self_attn.q_proj.weight", &[], &ctx) {
            Some(MappedTensor::Direct(s)) => assert_eq!(s, "blk.80.attn_q.weight"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn unmapped_name_returns_none() {
        let ctx = vlm_ctx();
        assert!(map_tensor_name("model.totally_made_up_tensor.weight", &[], &ctx).is_none());
        assert!(map_tensor_name("garbage", &[], &ctx).is_none());
    }
}
