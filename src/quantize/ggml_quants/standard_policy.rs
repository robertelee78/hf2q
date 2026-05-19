//! `StandardPolicy` — port of llama.cpp's standard quant-target picker.
//!
//! Per ADR-033 §"Plan" / P1 and Decision §"QuantPolicy trait (Decision
//! §3 concrete)". Implements three pieces:
//!
//! 1. **`tensor_type_fallback`** — port of
//!    `/opt/llama.cpp/src/llama-quant.cpp:362-408` (the row-misalignment
//!    first-downshift table). Per ADR §"shape_fallback contract" the
//!    second-misalignment case returns `Err` instead of silently
//!    demoting to F16.
//!
//! 2. **`QsState`** — Rust mirror of llama.cpp's `quantize_state_impl`
//!    (`llama-quant.cpp:163-196`). Carries the per-(ftype,arch,model)
//!    state plus the MUT counters (`i_attention_wv` etc.) that
//!    `target_for` increments as it walks the tensor list.
//!
//! 3. **`StandardPolicy::target_for`** — port of
//!    `llama-quant.cpp:411-657` (`llama_tensor_get_type_impl`). The
//!    big ~247-LOC function with per-`(ftype, name, arch, category)`
//!    branching. C-line comments throughout document the source
//!    mapping; branch order, comparison order, and side-effects are
//!    mirrored 1:1 so the downstream P1 byte-cmp gate has a chance.

use super::error::QuantizeError;
use super::ggml_type::GgmlType;
use super::llama_ftype::LlamaFtype;
use super::tensor_ref::{ArchName, TensorRef};

/// Mirrors llama.cpp's `tensor_category` enum
/// (`/opt/llama.cpp/src/llama-quant.cpp:25-38, 115-150`). Decides which
/// `llama_tensor_get_type_impl` branch fires for a given tensor.
///
/// Per C:115 `tensor_get_category` does **substring** matches (not
/// suffix-only); we keep that semantic here so MoE `_exps` variants
/// land in the right category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorCategory {
    /// `output.weight`.
    Output,
    /// `token_embd.weight` / `per_layer_token_embd.weight`.
    TokenEmbd,
    /// `*attn_v.weight*`.
    AttnV,
    /// `*attn_q.weight*`.
    AttnQ,
    /// `*attn_k.weight*`.
    AttnK,
    /// `*attn_qkv.weight*` — fused QKV.
    AttnQkv,
    /// `*attn_kv_b.weight*` — DeepSeek2 fused KV-B.
    AttnKvB,
    /// `*attn_output.weight*`.
    AttnOutput,
    /// `*ffn_up*` (substring).
    FfnUp,
    /// `*ffn_gate*` (substring).
    FfnGate,
    /// `*ffn_down*` (substring).
    FfnDown,
    /// Catch-all (norms, biases, etc.).
    Other,
}

impl TensorCategory {
    /// Classify a tensor by name. Mirrors `tensor_get_category`
    /// (`llama-quant.cpp:115-150`) — note the priority order is
    /// significant (OUTPUT and TOKEN_EMBD checked first, then the QKV /
    /// KV_B fused variants before V/K/Q so the longer names win).
    pub fn classify(name: &str) -> Self {
        // C:116-117 — output.weight (exact).
        if name == "output.weight" {
            return TensorCategory::Output;
        }
        // C:119-120 — token_embd.weight or per_layer_token_embd.weight.
        if name == "token_embd.weight" || name == "per_layer_token_embd.weight" {
            return TensorCategory::TokenEmbd;
        }
        // C:122-138 — substring matches, priority order matters.
        if name.contains("attn_qkv.weight") {
            return TensorCategory::AttnQkv;
        }
        if name.contains("attn_kv_b.weight") {
            return TensorCategory::AttnKvB;
        }
        if name.contains("attn_v.weight") {
            return TensorCategory::AttnV;
        }
        if name.contains("attn_k.weight") {
            return TensorCategory::AttnK;
        }
        if name.contains("attn_q.weight") {
            return TensorCategory::AttnQ;
        }
        if name.contains("attn_output.weight") {
            return TensorCategory::AttnOutput;
        }
        // C:140-148 — ffn_up / ffn_gate / ffn_down substring matches.
        // C order: ffn_up THEN ffn_gate THEN ffn_down. We mirror it
        // even though they don't share prefixes today.
        if name.contains("ffn_up") {
            return TensorCategory::FfnUp;
        }
        if name.contains("ffn_gate") {
            return TensorCategory::FfnGate;
        }
        if name.contains("ffn_down") {
            return TensorCategory::FfnDown;
        }
        TensorCategory::Other
    }

    /// `category_is_attn_v` — port of `llama-quant.cpp:153-157`.
    /// Returns true for ATTENTION_V, ATTENTION_QKV, or ATTENTION_KV_B
    /// (all three are "more sensitive to quantization").
    pub const fn is_attn_v(self) -> bool {
        matches!(
            self,
            TensorCategory::AttnV | TensorCategory::AttnQkv | TensorCategory::AttnKvB
        )
    }
}

/// `tensor_type_fallback` — port of
/// `/opt/llama.cpp/src/llama-quant.cpp:362-408`.
///
/// If `n_per_row % target.block_size() == 0` the target type is
/// returned unchanged. Otherwise the first-downshift table picks a
/// 32-aligned legacy variant of the same family:
///
/// | original  | first-downshift |
/// |-----------|-----------------|
/// | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS | IQ4_NL |
/// | Q2_K, Q3_K, TQ1_0, TQ2_0 | Q4_0 |
/// | Q4_K      | Q5_0            |
/// | Q5_K      | Q5_1            |
/// | Q6_K      | Q8_0            |
///
/// **Per ADR §"shape_fallback contract"**: if the downshift's
/// `block_size` still doesn't divide `n_per_row` (the C source's
/// "second misalignment" case where llama.cpp silently demotes to
/// F16), we return `Err(QuantizeError::NotBlockAligned)` instead.
pub fn tensor_type_fallback(
    target: GgmlType,
    n_per_row: usize,
) -> Result<GgmlType, QuantizeError> {
    let target_block = target.block_size();
    if n_per_row % target_block == 0 {
        return Ok(target);
    }

    // First-downshift table per C:373-393.
    let downshift = match target {
        // C:375-382 — block-size-256 IQ family → IQ4_NL (block 32).
        GgmlType::IQ1_S
        | GgmlType::IQ1_M
        | GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ2_S
        | GgmlType::IQ3_XXS
        | GgmlType::IQ3_S
        | GgmlType::IQ4_XS => GgmlType::IQ4_NL,
        // C:383-386 — Q2_K, Q3_K, TQ1_0, TQ2_0 → Q4_0.
        GgmlType::Q2_K | GgmlType::Q3_K | GgmlType::TQ1_0 | GgmlType::TQ2_0 => GgmlType::Q4_0,
        // C:387 Q4_K → Q5_0.
        GgmlType::Q4_K => GgmlType::Q5_0,
        // C:388 Q5_K → Q5_1.
        GgmlType::Q5_K => GgmlType::Q5_1,
        // C:389 Q6_K → Q8_0.
        GgmlType::Q6_K => GgmlType::Q8_0,
        // C:390-392 default — runtime error in C. Here a typed error.
        _ => {
            return Err(QuantizeError::NotBlockAligned {
                ggml_type: target,
                n_per_row,
                block_size: target_block,
            });
        }
    };

    if n_per_row % downshift.block_size() != 0 {
        // C:394-403 falls back to F16 here. ADR §"shape_fallback
        // contract" makes this a typed error: F16 is reserved for
        // vision/audio tensors and explicit `--quant f16`.
        return Err(QuantizeError::NotBlockAligned {
            ggml_type: downshift,
            n_per_row,
            block_size: downshift.block_size(),
        });
    }

    Ok(downshift)
}

/// Hparams subset consumed by `target_for`.
///
/// Mirrors the fields `llama_tensor_get_type_impl` accesses on
/// `qs.model.hparams`: `n_expert`, `n_gqa()`, and (indirectly via
/// `n_gqa`) `n_head` / `n_head_kv`.
#[derive(Debug, Clone, Copy)]
pub struct HParams {
    /// `qs.model.hparams.n_expert` — number of experts. 0 for dense.
    pub n_expert: u32,
    /// `qs.model.hparams.n_head(il=0)` — number of attention heads.
    pub n_head: u32,
    /// `qs.model.hparams.n_head_kv(il=0)` — number of KV heads.
    pub n_head_kv: u32,
}

impl HParams {
    /// `n_gqa` — port of `llama-hparams.cpp:54-63`.
    /// `il = 0` slice (target_for never passes a layer index).
    pub const fn n_gqa(&self) -> u32 {
        if self.n_head_kv == 0 {
            return 0;
        }
        self.n_head / self.n_head_kv
    }
}

/// Subset of `llama_model::type` consumed by `target_for`.
///
/// Only one variant is referenced inside the function (`LLM_TYPE_70B`
/// at C:537). Other sizes collapse to `Other` since `target_for` only
/// branches on `== LLM_TYPE_70B`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmType {
    /// `LLM_TYPE_70B` — triggers the attn_v Q5_K bump at C:537-542.
    M70B,
    /// Any other size — target_for doesn't differentiate.
    Other,
}

/// Subset of `llama_model_quantize_params` consumed by `target_for`.
///
/// Mirrors the two override fields referenced inside the function:
/// `output_tensor_type` (C:440-441) and `token_embedding_type` (C:470-471).
/// Both are `Option<GgmlType>` — `None` mirrors C's `GGML_TYPE_COUNT`
/// sentinel ("no override specified").
#[derive(Debug, Clone, Copy, Default)]
pub struct QuantizeParams {
    pub output_tensor_type: Option<GgmlType>,
    pub token_embedding_type: Option<GgmlType>,
}

/// `QsState` — Rust mirror of llama.cpp's `quantize_state_impl`
/// (`llama-quant.cpp:163-196`).
///
/// Carries the per-conversion state that `target_for` reads and the
/// counters it MUTATES (`i_attention_wv` etc.). Constructed once at
/// the start of `llama_model_quantize_impl`, then passed by `&mut` to
/// each `target_for` call.
#[derive(Debug, Clone)]
pub struct QsState {
    pub ftype: LlamaFtype,
    pub arch: ArchName,
    pub model_type: LlmType,
    pub hparams: HParams,
    pub params: QuantizeParams,

    /// True when `output.weight` is tied to `token_embd.weight`. Set in
    /// C as `has_tied_embeddings = true` (C:181) and cleared the moment
    /// the loop encounters `output.weight`.
    pub has_tied_embeddings: bool,

    /// True when an imatrix is provided. Per C:178.
    pub has_imatrix: bool,

    // --- counters (MUT — incremented inside target_for) ---
    pub n_attention_wv: i32,
    pub i_attention_wv: i32,
    pub n_ffn_down: i32,
    pub i_ffn_down: i32,
    pub n_ffn_gate: i32,
    pub i_ffn_gate: i32,
    pub n_ffn_up: i32,
    pub i_ffn_up: i32,
}

impl QsState {
    /// Construct a `QsState` with all counters at zero. The convert
    /// driver populates `n_attention_wv` / `n_ffn_down` / etc. with the
    /// per-arch counts BEFORE running the per-tensor loop, then calls
    /// `target_for` on each tensor (which increments the matching `i_*`).
    pub const fn new(
        ftype: LlamaFtype,
        arch: ArchName,
        model_type: LlmType,
        hparams: HParams,
    ) -> Self {
        Self {
            ftype,
            arch,
            model_type,
            hparams,
            params: QuantizeParams {
                output_tensor_type: None,
                token_embedding_type: None,
            },
            has_tied_embeddings: true,
            has_imatrix: false,
            n_attention_wv: 0,
            i_attention_wv: 0,
            n_ffn_down: 0,
            i_ffn_down: 0,
            n_ffn_gate: 0,
            i_ffn_gate: 0,
            n_ffn_up: 0,
            i_ffn_up: 0,
        }
    }
}

/// `use_more_bits` — port of the lambda at `llama-quant.cpp:417-419`.
/// Returns true for the layer ranges that should get the higher-precision
/// variant (first eighth, last eighth, plus every third layer in the middle).
#[inline]
const fn use_more_bits(i_layer: i32, n_layers: i32) -> bool {
    i_layer < n_layers / 8
        || i_layer >= 7 * n_layers / 8
        || (i_layer - n_layers / 8) % 3 == 2
}

/// `layer_info` — port of the lambda at `llama-quant.cpp:421-435`.
///
/// For dense (`n_expert <= 1`) models the layer index passed in is
/// trusted (it's `qs.i_ffn_down` etc.). For MoE (`n_expert > 1`), expert
/// tensors are sprinkled non-consecutively, so the C code re-parses
/// `blk.<N>.` out of the name. Returns `(i_layer, n_layer)`.
fn layer_info(i_layer: i32, n_layer: i32, name: &str, n_expert: i32) -> Result<(i32, i32), QuantizeError> {
    if n_expert > 1 {
        // C:427-432 — parse `blk.<i>.` out of the name.
        let parsed = name
            .strip_prefix("blk.")
            .and_then(|rest| {
                let dot = rest.find('.')?;
                rest[..dot].parse::<i32>().ok()
            });
        let parsed = match parsed {
            Some(v) => v,
            None => {
                return Err(QuantizeError::BadLayerForTensor {
                    name: name.to_string(),
                    n_layer,
                });
            }
        };
        if parsed < 0 || parsed >= n_layer {
            return Err(QuantizeError::BadLayerForTensor {
                name: name.to_string(),
                n_layer,
            });
        }
        Ok((parsed, n_layer))
    } else {
        Ok((i_layer, n_layer))
    }
}

/// `StandardPolicy` — picks `GgmlType` per tensor at a given `LlamaFtype`.
/// Per ADR-033 Decision §"QuantPolicy trait (Decision §3 concrete)".
///
/// The body of `target_for` is a 1:1 port of `llama_tensor_get_type_impl`
/// at SHA `c779f6198` (`data/llama_cpp_pin.txt`), `llama-quant.cpp:411-657`.
pub struct StandardPolicy;

impl StandardPolicy {
    pub const fn new() -> Self {
        Self
    }

    /// Pick a `GgmlType` for `tensor` at this policy's `qs.ftype`.
    ///
    /// Faithful port of `llama_tensor_get_type_impl`
    /// (`/opt/llama.cpp/src/llama-quant.cpp:411-657`). Mirrors:
    /// - branch order
    /// - counter side-effects (`++qs.i_attention_wv` etc.)
    /// - comparison order (relevant for FP-equality-adjacent branches)
    ///
    /// Final step is `tensor_type_fallback(picked, n_per_row)` which
    /// applies the shape-misalignment downshift (or returns
    /// `Err(NotBlockAligned)` per ADR §"shape_fallback contract").
    pub fn target_for(
        &self,
        qs: &mut QsState,
        tensor: &TensorRef,
        category: TensorCategory,
    ) -> Result<GgmlType, QuantizeError> {
        // C:411-415 — name, arch.
        let name = tensor.name;
        let arch = qs.arch;
        let ftype = qs.ftype;

        // Start from the ftype's primary (i.e. `default_type` parameter
        // in C — see `llama_model_quantize_impl`'s default-type pick).
        let mut new_type = ftype.primary_type();

        // C:420 — n_expert (clamped to >= 1 for the layer_info lambda).
        let n_expert = (qs.hparams.n_expert as i32).max(1);

        // ------------------------------------------------------------------
        // C:437-486 — OUTPUT / (tied-)TOKEN_EMBD special-case branch.
        // ------------------------------------------------------------------
        if category == TensorCategory::Output
            || (qs.has_tied_embeddings && category == TensorCategory::TokenEmbd)
        {
            if let Some(t) = qs.params.output_tensor_type {
                // C:440-441 — explicit override wins.
                new_type = t;
            } else {
                // C:443-444 — nx (= ne[0]), qk_k (= block_size(new_type)).
                let nx = tensor.n_per_row() as i64;
                let qk_k = new_type.block_size() as i64;

                if ftype == LlamaFtype::MostlyMXFP4_MOE {
                    // C:446-448
                    new_type = GgmlType::Q8_0;
                } else if arch == ArchName::Falcon || nx % qk_k != 0 {
                    // C:449-451
                    new_type = GgmlType::Q8_0;
                } else if matches!(
                    ftype,
                    LlamaFtype::MostlyIQ2_XXS
                        | LlamaFtype::MostlyIQ2_XS
                        | LlamaFtype::MostlyIQ3_XXS
                        | LlamaFtype::MostlyIQ1_S
                        | LlamaFtype::MostlyIQ2_S
                        | LlamaFtype::MostlyIQ2_M
                        | LlamaFtype::MostlyIQ1_M
                ) {
                    // C:452-456
                    new_type = GgmlType::Q5_K;
                } else if new_type != GgmlType::Q8_0 {
                    // C:457-459
                    new_type = GgmlType::Q6_K;
                }
            }
        } else if ftype == LlamaFtype::MostlyMXFP4_MOE {
            // C:461-468 — MoE tensors → MXFP4 (3D), other → Q8_0.
            // We use ne[2]>1 as the 3D test. TensorRef stores row-major
            // shape; `shape.len() > 2 && shape[2] > 1` mirrors C's
            // `tensor->ne[2] > 1`.
            let is_3d = tensor.shape.len() > 2 && tensor.shape[2] > 1;
            new_type = if is_3d { GgmlType::MXFP4 } else { GgmlType::Q8_0 };
        } else if category == TensorCategory::TokenEmbd {
            // C:469-486 — non-tied TOKEN_EMBD path.
            if let Some(t) = qs.params.token_embedding_type {
                new_type = t;
            } else if matches!(
                ftype,
                LlamaFtype::MostlyIQ2_XXS
                    | LlamaFtype::MostlyIQ2_XS
                    | LlamaFtype::MostlyIQ1_S
                    | LlamaFtype::MostlyIQ1_M
            ) {
                // C:473-476
                new_type = GgmlType::Q2_K;
            } else if matches!(ftype, LlamaFtype::MostlyIQ2_S | LlamaFtype::MostlyIQ2_M) {
                // C:477-479
                new_type = GgmlType::IQ3_S;
            } else if ftype == LlamaFtype::MostlyIQ3_XXS {
                // C:480-482
                new_type = GgmlType::IQ3_S;
            } else if matches!(ftype, LlamaFtype::MostlyTQ1_0 | LlamaFtype::MostlyTQ2_0) {
                // C:483-485
                new_type = GgmlType::Q4_K;
            }
        } else if matches!(
            ftype,
            LlamaFtype::MostlyIQ2_XXS
                | LlamaFtype::MostlyIQ2_XS
                | LlamaFtype::MostlyIQ1_S
                | LlamaFtype::MostlyIQ2_S
                | LlamaFtype::MostlyIQ2_M
                | LlamaFtype::MostlyIQ1_M
        ) {
            // C:487-510 — IQ1/IQ2 super-low-bit ftype family.
            if category.is_attn_v() {
                // C:489-492
                if qs.hparams.n_gqa() >= 4 || qs.hparams.n_expert >= 4 {
                    new_type = GgmlType::Q4_K;
                } else if matches!(ftype, LlamaFtype::MostlyIQ2_S | LlamaFtype::MostlyIQ2_M) {
                    new_type = GgmlType::IQ3_S;
                } else {
                    new_type = GgmlType::Q2_K;
                }
                qs.i_attention_wv += 1;
            } else if qs.hparams.n_expert == 8 && category == TensorCategory::AttnK {
                // C:494-496
                new_type = GgmlType::Q4_K;
            } else if category == TensorCategory::FfnDown {
                // C:497-502
                if qs.i_ffn_down < qs.n_ffn_down / 8 {
                    new_type = if matches!(ftype, LlamaFtype::MostlyIQ2_S | LlamaFtype::MostlyIQ2_M)
                    {
                        GgmlType::IQ3_S
                    } else {
                        GgmlType::Q2_K
                    };
                }
                qs.i_ffn_down += 1;
            } else if category == TensorCategory::AttnOutput {
                // C:503-510
                if qs.hparams.n_expert == 8 {
                    new_type = GgmlType::Q5_K;
                } else if matches!(ftype, LlamaFtype::MostlyIQ1_S | LlamaFtype::MostlyIQ1_M) {
                    new_type = GgmlType::IQ2_XXS;
                } else if matches!(ftype, LlamaFtype::MostlyIQ2_S | LlamaFtype::MostlyIQ2_M) {
                    new_type = GgmlType::IQ3_S;
                }
            }
        } else if category.is_attn_v() {
            // C:511-548 — generic attn_v / attn_qkv / attn_kv_b branch.
            if ftype == LlamaFtype::MostlyQ2_K {
                // C:512-514
                new_type = if qs.hparams.n_gqa() >= 4 {
                    GgmlType::Q4_K
                } else {
                    GgmlType::Q3_K
                };
            } else if ftype == LlamaFtype::MostlyQ2_K_S && qs.hparams.n_gqa() >= 4 {
                // C:515-517
                new_type = GgmlType::Q4_K;
            } else if ftype == LlamaFtype::MostlyIQ3_XXS {
                // C:518-520
                new_type = if qs.hparams.n_gqa() >= 4 {
                    GgmlType::Q4_K
                } else if !qs.has_imatrix {
                    GgmlType::IQ3_S
                } else {
                    GgmlType::IQ3_XXS
                };
            } else if matches!(ftype, LlamaFtype::MostlyIQ3_XS | LlamaFtype::MostlyIQ3_S)
                && qs.hparams.n_gqa() >= 4
            {
                // C:521-523
                new_type = GgmlType::Q4_K;
            } else if ftype == LlamaFtype::MostlyIQ3_M {
                // C:524-526
                new_type = GgmlType::Q4_K;
            } else if ftype == LlamaFtype::MostlyQ3_K_M {
                // C:527-529
                new_type = if qs.i_attention_wv < 2 {
                    GgmlType::Q5_K
                } else {
                    GgmlType::Q4_K
                };
            } else if ftype == LlamaFtype::MostlyQ3_K_L {
                // C:530
                new_type = GgmlType::Q5_K;
            } else if matches!(ftype, LlamaFtype::MostlyIQ4_NL | LlamaFtype::MostlyIQ4_XS)
                && qs.hparams.n_gqa() >= 4
            {
                // C:531-533
                new_type = GgmlType::Q5_K;
            } else if matches!(ftype, LlamaFtype::MostlyQ4_K_M | LlamaFtype::MostlyQ5_K_M)
                && use_more_bits(qs.i_attention_wv, qs.n_attention_wv)
            {
                // C:534-535
                new_type = GgmlType::Q6_K;
            } else if ftype == LlamaFtype::MostlyQ4_K_S && qs.i_attention_wv < 4 {
                // C:536
                new_type = GgmlType::Q5_K;
            }
            // C:537-542 — 70B attn_v bump.
            if qs.model_type == LlmType::M70B
                && matches!(new_type, GgmlType::Q3_K | GgmlType::Q4_K)
            {
                new_type = GgmlType::Q5_K;
            }
            // C:543-547 — 8-expert MoE attn_v bump.
            if qs.hparams.n_expert == 8 {
                new_type = GgmlType::Q8_0;
            }
            qs.i_attention_wv += 1;
        } else if category == TensorCategory::AttnK {
            // C:549-560 — attn_k branch.
            if qs.hparams.n_expert == 8 {
                // C:550-554
                new_type = GgmlType::Q8_0;
            } else if ftype == LlamaFtype::MostlyIQ3_XS {
                // C:555-557
                new_type = GgmlType::IQ3_XXS;
            } else if ftype == LlamaFtype::MostlyIQ3_XXS {
                // C:558-560
                new_type = GgmlType::IQ2_S;
            }
        } else if category == TensorCategory::AttnQ {
            // C:561-567 — attn_q branch.
            if ftype == LlamaFtype::MostlyIQ3_XS {
                // C:562-564
                new_type = GgmlType::IQ3_XXS;
            } else if ftype == LlamaFtype::MostlyIQ3_XXS {
                // C:565-567
                new_type = GgmlType::IQ2_S;
            }
        } else if category == TensorCategory::FfnDown {
            // C:568-612 — ffn_down branch.
            let (i_layer, n_layer) =
                layer_info(qs.i_ffn_down, qs.n_ffn_down, name, n_expert)?;

            if ftype == LlamaFtype::MostlyQ2_K {
                // C:571
                new_type = GgmlType::Q3_K;
            } else if ftype == LlamaFtype::MostlyQ2_K_S {
                // C:572-574
                if i_layer < n_layer / 8 {
                    new_type = GgmlType::Q4_K;
                }
            } else if ftype == LlamaFtype::MostlyIQ3_XXS && !qs.has_imatrix {
                // C:575-577
                new_type = if i_layer < n_layer / 8 {
                    GgmlType::Q4_K
                } else {
                    GgmlType::Q3_K
                };
            } else if ftype == LlamaFtype::MostlyQ3_K_M {
                // C:578-582 — three-way ternary on layer/arch.
                new_type = if i_layer < n_layer / 16 {
                    GgmlType::Q5_K
                } else if arch != ArchName::Falcon || use_more_bits(i_layer, n_layer) {
                    GgmlType::Q4_K
                } else {
                    GgmlType::Q3_K
                };
            } else if ftype == LlamaFtype::MostlyIQ3_M
                && (i_layer < n_layer / 8
                    || (qs.hparams.n_expert == 8 && use_more_bits(i_layer, n_layer)))
            {
                // C:583-586
                new_type = GgmlType::Q4_K;
            } else if ftype == LlamaFtype::MostlyQ3_K_L {
                // C:587-589
                new_type = if arch == ArchName::Falcon {
                    GgmlType::Q4_K
                } else {
                    GgmlType::Q5_K
                };
            } else if ftype == LlamaFtype::MostlyQ4_K_M {
                // C:590-597
                if arch == ArchName::Falcon {
                    new_type = if i_layer < n_layer / 16 {
                        GgmlType::Q6_K
                    } else if use_more_bits(i_layer, n_layer) {
                        GgmlType::Q5_K
                    } else {
                        GgmlType::Q4_K
                    };
                } else if use_more_bits(i_layer, n_layer) {
                    new_type = GgmlType::Q6_K;
                }
            } else if i_layer < n_layer / 8
                && matches!(ftype, LlamaFtype::MostlyIQ4_NL | LlamaFtype::MostlyIQ4_XS)
                && !qs.has_imatrix
            {
                // C:598-600
                new_type = GgmlType::Q5_K;
            } else if ftype == LlamaFtype::MostlyQ5_K_M && use_more_bits(i_layer, n_layer) {
                // C:601
                new_type = GgmlType::Q6_K;
            } else if ftype == LlamaFtype::MostlyQ4_K_S
                && arch != ArchName::Falcon
                && i_layer < n_layer / 8
            {
                // C:602-604
                new_type = GgmlType::Q5_K;
            } else if matches!(ftype, LlamaFtype::MostlyQ4_0 | LlamaFtype::MostlyQ5_0)
                && qs.has_imatrix
                && i_layer < n_layer / 8
            {
                // C:605-611 — imatrix-only ffn_down guard.
                new_type = if ftype == LlamaFtype::MostlyQ4_0 {
                    GgmlType::Q4_1
                } else {
                    GgmlType::Q5_1
                };
            }
            qs.i_ffn_down += 1;
        } else if category == TensorCategory::AttnOutput {
            // C:613-632 — attn_output branch.
            if arch != ArchName::Falcon {
                if qs.hparams.n_expert == 8 {
                    // C:615-621 — 8-expert MoE bumps a wide ftype list to Q5_K.
                    if matches!(
                        ftype,
                        LlamaFtype::MostlyQ2_K
                            | LlamaFtype::MostlyIQ3_XS
                            | LlamaFtype::MostlyIQ3_XXS
                            | LlamaFtype::MostlyQ3_K_S
                            | LlamaFtype::MostlyQ3_K_M
                            | LlamaFtype::MostlyIQ4_NL
                            | LlamaFtype::MostlyQ4_K_S
                            | LlamaFtype::MostlyQ4_K_M
                            | LlamaFtype::MostlyIQ3_S
                            | LlamaFtype::MostlyIQ3_M
                            | LlamaFtype::MostlyIQ4_XS
                    ) {
                        new_type = GgmlType::Q5_K;
                    }
                } else {
                    // C:622-628
                    if ftype == LlamaFtype::MostlyQ2_K {
                        new_type = GgmlType::Q3_K;
                    } else if ftype == LlamaFtype::MostlyIQ3_XXS {
                        new_type = GgmlType::IQ3_S;
                    } else if ftype == LlamaFtype::MostlyQ3_K_M {
                        new_type = GgmlType::Q4_K;
                    } else if ftype == LlamaFtype::MostlyQ3_K_L {
                        new_type = GgmlType::Q5_K;
                    } else if ftype == LlamaFtype::MostlyIQ3_M {
                        new_type = GgmlType::Q4_K;
                    }
                }
            } else if ftype == LlamaFtype::MostlyQ3_K_L {
                // C:629-631 — Falcon attn_output @ Q3_K_L → Q4_K.
                new_type = GgmlType::Q4_K;
            }
        } else if category == TensorCategory::AttnQkv {
            // C:633-639 — fused QKV branch.
            if matches!(
                ftype,
                LlamaFtype::MostlyQ3_K_M | LlamaFtype::MostlyQ3_K_L | LlamaFtype::MostlyIQ3_M
            ) {
                new_type = GgmlType::Q4_K;
            } else if ftype == LlamaFtype::MostlyQ4_K_M {
                new_type = GgmlType::Q5_K;
            } else if ftype == LlamaFtype::MostlyQ5_K_M {
                new_type = GgmlType::Q6_K;
            }
        } else if category == TensorCategory::FfnGate {
            // C:640-647 — ffn_gate branch.
            let (i_layer, n_layer) =
                layer_info(qs.i_ffn_gate, qs.n_ffn_gate, name, n_expert)?;
            if ftype == LlamaFtype::MostlyIQ3_XS
                && i_layer >= n_layer / 8
                && i_layer < 7 * n_layer / 8
            {
                new_type = GgmlType::IQ3_XXS;
            }
            qs.i_ffn_gate += 1;
        } else if category == TensorCategory::FfnUp {
            // C:648-655 — ffn_up branch.
            let (i_layer, n_layer) =
                layer_info(qs.i_ffn_up, qs.n_ffn_up, name, n_expert)?;
            if ftype == LlamaFtype::MostlyIQ3_XS
                && i_layer >= n_layer / 8
                && i_layer < 7 * n_layer / 8
            {
                new_type = GgmlType::IQ3_XXS;
            }
            qs.i_ffn_up += 1;
        }

        // C:657 — return. Apply ADR §"shape_fallback contract" before
        // handing back to the caller (this corresponds to the
        // `tensor_type_fallback` call made by the OUTER wrapper at C:728
        // — see `llama_tensor_get_type` — which we collapse into the
        // same `target_for` Result here).
        tensor_type_fallback(new_type, tensor.n_per_row())
    }
}

impl Default for StandardPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::tensor_ref::{ArchName, SourceDtype};
    use super::*;

    fn mk_qs(ftype: LlamaFtype, arch: ArchName) -> QsState {
        // Default-ish hparams: 32 heads, 8 kv-heads → n_gqa = 4 (the
        // canonical Llama-3 8B shape; tests that need different shapes
        // override fields after construction).
        let hparams = HParams {
            n_expert: 0,
            n_head: 32,
            n_head_kv: 8,
        };
        QsState::new(ftype, arch, LlmType::Other, hparams)
    }

    fn mk_tensor<'a>(name: &'a str, shape: &'a [usize], arch: ArchName) -> TensorRef<'a> {
        TensorRef {
            name,
            shape,
            source_dtype: SourceDtype::BF16,
            arch,
            layer_index: None,
        }
    }

    #[test]
    fn passthrough_f32() {
        let mut qs = mk_qs(LlamaFtype::AllF32, ArchName::Llama3);
        let shape = [4096, 1];
        let t = mk_tensor("blk.0.attn_q.weight", &shape, ArchName::Llama3);
        let pol = StandardPolicy::new();
        let cat = TensorCategory::classify(t.name);
        assert_eq!(pol.target_for(&mut qs, &t, cat).unwrap(), GgmlType::F32);
    }

    #[test]
    fn passthrough_f16() {
        let mut qs = mk_qs(LlamaFtype::MostlyF16, ArchName::Llama3);
        let shape = [4096, 1];
        let t = mk_tensor("blk.0.attn_q.weight", &shape, ArchName::Llama3);
        let pol = StandardPolicy::new();
        let cat = TensorCategory::classify(t.name);
        assert_eq!(pol.target_for(&mut qs, &t, cat).unwrap(), GgmlType::F16);
    }

    #[test]
    fn passthrough_bf16() {
        let mut qs = mk_qs(LlamaFtype::BF16, ArchName::Llama3);
        let shape = [4096, 1];
        let t = mk_tensor("blk.0.attn_q.weight", &shape, ArchName::Llama3);
        let pol = StandardPolicy::new();
        let cat = TensorCategory::classify(t.name);
        assert_eq!(pol.target_for(&mut qs, &t, cat).unwrap(), GgmlType::BF16);
    }

    #[test]
    fn output_q5_k_m_bumps_to_q6_k() {
        // C:457-459 — at Q5_K_M, output.weight bumps to Q6_K (because
        // new_type != Q8_0).
        let mut qs = mk_qs(LlamaFtype::MostlyQ5_K_M, ArchName::Llama3);
        let shape = [4096, 1];
        let t = mk_tensor("output.weight", &shape, ArchName::Llama3);
        let pol = StandardPolicy::new();
        assert_eq!(
            pol.target_for(&mut qs, &t, TensorCategory::Output).unwrap(),
            GgmlType::Q6_K
        );
    }

    #[test]
    fn token_embd_q4_k_s_stays_q4_k() {
        // No `token_embedding_type` override; ftype = Q4_K_S; not
        // IQ-family and not TQ; so the C:469-486 branch leaves new_type
        // at primary = Q4_K. `has_tied_embeddings` is true by default so
        // it would route to the OUTPUT branch — set false to test the
        // standalone TOKEN_EMBD branch.
        let mut qs = mk_qs(LlamaFtype::MostlyQ4_K_S, ArchName::Llama3);
        qs.has_tied_embeddings = false;
        let shape = [4096, 1];
        let t = mk_tensor("token_embd.weight", &shape, ArchName::Llama3);
        let pol = StandardPolicy::new();
        assert_eq!(
            pol.target_for(&mut qs, &t, TensorCategory::TokenEmbd).unwrap(),
            GgmlType::Q4_K
        );
    }

    #[test]
    fn token_embd_tied_routes_to_output_branch() {
        // With has_tied_embeddings=true (default), token_embd.weight at
        // Q5_K_M should hit the OUTPUT branch and bump to Q6_K (C:439).
        let mut qs = mk_qs(LlamaFtype::MostlyQ5_K_M, ArchName::Llama3);
        let shape = [4096, 1];
        let t = mk_tensor("token_embd.weight", &shape, ArchName::Llama3);
        let pol = StandardPolicy::new();
        assert_eq!(
            pol.target_for(&mut qs, &t, TensorCategory::TokenEmbd).unwrap(),
            GgmlType::Q6_K
        );
    }

    #[test]
    fn attn_v_q5_k_m_first_layers_bump_q6_k() {
        // C:534-535 — Q5_K_M + use_more_bits(i_attention_wv,
        // n_attention_wv) → Q6_K. With i_attention_wv=0 and
        // n_attention_wv=32, use_more_bits(0,32) is true (0 < 32/8 = 4).
        let mut qs = mk_qs(LlamaFtype::MostlyQ5_K_M, ArchName::Llama3);
        qs.n_attention_wv = 32;
        qs.i_attention_wv = 0;
        let shape = [1024, 1];
        let t = mk_tensor("blk.0.attn_v.weight", &shape, ArchName::Llama3);
        let pol = StandardPolicy::new();
        assert_eq!(
            pol.target_for(&mut qs, &t, TensorCategory::AttnV).unwrap(),
            GgmlType::Q6_K
        );
        assert_eq!(qs.i_attention_wv, 1, "counter must be incremented per C:548");
    }

    #[test]
    fn ffn_down_q4_k_m_first_layers_bump_q6_k() {
        // C:590-597 (non-Falcon Q4_K_M ffn_down + use_more_bits) → Q6_K.
        let mut qs = mk_qs(LlamaFtype::MostlyQ4_K_M, ArchName::Llama3);
        qs.n_ffn_down = 32;
        qs.i_ffn_down = 0; // 0 < 32/8 → use_more_bits=true
        let shape = [4096, 1];
        let t = mk_tensor("blk.0.ffn_down.weight", &shape, ArchName::Llama3);
        let pol = StandardPolicy::new();
        assert_eq!(
            pol.target_for(&mut qs, &t, TensorCategory::FfnDown).unwrap(),
            GgmlType::Q6_K
        );
        assert_eq!(qs.i_ffn_down, 1);
    }

    #[test]
    fn all_v1_arches_q5_k_m_attn_q_basic() {
        // attn_q on Q5_K_M doesn't hit any override branch in C:561-567
        // (those only fire on IQ3_XS / IQ3_XXS), so all arches should
        // pass through to primary = Q5_K (then tensor_type_fallback
        // leaves it alone since shape[0]=4096 % 256 = 0).
        let arches = [
            ArchName::Gemma4,
            ArchName::Gemma4Mmproj,
            ArchName::Qwen35Moe,
            ArchName::Qwen3VlText,
            ArchName::Bert,
            ArchName::NomicBert,
            ArchName::Llama3,
            ArchName::MiniMaxM2,
        ];
        for arch in arches {
            let mut qs = mk_qs(LlamaFtype::MostlyQ5_K_M, arch);
            let shape = [4096, 1];
            let t = mk_tensor("blk.0.attn_q.weight", &shape, arch);
            let pol = StandardPolicy::new();
            let got = pol.target_for(&mut qs, &t, TensorCategory::AttnQ).unwrap();
            assert_eq!(got, GgmlType::Q5_K, "arch {} should pick Q5_K", arch.name());
        }
    }

    #[test]
    fn tensor_type_fallback_chains_on_misaligned_k_quant() {
        // n_per_row=17 ⇒ not divisible by 256 (Q5_K) AND not by 32
        // (Q5_1) ⇒ ADR shape_fallback contract returns typed error.
        let mut qs = mk_qs(LlamaFtype::MostlyQ5_K_M, ArchName::Llama3);
        let shape = [17, 1];
        let t = mk_tensor("blk.0.attn_q.weight", &shape, ArchName::Llama3);
        let pol = StandardPolicy::new();
        let err = pol
            .target_for(&mut qs, &t, TensorCategory::AttnQ)
            .unwrap_err();
        assert!(matches!(err, QuantizeError::NotBlockAligned { .. }));
    }

    #[test]
    fn tensor_type_fallback_first_shift_q5_k_to_q5_1() {
        // n_per_row=128 — not divisible by 256 (Q5_K block) but is
        // divisible by 32 (Q5_1 block) ⇒ downshift to Q5_1 per C:388.
        let mut qs = mk_qs(LlamaFtype::MostlyQ5_K_M, ArchName::Llama3);
        let shape = [128, 1];
        // attn_q (NOT a use_more_bits / bump branch); primary Q5_K → Q5_1.
        let t = mk_tensor("blk.0.attn_q.weight", &shape, ArchName::Llama3);
        let pol = StandardPolicy::new();
        assert_eq!(
            pol.target_for(&mut qs, &t, TensorCategory::AttnQ).unwrap(),
            GgmlType::Q5_1
        );
    }

    #[test]
    fn category_classify_basic() {
        assert_eq!(TensorCategory::classify("token_embd.weight"), TensorCategory::TokenEmbd);
        assert_eq!(TensorCategory::classify("output.weight"), TensorCategory::Output);
        assert_eq!(TensorCategory::classify("blk.0.attn_q.weight"), TensorCategory::AttnQ);
        assert_eq!(TensorCategory::classify("blk.10.attn_v.weight"), TensorCategory::AttnV);
        assert_eq!(TensorCategory::classify("blk.5.attn_k.weight"), TensorCategory::AttnK);
        assert_eq!(TensorCategory::classify("blk.3.attn_output.weight"), TensorCategory::AttnOutput);
        assert_eq!(TensorCategory::classify("blk.7.ffn_down.weight"), TensorCategory::FfnDown);
        assert_eq!(TensorCategory::classify("blk.7.ffn_down_exps.weight"), TensorCategory::FfnDown);
        assert_eq!(TensorCategory::classify("blk.7.ffn_up.weight"), TensorCategory::FfnUp);
        assert_eq!(TensorCategory::classify("blk.7.ffn_up_exps.weight"), TensorCategory::FfnUp);
        assert_eq!(TensorCategory::classify("blk.7.ffn_gate.weight"), TensorCategory::FfnGate);
        assert_eq!(TensorCategory::classify("blk.7.ffn_gate_exps.weight"), TensorCategory::FfnGate);
        assert_eq!(TensorCategory::classify("blk.7.ffn_gate_inp.weight"), TensorCategory::FfnGate);
        // C:122-127 — fused-QKV / KV_B priority.
        assert_eq!(TensorCategory::classify("blk.0.attn_qkv.weight"), TensorCategory::AttnQkv);
        assert_eq!(TensorCategory::classify("blk.0.attn_kv_b.weight"), TensorCategory::AttnKvB);
        assert_eq!(TensorCategory::classify("blk.0.attn_norm.weight"), TensorCategory::Other);
        // C:101 — per_layer_token_embd alias.
        assert_eq!(
            TensorCategory::classify("per_layer_token_embd.weight"),
            TensorCategory::TokenEmbd
        );
    }

    #[test]
    fn fallback_passthrough_when_aligned() {
        assert_eq!(tensor_type_fallback(GgmlType::Q5_K, 512).unwrap(), GgmlType::Q5_K);
        assert_eq!(tensor_type_fallback(GgmlType::Q4_K, 256).unwrap(), GgmlType::Q4_K);
        assert_eq!(tensor_type_fallback(GgmlType::Q4_0, 32).unwrap(), GgmlType::Q4_0);
    }

    #[test]
    fn fallback_q4_k_to_q5_0() {
        assert_eq!(tensor_type_fallback(GgmlType::Q4_K, 128).unwrap(), GgmlType::Q5_0);
    }

    #[test]
    fn fallback_q5_k_to_q5_1() {
        assert_eq!(tensor_type_fallback(GgmlType::Q5_K, 160).unwrap(), GgmlType::Q5_1);
    }

    #[test]
    fn fallback_q6_k_to_q8_0() {
        assert_eq!(tensor_type_fallback(GgmlType::Q6_K, 96).unwrap(), GgmlType::Q8_0);
    }

    #[test]
    fn fallback_q2_q3_to_q4_0() {
        assert_eq!(tensor_type_fallback(GgmlType::Q2_K, 128).unwrap(), GgmlType::Q4_0);
        assert_eq!(tensor_type_fallback(GgmlType::Q3_K, 64).unwrap(), GgmlType::Q4_0);
    }

    #[test]
    fn fallback_second_misalignment_is_typed_error() {
        let err = tensor_type_fallback(GgmlType::Q5_K, 15).unwrap_err();
        assert!(matches!(err, QuantizeError::NotBlockAligned { .. }));
    }

    #[test]
    fn fallback_no_path_for_unaligned_legacy() {
        let err = tensor_type_fallback(GgmlType::Q4_0, 17).unwrap_err();
        assert!(matches!(err, QuantizeError::NotBlockAligned { .. }));
    }
}
