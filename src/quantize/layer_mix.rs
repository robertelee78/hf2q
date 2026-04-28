//! `layer_mix` — per-tensor target-format dispatch for the M/S
//! variants of K-quants (`Q4_K_M`, `Q4_K_S`, `Q5_K_M`, `Q5_K_S`,
//! `Q6_K`, …). ADR-014 P7 iter-3r.
//!
//! ## Why this layer exists
//!
//! `KQuantTarget` (in [`crate::quantize::k_quant_codec`]) names a
//! single block format: Q4_K, Q5_K, Q6_K, etc. But the **user-facing**
//! variants in llama.cpp (`Q4_K_M`, `Q4_K_S`, etc.) are **policies**
//! that pick a target per tensor:
//!
//! - `Q4_K_M` (medium): most weights at Q4_K, but `token_embd` /
//!   `output` and certain layer-positioned `attn_v` / `ffn_down`
//!   tensors are bumped to Q5_K or Q6_K for quality.
//! - `Q4_K_S` (small): nearly everything at Q4_K, fewer upgrades.
//! - `Q5_K_M` / `Q5_K_S`: same shape as the Q4_K family, base Q5_K.
//! - `Q6_K`: base Q6_K (output may bump to Q8_0 in some archs).
//!
//! This module ports a documented subset of llama.cpp's
//! `llama_tensor_get_type_impl` (`/opt/llama.cpp/src/llama-quant.cpp:411-660`)
//! policy: the **headline rules** for the typical (non-Falcon,
//! non-MoE-8x7B) architecture. The canonical per-layer-position
//! `use_more_bits` upgrades are honoured; arch-specific edge cases
//! (Falcon, MoE-8x, GQA-70B) are deferred.
//!
//! ## Coverage
//!
//! Implemented:
//! - `token_embd.weight` / `output.weight` — Q6_K bump on every
//!   `_K_M` variant (matches `llama-quant.cpp:439-442` plus the
//!   `LLAMA_FTYPE_MOSTLY_Q*_K_M` defaults).
//! - `attn_v.weight` — Q5_K bump on `Q4_K_S` for layers `< 4`;
//!   Q6_K bump on `Q4_K_M`/`Q5_K_M` via `use_more_bits` (lines 530-536).
//! - `ffn_down.weight` — Q6_K bump on `Q4_K_M`/`Q5_K_M` via
//!   `use_more_bits` (lines 590-596). Falcon variant skipped.
//! - All other weight tensors → base target (e.g. `Q4_K` for any
//!   `Q4_K_*` variant).
//!
//! Deferred (each adds ~30 lines):
//! - Falcon arch policy (different `ffn_down` rules at layer 0..16).
//! - MoE 8-expert (Mixtral) `Q8_0` bumps for attn_v / attn_k.
//! - GQA-70B `Q5_K` upgrade for attn_v.
//! - IQ-family variants (IQ3_XXS, IQ3_M, IQ4_NL/XS, IQ2_S, …).
//!
//! ## Output
//!
//! `target_for(variant, tensor_name, i_layer, n_layers)` returns the
//! [`KQuantTarget`] to pass to
//! [`crate::quantize::k_quant_codec::quantize_row_to_bytes`]. The
//! policy is **stateless** and **arch-agnostic** at this iter — the
//! per-tensor counter (`qs.i_attention_wv`) used by
//! `llama_tensor_get_type_impl` is replaced by the explicit
//! `i_layer` / `n_layers` arguments.

use thiserror::Error;

use crate::quantize::k_quant_codec::KQuantTarget;

/// Errors from layer-mix policy lookup.
#[derive(Error, Debug, PartialEq, Eq)]
pub enum LayerMixError {
    /// `variant` string is not a recognised K-quant policy name
    /// (e.g. typo, IQ family not yet supported).
    #[error("layer_mix: unknown variant '{variant}'")]
    UnknownVariant { variant: String },
}

/// User-facing K-quant policy variants. Maps to llama.cpp's
/// `LLAMA_FTYPE_MOSTLY_Q*_K_*` enum values.
///
/// `non_camel_case_types` is allowed because the variant names are the
/// canonical wire/CLI strings (`Q4_K_M`, `Q5_K_S`, etc.) — they show up
/// in `--quant` arguments and quant_info.method strings; renaming them
/// to UpperCamel would diverge from llama.cpp's published nomenclature
/// and force a translation layer in `parse()` and `name()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum KQuantVariant {
    /// `Q3_K_S` — base Q3_K, minimal upgrades. Output bumps to Q6_K
    /// per `llama-quant.cpp:439-460`.
    Q3_K_S,
    /// `Q3_K_M` — base Q3_K with attn_v Q5_K (i_layer<2) / Q4_K bump
    /// per `llama-quant.cpp:527-528`, ffn_down Q5_K (i_layer<n/16) /
    /// Q4_K (use_more_bits) / Q3_K bump per `:578-582`.
    Q3_K_M,
    /// `Q3_K_L` — base Q3_K with attn_v → Q5_K (`:530`) and ffn_down
    /// → Q5_K (`:587-588`).  Output bumps to Q6_K.
    Q3_K_L,
    /// `Q4_K_S` — base Q4_K, minimal upgrades.
    Q4_K_S,
    /// `Q4_K_M` — base Q4_K with token_embd/output → Q6_K, attn_v
    /// and ffn_down upgrades on `use_more_bits` layers.
    Q4_K_M,
    /// `Q5_K_S` — base Q5_K, minimal upgrades.
    Q5_K_S,
    /// `Q5_K_M` — base Q5_K with token_embd/output → Q6_K, attn_v
    /// upgrades on `use_more_bits` layers.
    Q5_K_M,
    /// `Q6_K` — base Q6_K. Output retains Q6_K (no upgrade beyond
    /// the base type).
    Q6_K,
}

impl KQuantVariant {
    /// Parse a user-supplied variant string.
    pub fn parse(variant: &str) -> Result<Self, LayerMixError> {
        match variant {
            "Q3_K_S" | "q3_k_s" => Ok(Self::Q3_K_S),
            "Q3_K_M" | "q3_k_m" | "Q3_K" | "q3_k" => Ok(Self::Q3_K_M),
            "Q3_K_L" | "q3_k_l" => Ok(Self::Q3_K_L),
            "Q4_K_S" | "q4_k_s" => Ok(Self::Q4_K_S),
            "Q4_K_M" | "q4_k_m" | "Q4_K" | "q4_k" => Ok(Self::Q4_K_M),
            "Q5_K_S" | "q5_k_s" => Ok(Self::Q5_K_S),
            "Q5_K_M" | "q5_k_m" | "Q5_K" | "q5_k" => Ok(Self::Q5_K_M),
            "Q6_K" | "q6_k" => Ok(Self::Q6_K),
            _ => Err(LayerMixError::UnknownVariant {
                variant: variant.to_string(),
            }),
        }
    }

    /// The base K-quant target this variant rounds to (before any
    /// per-tensor upgrades).
    pub fn base_target(&self) -> KQuantTarget {
        match self {
            Self::Q3_K_S | Self::Q3_K_M | Self::Q3_K_L => KQuantTarget::Q3K,
            Self::Q4_K_S | Self::Q4_K_M => KQuantTarget::Q4K,
            Self::Q5_K_S | Self::Q5_K_M => KQuantTarget::Q5K,
            Self::Q6_K => KQuantTarget::Q6K,
        }
    }

    /// Canonical name for logging / `quant_info.method` strings.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Q3_K_S => "Q3_K_S",
            Self::Q3_K_M => "Q3_K_M",
            Self::Q3_K_L => "Q3_K_L",
            Self::Q4_K_S => "Q4_K_S",
            Self::Q4_K_M => "Q4_K_M",
            Self::Q5_K_S => "Q5_K_S",
            Self::Q5_K_M => "Q5_K_M",
            Self::Q6_K => "Q6_K",
        }
    }

    /// Enumerate every supported variant in canonical order. Used by
    /// the P8 CLI to register the variant menu and by integration
    /// tests to exercise every policy end-to-end without re-listing
    /// the variants by hand.
    pub fn all() -> &'static [Self] {
        &[
            Self::Q3_K_S,
            Self::Q3_K_M,
            Self::Q3_K_L,
            Self::Q4_K_S,
            Self::Q4_K_M,
            Self::Q5_K_S,
            Self::Q5_K_M,
            Self::Q6_K,
        ]
    }
}

impl std::fmt::Display for KQuantVariant {
    /// Display matches the canonical name (e.g. `"Q4_K_M"`). Round-trips
    /// through [`KQuantVariant::parse`].
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Tensor category — coarse classification of GGUF weight tensors,
/// used to route per-tensor targets. **Matches llama.cpp's full
/// `tensor_category` enum** (`llama-quant.cpp:99-108` + classification
/// at `:115-150`) so MoE expert variants (`*_exps.weight`,
/// `*_shexp.weight`) and merged-attention QKV / KV-B layouts route to
/// the same per-tensor policy llama.cpp would pick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorCategory {
    /// `output.weight` (lm_head). With tied embeddings, also covers
    /// `token_embd.weight` per `llama-quant.cpp:439`.
    Output,
    /// `token_embd.weight` (separate from output for non-tied archs).
    TokenEmbd,
    /// `attn_qkv.weight` — merged Q/K/V projection (treated as
    /// attn_v-equivalent in `category_is_attn_v`,
    /// `llama-quant.cpp:153-157`).
    AttentionQkv,
    /// `attn_kv_b.weight` — DeepSeek MLA's KV down-projection
    /// (treated as attn_v-equivalent likewise).
    AttentionKvB,
    /// `attn_v.weight` — value projection in attention. Substring
    /// match → covers `blk.<i>.attn_v.weight` and any future
    /// arch-specific names ending in `attn_v.weight`.
    AttentionV,
    /// `attn_k.weight` — key projection.
    AttentionK,
    /// `attn_q.weight` — query projection.
    AttentionQ,
    /// `attn_output.weight` — attention output projection.
    AttentionOutput,
    /// `ffn_up*` — FFN up-projection. Substring match covers MoE
    /// variants (`ffn_up_exps.weight`, `ffn_up_shexp.weight`).
    FfnUp,
    /// `ffn_gate*` — FFN gate. Substring match covers MoE variants
    /// (`ffn_gate_exps.weight`, `ffn_gate_shexp.weight`) AND the
    /// router gate `ffn_gate_inp.weight` (which `should_skip_quantization`
    /// then filters out per `llama-quant.cpp:307`).
    FfnGate,
    /// `ffn_down*` — FFN down-projection. Substring match covers MoE
    /// variants (`ffn_down_exps.weight`, `ffn_down_shexp.weight`).
    /// The per-layer special case for `_K_M` upgrades.
    FfnDown,
    /// Any other weight tensor (norm, embedding scale, etc.) — gets
    /// the variant's base target unchanged.
    Other,
}

impl TensorCategory {
    /// Classify a tensor by its GGUF name (post-`map_tensor_name_to_gguf`
    /// transformation). Pure substring-matching port of llama.cpp's
    /// `tensor_get_category` (`llama-quant.cpp:115-150`):
    ///
    /// ```c
    /// if (tensor_name_match_output_weight(...))         return OUTPUT;
    /// if (tensor_name_match_token_embd(...))            return TOKEN_EMBD;
    /// if (find("attn_qkv.weight"))                      return ATTENTION_QKV;
    /// if (find("attn_kv_b.weight"))                     return ATTENTION_KV_B;
    /// if (find("attn_v.weight"))                        return ATTENTION_V;
    /// if (find("attn_k.weight"))                        return ATTENTION_K;
    /// if (find("attn_q.weight"))                        return ATTENTION_Q;
    /// if (find("attn_output.weight"))                   return ATTENTION_OUTPUT;
    /// if (find("ffn_up"))                               return FFN_UP;
    /// if (find("ffn_gate"))                             return FFN_GATE;
    /// if (find("ffn_down"))                             return FFN_DOWN;
    /// return OTHER;
    /// ```
    ///
    /// Priority order matters: `attn_qkv.weight` contains the substring
    /// `attn_v`, so the QKV check must run first. Likewise `attn_kv_b`
    /// before `attn_v`, `attn_v` before `attn_k`, etc.
    ///
    /// **Substring matching → MoE coverage:** `blk.0.ffn_down_exps.weight`
    /// matches the `ffn_down` substring → `FfnDown`. The pre-iter-3u
    /// exact-match logic was a real Q4_K_M parity gap on MoE models —
    /// expert FFN tensors fell through to `Other` and missed the
    /// `use_more_bits → Q6_K` bump llama.cpp applies.
    pub fn classify(tensor_name: &str) -> Self {
        if tensor_name == "output.weight" {
            return Self::Output;
        }
        if tensor_name == "token_embd.weight" {
            return Self::TokenEmbd;
        }
        // llama.cpp priority order: attn_qkv → attn_kv_b → attn_v →
        // attn_k → attn_q → attn_output → ffn_up → ffn_gate → ffn_down.
        if tensor_name.contains("attn_qkv.weight") {
            return Self::AttentionQkv;
        }
        if tensor_name.contains("attn_kv_b.weight") {
            return Self::AttentionKvB;
        }
        if tensor_name.contains("attn_v.weight") {
            return Self::AttentionV;
        }
        if tensor_name.contains("attn_k.weight") {
            return Self::AttentionK;
        }
        if tensor_name.contains("attn_q.weight") {
            return Self::AttentionQ;
        }
        if tensor_name.contains("attn_output.weight") {
            return Self::AttentionOutput;
        }
        if tensor_name.contains("ffn_up") {
            return Self::FfnUp;
        }
        if tensor_name.contains("ffn_gate") {
            return Self::FfnGate;
        }
        if tensor_name.contains("ffn_down") {
            return Self::FfnDown;
        }
        Self::Other
    }
}

/// `true` if this tensor must NOT be quantized regardless of variant —
/// matches `llama-quant.cpp:307`:
///
/// ```c
/// quantize &= name.find("ffn_gate_inp.weight") == std::string::npos;
/// ```
///
/// `ffn_gate_inp.weight` is the MoE router gate (small, [n_experts,
/// hidden_size] typically) — llama.cpp keeps it at original precision
/// because routing decisions are extremely sensitive to perturbation.
/// Without this skip, MoE Q4_K_M output diverges from llama.cpp's gguf
/// at the router tensor.
pub fn should_skip_quantization(tensor_name: &str) -> bool {
    tensor_name.contains("ffn_gate_inp.weight")
}

/// Returns `true` if this tensor should NOT be K-quant-encoded — instead emit
/// as F16 passthrough. Triggers:
///
///   1. **Vision encoder tensors** (HF naming variants:
///      `model.visual.*`, `vision_tower.*`, `vision_model.*`, `vit.*`,
///      `visual.*`, and `.visual.*` substring for prefix-stripped layouts).
///      Vision encoders are tiny relative to language tensors; F16 preserves
///      quality with negligible file-size cost AND matches the `is_vision_tensor`
///      preserve-at-full-precision policy at `src/ir/mod.rs:138`.
///   2. **Defensive:** any tensor whose row length (last dim) is not divisible
///      by `QK_K = 256`. The K-quant codec at
///      `src/quantize/k_quant_codec.rs::quantize_row_to_bytes` rejects
///      misaligned rows with a typed error; pre-empting that rejection here
///      keeps the convert pipeline alive on shapes the codec cannot encode
///      (e.g. Qwen3.6-27B vision-attn `proj.weight` at row_len = 1152, the
///      LIVE blocker recorded in ADR-014 § "P11-prereq Iter D" 2026-04-27).
///
/// **Pattern matching is substring-based** so this catches both raw HF names
/// (pre-prefix-strip) AND the prefix-stripped `language_model.` variants.
/// Mirrors the substring style of `should_skip_quantization` and
/// `TensorCategory::classify` in this module.
///
/// Used by `KQuantCodecQuantizer`, `VariantKQuantizer`, and `DwqKQuantizer`
/// at the top of their `quantize_tensor` impls — skipped tensors emit as F16
/// via `crate::quantize::k_quant_codec_quantizer::f16_passthrough`, which
/// routes correctly through the F16 arms in `repack_to_ggml_blocks` AND
/// `quant_info_to_ggml_type` in `src/backends/gguf.rs`.
pub fn should_emit_f16_for_kquant(tensor_name: &str, row_len: usize) -> bool {
    // Vision encoder patterns. Order doesn't matter (any match wins).
    if tensor_name.contains("model.visual.")
        || tensor_name.contains("vision_tower.")
        || tensor_name.contains("vision_model.")
        || tensor_name.contains("vit.")
        || tensor_name.starts_with("visual.")
        || tensor_name.contains(".visual.")
    {
        return true;
    }
    // Defensive: any non-256-multiple row length cannot encode under any
    // K-quant target (Q4_K/Q5_K/Q6_K all use QK_K = 256).
    if row_len % 256 != 0 {
        return true;
    }
    false
}

/// llama.cpp's `use_more_bits` predicate (line 417-419):
/// `i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8) % 3 == 2`.
/// The "first 1/8 + last 1/8 + every-3rd-after-1/8" rule that's used
/// to bump select attn_v / ffn_down layers to higher precision.
#[inline]
fn use_more_bits(i_layer: usize, n_layers: usize) -> bool {
    let eighth = n_layers / 8;
    let seven_eighths = 7 * n_layers / 8;
    if i_layer < eighth || i_layer >= seven_eighths {
        return true;
    }
    // (i_layer - n_layers/8) % 3 == 2 — only meaningful when
    // i_layer >= eighth (caller already handled lower path).
    (i_layer - eighth) % 3 == 2
}

/// Per-tensor target dispatch for a K-quant variant. Maps a tensor
/// name + layer position to the [`KQuantTarget`] that should be
/// passed to the codec.
///
/// `tensor_name` is the GGUF tensor name (post-`map_tensor_name_to_gguf`).
/// `i_layer` is the tensor's transformer block index (parsed from
/// `blk.<N>.…`); ignored for non-`blk.*` tensors. `n_layers` is the
/// model's total transformer depth.
///
/// **Policy rules (subset of `llama_tensor_get_type_impl`):**
///
/// 1. `output.weight` / `token_embd.weight` — bumps to `Q6_K` on every
///    `_K_M` variant; stays at base on `_K_S`. Returns base for
///    `Q6_K` variant (already at Q6_K).
/// 2. `attn_v.weight`:
///    - `Q4_K_M` / `Q5_K_M` + `use_more_bits(i_layer, n_layers)` → `Q6_K`.
///    - `Q4_K_S` + `i_layer < 4` → `Q5_K`.
///    - else → base.
/// 3. `ffn_down.weight`:
///    - `Q4_K_M` / `Q5_K_M` + `use_more_bits` → `Q6_K`.
///    - else → base.
/// 4. Everything else → base.
///
/// Architecture-specific edge cases (Falcon, MoE-8x, GQA-70B) and
/// the IQ family are deferred per the module doc.
pub fn target_for(
    variant: KQuantVariant,
    tensor_name: &str,
    i_layer: usize,
    n_layers: usize,
) -> KQuantTarget {
    let base = variant.base_target();
    let category = TensorCategory::classify(tensor_name);

    match category {
        TensorCategory::Output | TensorCategory::TokenEmbd => match variant {
            // Q3_K_S/M/L all bump output → Q6_K per `llama-quant.cpp:439-460`
            // (the OUTPUT branch falls through `else if (new_type !=
            // GGML_TYPE_Q8_0) new_type = GGML_TYPE_Q6_K;` for K-family
            // ftypes).
            KQuantVariant::Q3_K_S
            | KQuantVariant::Q3_K_M
            | KQuantVariant::Q3_K_L
            | KQuantVariant::Q4_K_M
            | KQuantVariant::Q5_K_M => KQuantTarget::Q6K,
            _ => base,
        },
        // ATTENTION_V, ATTENTION_QKV, ATTENTION_KV_B — `category_is_attn_v`
        // at `llama-quant.cpp:153-157` lumps these together so the V-policy
        // applies to all three.
        TensorCategory::AttentionV
        | TensorCategory::AttentionQkv
        | TensorCategory::AttentionKvB => match variant {
            KQuantVariant::Q4_K_M | KQuantVariant::Q5_K_M => {
                if use_more_bits(i_layer, n_layers) {
                    KQuantTarget::Q6K
                } else {
                    base
                }
            }
            KQuantVariant::Q4_K_S => {
                if i_layer < 4 {
                    KQuantTarget::Q5K
                } else {
                    base
                }
            }
            // Q3_K_M: i_layer < 2 → Q5_K, else → Q4_K (`llama-quant.cpp:527-528`).
            KQuantVariant::Q3_K_M => {
                if i_layer < 2 {
                    KQuantTarget::Q5K
                } else {
                    KQuantTarget::Q4K
                }
            }
            // Q3_K_L: attn_v → Q5_K (`llama-quant.cpp:530`).
            KQuantVariant::Q3_K_L => KQuantTarget::Q5K,
            _ => base,
        },
        TensorCategory::FfnDown => match variant {
            KQuantVariant::Q4_K_M | KQuantVariant::Q5_K_M => {
                if use_more_bits(i_layer, n_layers) {
                    KQuantTarget::Q6K
                } else {
                    base
                }
            }
            // Q3_K_M ffn_down (`llama-quant.cpp:578-582`):
            //   i_layer < n/16    → Q5_K
            //   use_more_bits     → Q4_K
            //   else              → Q3_K (base)
            // (Falcon-arch override deferred — IR has no arch field here.)
            KQuantVariant::Q3_K_M => {
                if i_layer < n_layers / 16 {
                    KQuantTarget::Q5K
                } else if use_more_bits(i_layer, n_layers) {
                    KQuantTarget::Q4K
                } else {
                    base
                }
            }
            // Q3_K_L: ffn_down → Q5_K (`llama-quant.cpp:587-588`, non-Falcon path).
            KQuantVariant::Q3_K_L => KQuantTarget::Q5K,
            _ => base,
        },
        // ATTENTION_K, ATTENTION_Q, ATTENTION_OUTPUT, FFN_UP, FFN_GATE,
        // OTHER → base target.  llama.cpp has stateful per-layer
        // counters for FFN_UP / FFN_GATE / ATTENTION_OUTPUT (`i_ffn_up`,
        // `i_ffn_gate`, etc.) that drive `use_more_bits`-style bumps;
        // those need a stateful call site (P7 iter-3v: track per-layer
        // counters in `VariantKQuantizer` if/when MoE quality data
        // shows the bumps matter).
        TensorCategory::AttentionK
        | TensorCategory::AttentionQ
        | TensorCategory::AttentionOutput
        | TensorCategory::FfnUp
        | TensorCategory::FfnGate
        | TensorCategory::Other => base,
    }
}

/// Convenience: parse a variant string + dispatch to `target_for`
/// in a single call. Surfaces typed errors for unknown variants.
pub fn target_for_str(
    variant: &str,
    tensor_name: &str,
    i_layer: usize,
    n_layers: usize,
) -> Result<KQuantTarget, LayerMixError> {
    let v = KQuantVariant::parse(variant)?;
    Ok(target_for(v, tensor_name, i_layer, n_layers))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_parse_canonical_names() {
        assert_eq!(KQuantVariant::parse("Q4_K_S").unwrap(), KQuantVariant::Q4_K_S);
        assert_eq!(KQuantVariant::parse("Q4_K_M").unwrap(), KQuantVariant::Q4_K_M);
        assert_eq!(KQuantVariant::parse("Q5_K_S").unwrap(), KQuantVariant::Q5_K_S);
        assert_eq!(KQuantVariant::parse("Q5_K_M").unwrap(), KQuantVariant::Q5_K_M);
        assert_eq!(KQuantVariant::parse("Q6_K").unwrap(), KQuantVariant::Q6_K);
    }

    #[test]
    fn variant_parse_lowercase_accepted() {
        assert_eq!(KQuantVariant::parse("q4_k_m").unwrap(), KQuantVariant::Q4_K_M);
        assert_eq!(KQuantVariant::parse("q6_k").unwrap(), KQuantVariant::Q6_K);
    }

    #[test]
    fn variant_parse_q4_k_alias_normalises_to_m() {
        // "Q4_K" without suffix is conventionally Q4_K_M (medium default).
        assert_eq!(KQuantVariant::parse("Q4_K").unwrap(), KQuantVariant::Q4_K_M);
        assert_eq!(KQuantVariant::parse("Q5_K").unwrap(), KQuantVariant::Q5_K_M);
    }

    #[test]
    fn variant_parse_unknown_errors() {
        // Q3_K_M is now a real variant (iter-9); use a still-unsupported
        // IQ-family name to exercise the error path.
        let err = KQuantVariant::parse("IQ2_M").unwrap_err();
        match err {
            LayerMixError::UnknownVariant { variant } => assert_eq!(variant, "IQ2_M"),
        }
        assert!(KQuantVariant::parse("IQ4_NL").is_err());
        assert!(KQuantVariant::parse("garbage").is_err());
    }

    #[test]
    fn variant_base_targets() {
        assert_eq!(KQuantVariant::Q3_K_S.base_target(), KQuantTarget::Q3K);
        assert_eq!(KQuantVariant::Q3_K_M.base_target(), KQuantTarget::Q3K);
        assert_eq!(KQuantVariant::Q3_K_L.base_target(), KQuantTarget::Q3K);
        assert_eq!(KQuantVariant::Q4_K_S.base_target(), KQuantTarget::Q4K);
        assert_eq!(KQuantVariant::Q4_K_M.base_target(), KQuantTarget::Q4K);
        assert_eq!(KQuantVariant::Q5_K_S.base_target(), KQuantTarget::Q5K);
        assert_eq!(KQuantVariant::Q5_K_M.base_target(), KQuantTarget::Q5K);
        assert_eq!(KQuantVariant::Q6_K.base_target(), KQuantTarget::Q6K);
    }

    #[test]
    fn classify_recognises_canonical_names() {
        assert_eq!(TensorCategory::classify("output.weight"), TensorCategory::Output);
        assert_eq!(TensorCategory::classify("token_embd.weight"), TensorCategory::TokenEmbd);
        assert_eq!(TensorCategory::classify("blk.0.attn_v.weight"), TensorCategory::AttentionV);
        assert_eq!(TensorCategory::classify("blk.5.attn_k.weight"), TensorCategory::AttentionK);
        assert_eq!(TensorCategory::classify("blk.10.attn_q.weight"), TensorCategory::AttentionQ);
        assert_eq!(TensorCategory::classify("blk.0.ffn_down.weight"), TensorCategory::FfnDown);
        // iter-3u: substring matching brings ffn_up under FfnUp (was
        // Other under the pre-iter-3u exact-match logic).  This closes
        // the parity gap with `llama-quant.cpp:140-142`.
        assert_eq!(TensorCategory::classify("blk.0.ffn_up.weight"), TensorCategory::FfnUp);
        assert_eq!(TensorCategory::classify("blk.0.attn_norm.weight"), TensorCategory::Other);
        assert_eq!(TensorCategory::classify("rope_freqs.weight"), TensorCategory::Other);
    }

    #[test]
    fn use_more_bits_first_eighth() {
        // n_layers = 32 → eighth = 4, seven_eighths = 28.
        // First 4 layers: always true.
        for i in 0..4 {
            assert!(use_more_bits(i, 32), "i={i}");
        }
    }

    #[test]
    fn use_more_bits_last_eighth() {
        // Last 4 layers (28..32): always true.
        for i in 28..32 {
            assert!(use_more_bits(i, 32), "i={i}");
        }
    }

    #[test]
    fn use_more_bits_every_third_in_middle() {
        // Middle range 4..28: (i - 4) % 3 == 2 means i = 6, 9, 12, ...
        // Pattern: true at 6, 9, 12, 15, 18, 21, 24, 27.
        // Build expected from the formula directly.
        let n_layers = 32;
        for i in 4..28 {
            let expected = (i - 4) % 3 == 2;
            assert_eq!(use_more_bits(i, n_layers), expected, "i={i}");
        }
    }

    /// Output / token_embd bump to Q6_K on `_K_M` variants.
    #[test]
    fn target_output_bumps_to_q6k_on_km() {
        for variant in [KQuantVariant::Q4_K_M, KQuantVariant::Q5_K_M] {
            assert_eq!(
                target_for(variant, "output.weight", 0, 32),
                KQuantTarget::Q6K
            );
            assert_eq!(
                target_for(variant, "token_embd.weight", 0, 32),
                KQuantTarget::Q6K
            );
        }
    }

    /// Output / token_embd stay at base on `_K_S` variants.
    #[test]
    fn target_output_stays_on_ks() {
        assert_eq!(
            target_for(KQuantVariant::Q4_K_S, "output.weight", 0, 32),
            KQuantTarget::Q4K
        );
        assert_eq!(
            target_for(KQuantVariant::Q5_K_S, "token_embd.weight", 0, 32),
            KQuantTarget::Q5K
        );
    }

    /// `Q6_K` variant: output stays at Q6_K (already at base).
    #[test]
    fn target_output_q6k_stays_q6k() {
        assert_eq!(
            target_for(KQuantVariant::Q6_K, "output.weight", 0, 32),
            KQuantTarget::Q6K
        );
    }

    /// attn_v on Q4_K_M bumps to Q6_K when `use_more_bits` is true.
    #[test]
    fn target_attn_v_q4_k_m_more_bits_layers_bump_to_q6k() {
        // n_layers=32. Layer 0 is in first 1/8 → use_more_bits=true.
        assert_eq!(
            target_for(KQuantVariant::Q4_K_M, "blk.0.attn_v.weight", 0, 32),
            KQuantTarget::Q6K
        );
        // Layer 31 is in last 1/8.
        assert_eq!(
            target_for(KQuantVariant::Q4_K_M, "blk.31.attn_v.weight", 31, 32),
            KQuantTarget::Q6K
        );
        // Layer 6 (middle, every-3rd-after-eighth) → use_more_bits=true.
        assert_eq!(
            target_for(KQuantVariant::Q4_K_M, "blk.6.attn_v.weight", 6, 32),
            KQuantTarget::Q6K
        );
    }

    /// attn_v on Q4_K_M stays at Q4_K when `use_more_bits` is false.
    #[test]
    fn target_attn_v_q4_k_m_normal_layers_stay() {
        // Layer 5 (middle, not every-3rd) → use_more_bits=false.
        assert_eq!(
            target_for(KQuantVariant::Q4_K_M, "blk.5.attn_v.weight", 5, 32),
            KQuantTarget::Q4K
        );
    }

    /// attn_v on Q4_K_S: layers < 4 bump to Q5_K.
    #[test]
    fn target_attn_v_q4_k_s_first_four_layers_bump() {
        for i in 0..4 {
            assert_eq!(
                target_for(KQuantVariant::Q4_K_S, "blk.0.attn_v.weight", i, 32),
                KQuantTarget::Q5K,
                "layer {i}"
            );
        }
    }

    /// attn_v on Q4_K_S: layers >= 4 stay at base Q4_K.
    #[test]
    fn target_attn_v_q4_k_s_later_layers_stay() {
        assert_eq!(
            target_for(KQuantVariant::Q4_K_S, "blk.5.attn_v.weight", 5, 32),
            KQuantTarget::Q4K
        );
    }

    /// ffn_down on Q4_K_M / Q5_K_M follows `use_more_bits`.
    #[test]
    fn target_ffn_down_km_use_more_bits() {
        // Layer 0 (first 1/8): bump.
        assert_eq!(
            target_for(KQuantVariant::Q4_K_M, "blk.0.ffn_down.weight", 0, 32),
            KQuantTarget::Q6K
        );
        // Layer 5 (middle, not every-3rd): no bump.
        assert_eq!(
            target_for(KQuantVariant::Q5_K_M, "blk.5.ffn_down.weight", 5, 32),
            KQuantTarget::Q5K
        );
    }

    /// attn_q / attn_k / norm / other tensors always get the base.
    #[test]
    fn target_other_categories_get_base() {
        for tensor in [
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_gate.weight",
            "rope_freqs.weight",
        ] {
            assert_eq!(
                target_for(KQuantVariant::Q4_K_M, tensor, 0, 32),
                KQuantTarget::Q4K,
                "{tensor}"
            );
            assert_eq!(
                target_for(KQuantVariant::Q5_K_S, tensor, 0, 32),
                KQuantTarget::Q5K,
                "{tensor}"
            );
        }
    }

    /// `target_for_str` parses + dispatches in one call.
    #[test]
    fn target_for_str_round_trip() {
        let target = target_for_str("Q4_K_M", "output.weight", 0, 32).unwrap();
        assert_eq!(target, KQuantTarget::Q6K);

        let err = target_for_str("garbage", "output.weight", 0, 32).unwrap_err();
        match err {
            LayerMixError::UnknownVariant { variant } => assert_eq!(variant, "garbage"),
        }
    }

    // ── iter-3u: classify() substring-priority tests ────────────────

    /// llama.cpp priority order: `attn_qkv` checked before `attn_v`
    /// because `attn_qkv` contains the substring `attn_v`.
    #[test]
    fn classify_attn_qkv_before_attn_v() {
        assert_eq!(
            TensorCategory::classify("blk.0.attn_qkv.weight"),
            TensorCategory::AttentionQkv
        );
        // Pure attn_v.
        assert_eq!(
            TensorCategory::classify("blk.0.attn_v.weight"),
            TensorCategory::AttentionV
        );
    }

    /// `attn_kv_b.weight` (DeepSeek MLA) classifies as AttentionKvB.
    #[test]
    fn classify_attn_kv_b() {
        assert_eq!(
            TensorCategory::classify("blk.0.attn_kv_b.weight"),
            TensorCategory::AttentionKvB
        );
    }

    /// `attn_output.weight` is its own category (was Other pre-iter-3u).
    #[test]
    fn classify_attn_output() {
        assert_eq!(
            TensorCategory::classify("blk.0.attn_output.weight"),
            TensorCategory::AttentionOutput
        );
    }

    /// MoE FFN expert variants classify under their non-MoE category
    /// via substring matching — closes the Q4_K_M parity gap on MoE
    /// models pre-iter-3u where `*_exps.weight` fell through to Other.
    #[test]
    fn classify_moe_ffn_expert_variants() {
        // ffn_down_exps → FfnDown (eligible for use_more_bits Q6_K bump
        // on _K_M variants).
        assert_eq!(
            TensorCategory::classify("blk.0.ffn_down_exps.weight"),
            TensorCategory::FfnDown
        );
        assert_eq!(
            TensorCategory::classify("blk.0.ffn_down_shexp.weight"),
            TensorCategory::FfnDown
        );
        // ffn_up* and ffn_gate* match analogously.
        assert_eq!(
            TensorCategory::classify("blk.0.ffn_up_exps.weight"),
            TensorCategory::FfnUp
        );
        assert_eq!(
            TensorCategory::classify("blk.0.ffn_up_shexp.weight"),
            TensorCategory::FfnUp
        );
        assert_eq!(
            TensorCategory::classify("blk.0.ffn_gate_exps.weight"),
            TensorCategory::FfnGate
        );
        assert_eq!(
            TensorCategory::classify("blk.0.ffn_gate_shexp.weight"),
            TensorCategory::FfnGate
        );
    }

    /// MoE router gate `ffn_gate_inp.weight` classifies as FfnGate
    /// (substring match), AND `should_skip_quantization` returns true
    /// — matches `llama-quant.cpp:307` skip rule.
    #[test]
    fn ffn_gate_inp_classifies_then_skips() {
        assert_eq!(
            TensorCategory::classify("blk.0.ffn_gate_inp.weight"),
            TensorCategory::FfnGate
        );
        assert!(should_skip_quantization("blk.0.ffn_gate_inp.weight"));
        // Non-router ffn_gate variants do NOT trigger the skip.
        assert!(!should_skip_quantization("blk.0.ffn_gate.weight"));
        assert!(!should_skip_quantization("blk.0.ffn_gate_exps.weight"));
        assert!(!should_skip_quantization("blk.0.ffn_gate_shexp.weight"));
        assert!(!should_skip_quantization("blk.0.ffn_down.weight"));
    }

    /// AttentionQkv / AttentionKvB get the AttentionV policy per
    /// `category_is_attn_v` at `llama-quant.cpp:153-157`.
    #[test]
    fn target_attn_qkv_kvb_follow_attn_v_policy() {
        // Q4_K_M + use_more_bits at layer 0 → Q6K.
        for name in ["blk.0.attn_qkv.weight", "blk.0.attn_kv_b.weight"] {
            assert_eq!(
                target_for(KQuantVariant::Q4_K_M, name, 0, 32),
                KQuantTarget::Q6K,
                "{name} should bump to Q6_K with Q4_K_M + use_more_bits"
            );
        }
        // Q4_K_S + i_layer<4 → Q5_K bump.
        for name in ["blk.0.attn_qkv.weight", "blk.0.attn_kv_b.weight"] {
            assert_eq!(
                target_for(KQuantVariant::Q4_K_S, name, 2, 32),
                KQuantTarget::Q5K,
                "{name} should bump to Q5_K with Q4_K_S at i_layer<4"
            );
        }
        // Q6_K → base (Q6K) for both.
        for name in ["blk.0.attn_qkv.weight", "blk.0.attn_kv_b.weight"] {
            assert_eq!(
                target_for(KQuantVariant::Q6_K, name, 0, 32),
                KQuantTarget::Q6K,
                "{name} on Q6_K stays at base Q6_K"
            );
        }
    }

    /// MoE FfnDown experts on Q4_K_M get the use_more_bits Q6_K bump
    /// — was a parity gap pre-iter-3u (fell through to Other → Q4_K).
    #[test]
    fn target_moe_ffn_down_exps_q4km_use_more_bits_bump() {
        // Layer 0 (first 1/8): bump.
        assert_eq!(
            target_for(KQuantVariant::Q4_K_M, "blk.0.ffn_down_exps.weight", 0, 32),
            KQuantTarget::Q6K
        );
        // Layer 6 (every-3rd-after-1/8): bump.
        assert_eq!(
            target_for(KQuantVariant::Q4_K_M, "blk.6.ffn_down_exps.weight", 6, 32),
            KQuantTarget::Q6K
        );
        // Layer 5 (middle, not every-3rd): no bump.
        assert_eq!(
            target_for(KQuantVariant::Q4_K_M, "blk.5.ffn_down_exps.weight", 5, 32),
            KQuantTarget::Q4K
        );
    }

    /// FfnUp / FfnGate / AttentionOutput / AttentionK / AttentionQ /
    /// Other still default to base — explicit coverage to lock current
    /// policy and catch any future drift.  llama.cpp's per-layer
    /// counters for these categories are stateful (need
    /// `i_ffn_up`/`i_ffn_gate`/etc.) so a stateless dispatch can't
    /// reproduce them — that's deferred to iter-3v.
    #[test]
    fn target_other_attn_and_ffn_categories_get_base() {
        let names_and_categories = [
            ("blk.0.attn_q.weight", TensorCategory::AttentionQ),
            ("blk.0.attn_k.weight", TensorCategory::AttentionK),
            ("blk.0.attn_output.weight", TensorCategory::AttentionOutput),
            ("blk.0.ffn_up.weight", TensorCategory::FfnUp),
            ("blk.0.ffn_gate.weight", TensorCategory::FfnGate),
            ("blk.0.ffn_up_exps.weight", TensorCategory::FfnUp),
            ("blk.0.ffn_gate_exps.weight", TensorCategory::FfnGate),
            ("rope_freqs.weight", TensorCategory::Other),
        ];
        for (name, expected_cat) in names_and_categories {
            assert_eq!(
                TensorCategory::classify(name),
                expected_cat,
                "{name} classify"
            );
            assert_eq!(
                target_for(KQuantVariant::Q4_K_M, name, 0, 32),
                KQuantTarget::Q4K,
                "{name} target_for Q4_K_M"
            );
            assert_eq!(
                target_for(KQuantVariant::Q5_K_S, name, 0, 32),
                KQuantTarget::Q5K,
                "{name} target_for Q5_K_S"
            );
        }
    }

    /// `KQuantVariant::all()` enumerates every variant in canonical
    /// order; round-trips through `name()` and `parse()`.
    #[test]
    fn variant_all_round_trips_through_name_and_parse() {
        let all = KQuantVariant::all();
        assert_eq!(all.len(), 8, "exactly 8 supported variants (Q3_K_S/M/L added iter-9)");
        // canonical order matches the enum declaration
        assert_eq!(all[0], KQuantVariant::Q3_K_S);
        assert_eq!(all[1], KQuantVariant::Q3_K_M);
        assert_eq!(all[2], KQuantVariant::Q3_K_L);
        assert_eq!(all[3], KQuantVariant::Q4_K_S);
        assert_eq!(all[4], KQuantVariant::Q4_K_M);
        assert_eq!(all[5], KQuantVariant::Q5_K_S);
        assert_eq!(all[6], KQuantVariant::Q5_K_M);
        assert_eq!(all[7], KQuantVariant::Q6_K);
        // every variant round-trips through its name string
        for v in all {
            let parsed = KQuantVariant::parse(v.name())
                .unwrap_or_else(|e| panic!("parse({}) failed: {e:?}", v.name()));
            assert_eq!(parsed, *v, "round-trip name for {}", v.name());
            // Display matches name
            assert_eq!(format!("{v}"), v.name(), "Display matches name for {}", v.name());
        }
    }

    // ── Iter D: should_emit_f16_for_kquant predicate tests ─────────

    /// Qwen3.6 vision-encoder proj tensor at row_len = 1152 (the live
    /// blocker recorded in ADR-014 P11-prereq Iter D, 2026-04-27).
    /// Both the vision pattern AND the non-256-multiple defensive
    /// trigger fire here; either alone would suffice.
    #[test]
    fn vision_tensor_qwen35_proj_returns_true() {
        assert!(should_emit_f16_for_kquant(
            "model.visual.blocks.0.attn.proj.weight",
            1152,
        ));
    }

    /// CLIP-style `vision_tower.…` HF naming with an aligned row_len —
    /// the vision pattern alone must trigger.
    #[test]
    fn vision_tensor_clip_qkv_returns_true() {
        assert!(should_emit_f16_for_kquant(
            "vision_tower.encoder.layers.0.self_attn.k_proj.weight",
            1024,
        ));
    }

    /// Aligned language tensor with a non-256-multiple row_len fires
    /// the defensive arm. Synthetic name (no real model has 1153-dim
    /// attn_q today; the predicate is forward-looking).
    #[test]
    fn non_256_multiple_returns_true() {
        assert!(should_emit_f16_for_kquant("blk.0.attn_q.weight", 1153));
    }

    /// Standard 4096-dim attn_q in `blk.<N>.…` GGUF naming — pure
    /// language tensor, perfectly aligned. Must NOT trigger.
    #[test]
    fn aligned_language_tensor_returns_false() {
        assert!(!should_emit_f16_for_kquant("blk.0.attn_q.weight", 4096));
    }

    /// Boundary case: row_len exactly equal to QK_K = 256. Must be
    /// treated as aligned (the defensive arm uses `% 256 != 0`).
    #[test]
    fn aligned_language_tensor_at_256_returns_false() {
        assert!(!should_emit_f16_for_kquant("blk.0.attn_q.weight", 256));
    }

    /// Coverage for every vision-pattern variant in the predicate so a
    /// future regression that drops one of them surfaces here directly.
    /// All names use aligned row_len = 256 so only the vision arm can
    /// be the trigger — proves each pattern fires independently.
    #[test]
    fn every_vision_pattern_variant_returns_true() {
        // model.visual.* (Qwen3.6 raw HF naming)
        assert!(should_emit_f16_for_kquant(
            "model.visual.blocks.0.attn.proj.weight",
            256,
        ));
        // vision_tower.* (CLIP / SigLIP raw HF naming)
        assert!(should_emit_f16_for_kquant(
            "vision_tower.encoder.layers.0.self_attn.q_proj.weight",
            256,
        ));
        // vision_model.* (HF AutoModel naming)
        assert!(should_emit_f16_for_kquant(
            "vision_model.encoder.layers.5.mlp.fc1.weight",
            256,
        ));
        // vit.* (DeiT / ViT-original naming)
        assert!(should_emit_f16_for_kquant(
            "vit.encoder.layer.0.attention.self.query.weight",
            256,
        ));
        // visual.* (top-level prefix-stripped naming)
        assert!(should_emit_f16_for_kquant(
            "visual.blocks.0.attn.proj.weight",
            256,
        ));
        // *.visual.* (mid-name substring — handles `language_model.visual.…`
        // and other prefix-prepended layouts)
        assert!(should_emit_f16_for_kquant(
            "language_model.visual.blocks.0.attn.proj.weight",
            256,
        ));
    }

    /// `output.weight` and `token_embd.weight` still match exactly —
    /// substring-matching for *other* names doesn't accidentally pull
    /// these into the wrong branch.
    #[test]
    fn classify_output_token_embd_exact_match_preserved() {
        assert_eq!(
            TensorCategory::classify("output.weight"),
            TensorCategory::Output
        );
        assert_eq!(
            TensorCategory::classify("token_embd.weight"),
            TensorCategory::TokenEmbd
        );
        // Crucially, "output.weight" doesn't classify via attn_output
        // substring (which it doesn't contain).  Sanity-check that
        // `attn_output.weight` does NOT classify as Output.
        assert_eq!(
            TensorCategory::classify("blk.0.attn_output.weight"),
            TensorCategory::AttentionOutput
        );
    }

    // ── iter-9: Q3_K_S/M/L policy tests ───────────────────────────

    /// Q3_K_S: minimal upgrades — base Q3_K everywhere except output
    /// (Q6_K bump per `llama-quant.cpp:439-460`).
    #[test]
    fn target_q3_k_s_minimal_upgrades() {
        assert_eq!(
            target_for(KQuantVariant::Q3_K_S, "output.weight", 0, 32),
            KQuantTarget::Q6K,
            "Q3_K_S output should bump to Q6_K"
        );
        // attn_v stays Q3_K base
        assert_eq!(
            target_for(KQuantVariant::Q3_K_S, "blk.0.attn_v.weight", 0, 32),
            KQuantTarget::Q3K,
        );
        // ffn_down stays Q3_K base
        assert_eq!(
            target_for(KQuantVariant::Q3_K_S, "blk.0.ffn_down.weight", 0, 32),
            KQuantTarget::Q3K,
        );
        assert_eq!(
            target_for(KQuantVariant::Q3_K_S, "blk.5.attn_q.weight", 5, 32),
            KQuantTarget::Q3K,
        );
    }

    /// Q3_K_M: attn_v `i_layer<2 → Q5_K, else → Q4_K`
    /// (`llama-quant.cpp:527-528`); ffn_down per :578-582.
    #[test]
    fn target_q3_k_m_attn_v_first_two_layers_q5k() {
        // layer 0, 1 → Q5_K
        assert_eq!(
            target_for(KQuantVariant::Q3_K_M, "blk.0.attn_v.weight", 0, 32),
            KQuantTarget::Q5K,
        );
        assert_eq!(
            target_for(KQuantVariant::Q3_K_M, "blk.1.attn_v.weight", 1, 32),
            KQuantTarget::Q5K,
        );
        // layer 2+ → Q4_K
        assert_eq!(
            target_for(KQuantVariant::Q3_K_M, "blk.2.attn_v.weight", 2, 32),
            KQuantTarget::Q4K,
        );
        assert_eq!(
            target_for(KQuantVariant::Q3_K_M, "blk.30.attn_v.weight", 30, 32),
            KQuantTarget::Q4K,
        );
    }

    /// Q3_K_M ffn_down policy:
    ///   i_layer < n_layer/16     → Q5_K
    ///   use_more_bits(i, n)      → Q4_K
    ///   else                     → Q3_K (base)
    #[test]
    fn target_q3_k_m_ffn_down_layer_bands() {
        const N: usize = 32; // n/16 = 2; n/8 = 4; 7n/8 = 28
        // i_layer = 0, 1 (i < n/16=2) → Q5_K
        assert_eq!(
            target_for(KQuantVariant::Q3_K_M, "blk.0.ffn_down.weight", 0, N),
            KQuantTarget::Q5K
        );
        assert_eq!(
            target_for(KQuantVariant::Q3_K_M, "blk.1.ffn_down.weight", 1, N),
            KQuantTarget::Q5K
        );
        // i_layer = 2, 3 (i ≥ n/16, but i < n/8 → use_more_bits=true) → Q4_K
        assert_eq!(
            target_for(KQuantVariant::Q3_K_M, "blk.2.ffn_down.weight", 2, N),
            KQuantTarget::Q4K
        );
        assert_eq!(
            target_for(KQuantVariant::Q3_K_M, "blk.3.ffn_down.weight", 3, N),
            KQuantTarget::Q4K
        );
        // i_layer = 30, 31 (last 1/8 → use_more_bits=true) → Q4_K
        assert_eq!(
            target_for(KQuantVariant::Q3_K_M, "blk.30.ffn_down.weight", 30, N),
            KQuantTarget::Q4K
        );
        // i_layer = 6 (every-3rd-after-eighth: (6-4)%3=2 → use_more_bits=true) → Q4_K
        assert_eq!(
            target_for(KQuantVariant::Q3_K_M, "blk.6.ffn_down.weight", 6, N),
            KQuantTarget::Q4K
        );
        // i_layer = 5 (middle, not every-3rd) → use_more_bits=false → Q3_K base
        assert_eq!(
            target_for(KQuantVariant::Q3_K_M, "blk.5.ffn_down.weight", 5, N),
            KQuantTarget::Q3K
        );
    }

    /// Q3_K_L: attn_v always Q5_K (`:530`); ffn_down always Q5_K (`:587-588`).
    #[test]
    fn target_q3_k_l_attn_v_and_ffn_down_q5k() {
        for i_layer in [0_usize, 5, 15, 30] {
            assert_eq!(
                target_for(
                    KQuantVariant::Q3_K_L,
                    &format!("blk.{i_layer}.attn_v.weight"),
                    i_layer,
                    32
                ),
                KQuantTarget::Q5K,
                "Q3_K_L attn_v at i={i_layer} should be Q5_K"
            );
            assert_eq!(
                target_for(
                    KQuantVariant::Q3_K_L,
                    &format!("blk.{i_layer}.ffn_down.weight"),
                    i_layer,
                    32
                ),
                KQuantTarget::Q5K,
                "Q3_K_L ffn_down at i={i_layer} should be Q5_K"
            );
        }
        // output bumps to Q6_K
        assert_eq!(
            target_for(KQuantVariant::Q3_K_L, "output.weight", 0, 32),
            KQuantTarget::Q6K,
        );
        // attn_q stays Q3_K base
        assert_eq!(
            target_for(KQuantVariant::Q3_K_L, "blk.5.attn_q.weight", 5, 32),
            KQuantTarget::Q3K,
        );
    }
}
