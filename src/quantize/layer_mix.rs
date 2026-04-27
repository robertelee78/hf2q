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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KQuantVariant {
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
            Self::Q4_K_S | Self::Q4_K_M => KQuantTarget::Q4K,
            Self::Q5_K_S | Self::Q5_K_M => KQuantTarget::Q5K,
            Self::Q6_K => KQuantTarget::Q6K,
        }
    }

    /// Canonical name for logging / `quant_info.method` strings.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Q4_K_S => "Q4_K_S",
            Self::Q4_K_M => "Q4_K_M",
            Self::Q5_K_S => "Q5_K_S",
            Self::Q5_K_M => "Q5_K_M",
            Self::Q6_K => "Q6_K",
        }
    }
}

/// Tensor category — coarse classification of GGUF weight tensors,
/// used to route per-tensor targets. Matches the subset of llama.cpp's
/// `tensor_category` enum the policy actually branches on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorCategory {
    /// `output.weight` (lm_head). With tied embeddings, also covers
    /// `token_embd.weight` per `llama-quant.cpp:439`.
    Output,
    /// `token_embd.weight` (separate from output for non-tied archs).
    TokenEmbd,
    /// `blk.<i>.attn_v.weight` — value projection in attention.
    AttentionV,
    /// `blk.<i>.attn_k.weight` — key projection in attention.
    AttentionK,
    /// `blk.<i>.attn_q.weight` — query projection in attention.
    AttentionQ,
    /// `blk.<i>.ffn_down.weight` — FFN down-projection (the per-layer
    /// special case for `_K_M` upgrades).
    FfnDown,
    /// Any other weight tensor (norm, gate, up, etc.) — gets the
    /// variant's base target unchanged.
    Other,
}

impl TensorCategory {
    /// Classify a tensor by its GGUF name (post-`map_tensor_name_to_gguf`
    /// transformation). Recognises the standard `blk.<i>.<role>.weight`
    /// + `output.weight` + `token_embd.weight` patterns.
    pub fn classify(tensor_name: &str) -> Self {
        if tensor_name == "output.weight" {
            return Self::Output;
        }
        if tensor_name == "token_embd.weight" {
            return Self::TokenEmbd;
        }
        // Strip `blk.<N>.` prefix.
        let suffix = tensor_name.strip_prefix("blk.").and_then(|s| {
            let dot = s.find('.')?;
            Some(&s[dot + 1..])
        });
        if let Some(role) = suffix {
            match role {
                "attn_v.weight" => Self::AttentionV,
                "attn_k.weight" => Self::AttentionK,
                "attn_q.weight" => Self::AttentionQ,
                "ffn_down.weight" => Self::FfnDown,
                _ => Self::Other,
            }
        } else {
            Self::Other
        }
    }
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
            KQuantVariant::Q4_K_M | KQuantVariant::Q5_K_M => KQuantTarget::Q6K,
            _ => base,
        },
        TensorCategory::AttentionV => match variant {
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
            _ => base,
        },
        TensorCategory::AttentionK
        | TensorCategory::AttentionQ
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
        let err = KQuantVariant::parse("Q3_K_M").unwrap_err();
        match err {
            LayerMixError::UnknownVariant { variant } => assert_eq!(variant, "Q3_K_M"),
        }
        assert!(KQuantVariant::parse("IQ4_NL").is_err());
        assert!(KQuantVariant::parse("garbage").is_err());
    }

    #[test]
    fn variant_base_targets() {
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
        assert_eq!(TensorCategory::classify("blk.0.ffn_up.weight"), TensorCategory::Other);
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
}
