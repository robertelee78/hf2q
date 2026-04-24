//! Qwen3.5 27B dense-variant inference.
//!
//! Ownership:
//! - Dense SwiGLU FFN (`gate_proj`, `up_proj`, `down_proj`).
//! - Dense tensor-name table resolution.
//! - Dense forward entry point.
//!
//! The linear-attention branch, gated full-attention branch, MROPE
//! dispatch, hybrid KV cache, and tokenizer are shared with
//! [`super::moe`] and live in [`super`].
//!
//! Implementation phases: P9 (FFN), P11 (forward wire-up), P13
//! (correctness gate) — see ADR-013.

// ======================================================================
// Tensor-name table (ADR-013 Decision 12(a), dense variant)
// ======================================================================
//
// Dense Qwen3.5-27B shares every structural tensor with the MoE variant
// EXCEPT the FFN: dense uses SwiGLU (gate/up/down projections per layer)
// while MoE uses the 256-expert router + shared expert.
//
// The per-linear-layer and per-full-layer norm+attention+ssm tensors are
// identical to the MoE variant; we re-export the relevant MoE constants
// plus override the FFN suffixes.

use super::moe::{FULL_LAYER_TENSOR_SUFFIXES, LINEAR_LAYER_TENSOR_SUFFIXES};

/// Dense-variant FFN tensor suffixes (SwiGLU).
///
/// Count = 3. Replaces the 8 MoE FFN tensors on the MoE side.
pub const DENSE_FFN_TENSOR_SUFFIXES: &[&str] = &[
    "ffn_gate.weight", // SwiGLU gate_proj
    "ffn_up.weight",   // SwiGLU up_proj
    "ffn_down.weight", // SwiGLU down_proj
];

/// Return the per-layer tensor suffixes for a dense-variant layer, with
/// MoE FFN tensors replaced by the 3 dense SwiGLU tensors.
///
/// The shared schema from `super::moe` has 8 MoE FFN suffixes at the end
/// of each layer's list; this helper splits them off and substitutes the
/// dense FFN set.
pub fn dense_layer_tensor_suffixes(kind: super::Qwen35LayerKind) -> Vec<String> {
    let base = match kind {
        super::Qwen35LayerKind::LinearAttention => LINEAR_LAYER_TENSOR_SUFFIXES,
        super::Qwen35LayerKind::FullAttention => FULL_LAYER_TENSOR_SUFFIXES,
    };
    // Strip the 8 MoE FFN suffixes (all ending in `_exps.weight` or `_shexp.weight`
    // or `_gate_inp.weight` / `_gate_inp_shexp.weight`).
    let non_moe: Vec<String> = base
        .iter()
        .filter(|s| {
            !(s.contains("_exps.weight")
                || s.contains("_shexp.weight")
                || s.ends_with("ffn_gate_inp.weight"))
        })
        .map(|s| s.to_string())
        .collect();
    let mut out = non_moe;
    for ffn in DENSE_FFN_TENSOR_SUFFIXES {
        out.push(ffn.to_string());
    }
    out
}

/// Full dense tensor names for a given layer index + kind.
pub fn tensor_names_for_layer(layer_idx: u32, kind: super::Qwen35LayerKind) -> Vec<String> {
    dense_layer_tensor_suffixes(kind)
        .into_iter()
        .map(|s| format!("blk.{layer_idx}.{s}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::Qwen35LayerKind;

    #[test]
    fn dense_ffn_has_swiglu_not_moe() {
        assert_eq!(DENSE_FFN_TENSOR_SUFFIXES.len(), 3);
        assert!(DENSE_FFN_TENSOR_SUFFIXES.contains(&"ffn_gate.weight"));
        assert!(DENSE_FFN_TENSOR_SUFFIXES.contains(&"ffn_up.weight"));
        assert!(DENSE_FFN_TENSOR_SUFFIXES.contains(&"ffn_down.weight"));
    }

    #[test]
    fn dense_full_layer_excludes_moe_ffn_tensors() {
        let names = tensor_names_for_layer(3, Qwen35LayerKind::FullAttention);
        assert!(!names.iter().any(|n| n.contains("_exps.weight")));
        assert!(!names.iter().any(|n| n.contains("_shexp.weight")));
        assert!(!names.iter().any(|n| n.ends_with(".ffn_gate_inp.weight")));
        // But does include SwiGLU.
        assert!(names.iter().any(|n| n == "blk.3.ffn_gate.weight"));
        assert!(names.iter().any(|n| n == "blk.3.ffn_up.weight"));
        assert!(names.iter().any(|n| n == "blk.3.ffn_down.weight"));
    }

    #[test]
    fn dense_linear_layer_keeps_ssm_tensors() {
        let names = tensor_names_for_layer(0, Qwen35LayerKind::LinearAttention);
        assert!(names.iter().any(|n| n == "blk.0.ssm_a"));
        assert!(names.iter().any(|n| n == "blk.0.ssm_out.weight"));
        assert!(names.iter().any(|n| n == "blk.0.attn_qkv.weight"));
        // No MoE.
        assert!(!names.iter().any(|n| n.contains("_exps.weight")));
    }
}
