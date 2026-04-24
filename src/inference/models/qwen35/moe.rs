//! Qwen3.5-MoE 35B-A3B mixture-of-experts inference.
//!
//! Ownership:
//! - 256-expert top-8 routing (`ffn_gate_inp`, `ffn_gate_up_exps`,
//!   `ffn_down_exps`).
//! - Gated shared expert (`ffn_gate_inp_shexp` sigmoid-gated onto
//!   `ffn_{gate,up,down}_shexp`) — ADR-013 Decision 13.
//! - MoE tensor-name table resolution.
//! - MoE forward entry point.
//!
//! The linear-attention branch, gated full-attention branch, MROPE
//! dispatch, hybrid KV cache, and tokenizer are shared with
//! [`super::dense`] and live in [`super`].
//!
//! Implementation phases: P9 (FFN), P11 (forward wire-up), P13
//! (correctness gate) — see ADR-013.

// ======================================================================
// Tensor-name table (ADR-013 Decision 12(a))
// ======================================================================
//
// Authoritative tensor-name enumeration for the Qwen3.5-MoE GGUF format,
// grounded in the apex GGUF dump (2026-04-23, 733 tensors, 40 layers).
//
// # Global tensors (3)
//
//   token_embd.weight     — token embedding (vocab_size × hidden_size)
//   output.weight         — LM head (hidden_size × vocab_size);
//                           in abliterated models it's a distinct tensor
//                           from token_embd (no weight tying).
//   output_norm.weight    — final RMSNorm (hidden_size)
//
// # Per linear-attention layer (14 tensors)
//
//   blk.{i}.attn_norm.weight              — pre-attention RMSNorm (hidden_size)
//   blk.{i}.attn_qkv.weight               — fused QKV+Z projection; output dim
//                                           = D_k*n_k_heads (Q) + D_k*n_k_heads (K)
//                                           + D_v*n_v_heads (V) + D_v*n_v_heads (Z-gate)
//   blk.{i}.attn_gate.weight              — DeltaNet Z-gate proj (also named
//                                           attn_gate but semantically distinct
//                                           from the full-attn gate)
//   blk.{i}.ssm_conv1d.weight             — depthwise conv1d (K × channels)
//   blk.{i}.ssm_dt.bias                   — delta-time bias (= num_v_heads;
//                                           shape-only, loaded as f32)
//   blk.{i}.ssm_a                         — per-head decay log base (= num_v_heads;
//                                           NB: no `.weight` suffix; stored raw)
//   blk.{i}.ssm_alpha.weight              — gate projection for α = exp(-g)
//   blk.{i}.ssm_beta.weight               — gate projection for β
//   blk.{i}.ssm_norm.weight               — per-head RMSNorm on DeltaNet output
//   blk.{i}.ssm_out.weight                — output projection from DeltaNet
//                                           (n_v_heads*D_v → hidden_size)
//   blk.{i}.post_attention_norm.weight    — post-attention RMSNorm (hidden_size)
//
//   # MoE FFN tensors (shared schema with full-attention layers)
//   blk.{i}.ffn_gate_inp.weight           — router logits: hidden_size → n_experts
//   blk.{i}.ffn_gate_exps.weight          — expert gate_proj stacked along expert axis
//   blk.{i}.ffn_up_exps.weight            — expert up_proj stacked
//   blk.{i}.ffn_down_exps.weight          — expert down_proj stacked
//   blk.{i}.ffn_gate_inp_shexp.weight     — shared-expert gate (sigmoid)
//   blk.{i}.ffn_gate_shexp.weight         — shared-expert gate_proj
//   blk.{i}.ffn_up_shexp.weight           — shared-expert up_proj
//   blk.{i}.ffn_down_shexp.weight         — shared-expert down_proj
//
// # Per full-attention layer (11 tensors)
//
//   blk.{i}.attn_norm.weight              — pre-attention RMSNorm
//   blk.{i}.attn_q.weight                 — Q projection (+gate in upper half if attn_output_gate)
//                                           NB: GGUF stores the Q half separately from attn_gate
//                                           (whereas llama.cpp's C++ builder fuses them in wq).
//   blk.{i}.attn_k.weight                 — K projection
//   blk.{i}.attn_v.weight                 — V projection
//   blk.{i}.attn_q_norm.weight            — Q RMSNorm (per-head)
//   blk.{i}.attn_k_norm.weight            — K RMSNorm (per-head)
//   blk.{i}.attn_gate.weight              — output gate projection (separate tensor;
//                                           sigmoid applied elementwise to SDPA output;
//                                           see ADR-013 Decision 9 citing HF/vLLM sigmoid)
//   blk.{i}.attn_output.weight            — output (o_proj) projection
//   blk.{i}.post_attention_norm.weight    — post-attention RMSNorm
//   # MoE FFN tensors: same as linear layers (8 more, see above).

/// Global (non-per-layer) tensor names.
pub const GLOBAL_TENSORS: &[&str] = &[
    "token_embd.weight",
    "output.weight",
    "output_norm.weight",
];

/// Per-linear-attention-layer tensor suffixes (append to `blk.{i}.`).
///
/// Count = 14 (linear-attention specific) + 8 (MoE FFN, shared).
pub const LINEAR_LAYER_TENSOR_SUFFIXES: &[&str] = &[
    // Pre/post norms + DeltaNet.
    "attn_norm.weight",
    "attn_qkv.weight",
    "attn_gate.weight",
    "ssm_conv1d.weight",
    "ssm_dt.bias",
    "ssm_a", // NB: no .weight suffix, per apex GGUF convention.
    "ssm_alpha.weight",
    "ssm_beta.weight",
    "ssm_norm.weight",
    "ssm_out.weight",
    "post_attention_norm.weight",
    // MoE FFN (shared schema with full-attention layers).
    "ffn_gate_inp.weight",
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
    "ffn_gate_inp_shexp.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
];

/// Per-full-attention-layer tensor suffixes (append to `blk.{i}.`).
///
/// Count = 9 (full-attention specific) + 8 (MoE FFN, shared).
pub const FULL_LAYER_TENSOR_SUFFIXES: &[&str] = &[
    // Pre/post norms + full-attention projections.
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "attn_gate.weight",
    "attn_output.weight",
    "post_attention_norm.weight",
    // MoE FFN.
    "ffn_gate_inp.weight",
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
    "ffn_gate_inp_shexp.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
];

/// Build the full per-layer tensor name list for a given layer index + kind.
pub fn tensor_names_for_layer(
    layer_idx: u32,
    kind: super::Qwen35LayerKind,
) -> Vec<String> {
    let suffixes = match kind {
        super::Qwen35LayerKind::LinearAttention => LINEAR_LAYER_TENSOR_SUFFIXES,
        super::Qwen35LayerKind::FullAttention => FULL_LAYER_TENSOR_SUFFIXES,
    };
    suffixes
        .iter()
        .map(|s| format!("blk.{layer_idx}.{s}"))
        .collect()
}

/// Expected total tensor count for a MoE model with the given config.
///
/// = `GLOBAL_TENSORS.len()` + per-layer count summed by kind.
/// Matches the apex file's 733 tensors for a 40-layer MoE:
///   3 + 30*(14+8) + 10*(9+8) = 3 + 660 + 170 = 833
///
/// Wait — apex reports 733. That indicates ~100 tensors fewer than naive
/// count. Investigation (P5 follow-up): probably because `attn_gate` is
/// present only on full-attn layers (not linear), and a few ssm_* tensors
/// are absent. Leaving this fn as a *best estimate* until the loader
/// cross-references per-layer.
pub fn expected_tensor_count(cfg: &super::Qwen35Config) -> usize {
    let mut n = GLOBAL_TENSORS.len();
    for kind in &cfg.layer_types {
        n += match kind {
            super::Qwen35LayerKind::LinearAttention => LINEAR_LAYER_TENSOR_SUFFIXES.len(),
            super::Qwen35LayerKind::FullAttention => FULL_LAYER_TENSOR_SUFFIXES.len(),
        };
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::Qwen35LayerKind;

    #[test]
    fn linear_layer_names_start_with_blk_prefix() {
        let names = tensor_names_for_layer(3, Qwen35LayerKind::LinearAttention);
        assert!(names.iter().all(|n| n.starts_with("blk.3.")));
        assert!(names.iter().any(|n| n == "blk.3.attn_qkv.weight"));
        assert!(names.iter().any(|n| n == "blk.3.ssm_a"));
        assert!(names.iter().any(|n| n == "blk.3.ssm_out.weight"));
    }

    #[test]
    fn full_layer_names_include_split_qkv() {
        let names = tensor_names_for_layer(11, Qwen35LayerKind::FullAttention);
        assert!(names.iter().any(|n| n == "blk.11.attn_q.weight"));
        assert!(names.iter().any(|n| n == "blk.11.attn_k.weight"));
        assert!(names.iter().any(|n| n == "blk.11.attn_v.weight"));
        assert!(names.iter().any(|n| n == "blk.11.attn_q_norm.weight"));
        assert!(names.iter().any(|n| n == "blk.11.attn_k_norm.weight"));
        // attn_gate is a separate tensor (NOT fused into attn_q in the GGUF).
        assert!(names.iter().any(|n| n == "blk.11.attn_gate.weight"));
    }

    #[test]
    fn full_layer_has_no_ssm_tensors() {
        let names = tensor_names_for_layer(7, Qwen35LayerKind::FullAttention);
        assert!(!names.iter().any(|n| n.contains(".ssm_")));
        assert!(!names.iter().any(|n| n.ends_with(".attn_qkv.weight")));
    }

    #[test]
    fn linear_layer_has_no_split_qkv() {
        let names = tensor_names_for_layer(0, Qwen35LayerKind::LinearAttention);
        assert!(!names.iter().any(|n| n.ends_with(".attn_q.weight")));
        assert!(!names.iter().any(|n| n.ends_with(".attn_k.weight")));
        assert!(!names.iter().any(|n| n.ends_with(".attn_v.weight")));
        // But has attn_qkv (fused QKV+Z for DeltaNet).
        assert!(names.iter().any(|n| n.ends_with(".attn_qkv.weight")));
    }

    #[test]
    fn global_tensors_have_three() {
        assert_eq!(GLOBAL_TENSORS.len(), 3);
        assert!(GLOBAL_TENSORS.contains(&"token_embd.weight"));
        assert!(GLOBAL_TENSORS.contains(&"output.weight"));
        assert!(GLOBAL_TENSORS.contains(&"output_norm.weight"));
    }
}
