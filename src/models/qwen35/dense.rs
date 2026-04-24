//! Qwen3.5 Dense (non-MoE) conversion — tensor naming and metadata.
//!
//! Covers Qwen3.5-27B-Instruct and similar single-expert variants.
//!
//! ADR-012 Decision 4:
//!   `hf_tensor_name_to_gguf_dense` maps HF tensor names to GGUF names.
//!   `emit_metadata_dense` writes GGUF metadata keys (body deferred to P4).
//!
//! This file does NOT implement:
//! - Metadata emission body (P4)
//! - Expert merge (P5 — MoE only)
//! - Dispatch wiring in `src/backends/gguf.rs` (P4)

use super::{ConvertError, Qwen35ConvertContext};

// ---------------------------------------------------------------------------
// Tensor name mapping
// ---------------------------------------------------------------------------

/// Map a HuggingFace tensor name to its GGUF counterpart for Qwen3.5 Dense.
///
/// Returns `None` if the tensor should be skipped (not emitted to GGUF).
/// Returns `Some(gguf_name)` for tensors that map to GGUF.
///
/// # Dense FFN naming (ADR-012 Decision 4)
///
/// The standard MLX/HF naming uses:
///   `model.layers.{N}.mlp.{gate,up,down}_proj.weight`
///
/// GGUF convention:
///   `blk.{N}.ffn_{gate,up,down}.weight`
///
/// Full-attention layers follow the same Q/K/V/O projection naming as any
/// transformer model.  Linear-attention layers follow the `linear_attn.*`
/// naming described in `mod.rs`.
///
/// # Unrecognised names
/// Tensor names that don't match any known pattern return the original name
/// with a `"unk."` prefix so they are identifiable in the output but not
/// silently dropped.
///
/// # Keys reserved for P4
/// `embed_tokens`, `lm_head`, and `norm` tensors are returned with their
/// tentative GGUF names but metadata emission (ordering, arch keys) is P4.
pub fn hf_tensor_name_to_gguf_dense(
    hf_name: &str,
    _ctx: &Qwen35ConvertContext,
) -> Option<String> {
    // Embeddings
    if hf_name == "model.embed_tokens.weight" {
        return Some("token_embd.weight".to_string());
    }
    // Final norm
    if hf_name == "model.norm.weight" {
        return Some("output_norm.weight".to_string());
    }
    // LM head
    if hf_name == "lm_head.weight" {
        return Some("output.weight".to_string());
    }

    // Layer tensors — extract layer index.
    let rest = hf_name.strip_prefix("model.layers.")?;
    let dot = rest.find('.')?;
    let layer_str = &rest[..dot];
    let layer_idx: usize = layer_str.parse().ok()?;
    let suffix = &rest[dot + 1..]; // everything after "model.layers.N."

    let blk = format!("blk.{}", layer_idx);

    // --- Full-attention projection ---
    if let Some(name) = map_full_attn_suffix(suffix, &blk) {
        return Some(name);
    }

    // --- Linear-attention projection ---
    if let Some(name) = map_linear_attn_suffix(suffix, &blk) {
        return Some(name);
    }

    // --- Dense FFN ---
    if let Some(name) = map_dense_ffn_suffix(suffix, &blk) {
        return Some(name);
    }

    // --- Layer norms ---
    if let Some(name) = map_norm_suffix(suffix, &blk) {
        return Some(name);
    }

    // Unknown — tag it so it's identifiable in the output.
    Some(format!("unk.{}", hf_name))
}

/// Map full-attention suffix → GGUF name.
fn map_full_attn_suffix(suffix: &str, blk: &str) -> Option<String> {
    match suffix {
        "self_attn.q_proj.weight" => Some(format!("{}.attn_q.weight", blk)),
        "self_attn.k_proj.weight" => Some(format!("{}.attn_k.weight", blk)),
        "self_attn.v_proj.weight" => Some(format!("{}.attn_v.weight", blk)),
        "self_attn.o_proj.weight" => Some(format!("{}.attn_output.weight", blk)),
        "self_attn.q_norm.weight" => Some(format!("{}.attn_q_norm.weight", blk)),
        "self_attn.k_norm.weight" => Some(format!("{}.attn_k_norm.weight", blk)),
        _ => None,
    }
}

/// Map linear-attention suffix → GGUF name.
fn map_linear_attn_suffix(suffix: &str, blk: &str) -> Option<String> {
    // in_proj_qkv / in_proj_z / in_proj_a / in_proj_b
    if let Some(rest) = suffix.strip_prefix("linear_attn.") {
        let gguf = match rest {
            "in_proj_qkv.weight" => format!("{}.lin_attn_in_proj_qkv.weight", blk),
            "in_proj_z.weight"   => format!("{}.lin_attn_in_proj_z.weight", blk),
            "in_proj_a.weight"   => format!("{}.lin_attn_in_proj_a.weight", blk),
            "in_proj_b.weight"   => format!("{}.lin_attn_in_proj_b.weight", blk),
            "out_proj.weight"    => format!("{}.lin_attn_out_proj.weight", blk),
            "A_log"              => format!("{}.lin_attn_A_log", blk),
            "dt_bias"            => format!("{}.lin_attn_dt_bias", blk),
            "dt_proj.weight"     => format!("{}.lin_attn_dt_proj.weight", blk),
            "conv1d.weight"      => format!("{}.lin_attn_conv1d.weight", blk),
            "conv1d.bias"        => format!("{}.lin_attn_conv1d.bias", blk),
            "norm.weight"        => format!("{}.lin_attn_norm.weight", blk),
            _ => return None,
        };
        return Some(gguf);
    }
    None
}

/// Map dense FFN suffix → GGUF name.
fn map_dense_ffn_suffix(suffix: &str, blk: &str) -> Option<String> {
    match suffix {
        "mlp.gate_proj.weight" => Some(format!("{}.ffn_gate.weight", blk)),
        "mlp.up_proj.weight"   => Some(format!("{}.ffn_up.weight", blk)),
        "mlp.down_proj.weight" => Some(format!("{}.ffn_down.weight", blk)),
        _ => None,
    }
}

/// Map layer-norm / RMS-norm suffix → GGUF name.
fn map_norm_suffix(suffix: &str, blk: &str) -> Option<String> {
    match suffix {
        "input_layernorm.weight"          => Some(format!("{}.attn_norm.weight", blk)),
        "post_attention_layernorm.weight" => Some(format!("{}.ffn_norm.weight", blk)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Metadata emission (P4 stub)
// ---------------------------------------------------------------------------

/// Write Qwen3.5 Dense–specific GGUF metadata keys.
///
/// # Phase stub (P4)
///
/// The metadata emission body — writing `qwen35.*` keys including
/// `feed_forward_length` and all dense-variant architecture keys — is
/// scheduled for P4 (ADR-012 Decision 8).
///
/// This function signature is stable and will be wired in P4 without an
/// API change.  Callers MUST check the error return.
///
/// Returns `Err(ConvertError::PhaseStub { phase: "P4", … })`.
#[allow(dead_code)]
pub fn emit_metadata_dense(
    _ctx: &Qwen35ConvertContext,
) -> Result<(), ConvertError> {
    Err(ConvertError::PhaseStub {
        phase: "P4",
        what: "emit_metadata_dense body — writes qwen35.* GGUF metadata keys",
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen35::{Qwen35Arch, Qwen35ConvertContext};

    fn dense_ctx() -> Qwen35ConvertContext {
        Qwen35ConvertContext {
            arch: Qwen35Arch::Dense,
            layer_types: vec!["linear_attention".to_string(), "full_attention".to_string()],
            num_layers: 2,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 128,
            linear_num_key_heads: 16,
            linear_value_head_dim: 128,
            linear_num_value_heads: 32,
            linear_num_v_per_k: 2,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: Some(8192),
        }
    }

    #[test]
    fn embed_tokens_maps_correctly() {
        let ctx = dense_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_dense("model.embed_tokens.weight", &ctx),
            Some("token_embd.weight".to_string())
        );
    }

    #[test]
    fn lm_head_maps_correctly() {
        let ctx = dense_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_dense("lm_head.weight", &ctx),
            Some("output.weight".to_string())
        );
    }

    #[test]
    fn dense_ffn_gate_proj_maps_correctly() {
        let ctx = dense_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_dense("model.layers.0.mlp.gate_proj.weight", &ctx),
            Some("blk.0.ffn_gate.weight".to_string())
        );
    }

    #[test]
    fn dense_ffn_up_proj_maps_correctly() {
        let ctx = dense_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_dense("model.layers.5.mlp.up_proj.weight", &ctx),
            Some("blk.5.ffn_up.weight".to_string())
        );
    }

    #[test]
    fn dense_ffn_down_proj_maps_correctly() {
        let ctx = dense_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_dense("model.layers.3.mlp.down_proj.weight", &ctx),
            Some("blk.3.ffn_down.weight".to_string())
        );
    }

    #[test]
    fn full_attn_q_proj_maps_correctly() {
        let ctx = dense_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_dense("model.layers.1.self_attn.q_proj.weight", &ctx),
            Some("blk.1.attn_q.weight".to_string())
        );
    }

    #[test]
    fn linear_attn_in_proj_qkv_maps_correctly() {
        let ctx = dense_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_dense(
                "model.layers.0.linear_attn.in_proj_qkv.weight",
                &ctx
            ),
            Some("blk.0.lin_attn_in_proj_qkv.weight".to_string())
        );
    }

    #[test]
    fn input_layernorm_maps_correctly() {
        let ctx = dense_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_dense("model.layers.2.input_layernorm.weight", &ctx),
            Some("blk.2.attn_norm.weight".to_string())
        );
    }

    /// dt_bias maps to lin_attn_dt_bias in GGUF (dense).
    ///
    /// The `.dt_bias` → `.dt_proj.bias` GGUF rename happens at the name-mapping
    /// layer per `convert_hf_to_gguf.py:4790-4791`. This test asserts the current
    /// mapping key is `lin_attn_dt_bias`; P4 will adjust the GGUF key to match
    /// llama.cpp convention.
    #[test]
    fn dt_bias_maps_to_gguf_name_dense() {
        let ctx = dense_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_dense("model.layers.0.linear_attn.dt_bias", &ctx),
            Some("blk.0.lin_attn_dt_bias".to_string())
        );
    }

    #[test]
    fn emit_metadata_dense_returns_phase_stub() {
        let ctx = dense_ctx();
        let err = emit_metadata_dense(&ctx)
            .expect_err("emit_metadata_dense should return PhaseStub");
        assert!(
            matches!(err, crate::models::qwen35::ConvertError::PhaseStub { phase: "P4", .. }),
            "expected PhaseStub(P4), got: {err}"
        );
    }
}
