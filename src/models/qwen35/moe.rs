//! Qwen3.5-MoE conversion — tensor naming, expert merge, and metadata.
//!
//! Covers Qwen3.5-MoE-35B-A3.9B (256 experts, top-8) and similar variants.
//!
//! ADR-012 Decisions 4, 7, 9:
//!   `hf_tensor_name_to_gguf_moe` — maps HF tensor names to GGUF names.
//!   `merge_expert_tensors` — stacks 256 per-expert weight tensors into
//!     a single merged tensor (body deferred to P5).
//!   `emit_metadata_moe` — writes GGUF metadata keys (body deferred to P4).
//!
//! This file does NOT implement:
//! - Expert merge body (P5)
//! - Metadata emission body (P4)
//! - Dispatch wiring in `src/backends/gguf.rs` (P4)

use super::{ConvertError, Qwen35ConvertContext};
use crate::ir::TensorRef;

// ---------------------------------------------------------------------------
// Tensor name mapping
// ---------------------------------------------------------------------------

/// Map a HuggingFace tensor name to its GGUF counterpart for Qwen3.5-MoE.
///
/// Returns `None` if the tensor should be skipped.
/// Returns `Some(gguf_name)` for tensors that map to GGUF.
///
/// # MoE FFN naming (ADR-012 Decision 4, 7)
///
/// Per-expert weights:
///   HF:   `model.layers.{N}.mlp.experts.{E}.{gate,up,down}_proj.weight`
///   GGUF: `blk.{N}.ffn_{gate,up,down}_exps.weight` (merged; E dimension merged by P5)
///
/// Shared expert:
///   HF:   `model.layers.{N}.mlp.shared_expert.{gate,up,down}_proj.weight`
///   GGUF: `blk.{N}.ffn_{gate,up,down}_shexp.weight`
///
/// Router:
///   HF:   `model.layers.{N}.mlp.gate.weight`
///   GGUF: `blk.{N}.ffn_gate_inp.weight`
///
/// Full-attention and linear-attention layers use the same naming as
/// `hf_tensor_name_to_gguf_dense` (shared logic is factored into helpers).
pub fn hf_tensor_name_to_gguf_moe(
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

    // --- Full-attention projection (same as dense) ---
    if let Some(name) = map_full_attn_suffix(suffix, &blk) {
        return Some(name);
    }

    // --- Linear-attention projection (same as dense) ---
    if let Some(name) = map_linear_attn_suffix(suffix, &blk) {
        return Some(name);
    }

    // --- MoE router ---
    if suffix == "mlp.gate.weight" {
        return Some(format!("{}.ffn_gate_inp.weight", blk));
    }

    // --- Per-expert weights (gate / up / down) ---
    // HF: "mlp.experts.{E}.{gate,up,down}_proj.weight"
    if let Some(name) = map_per_expert_suffix(suffix, &blk) {
        return Some(name);
    }

    // --- Shared expert ---
    if let Some(name) = map_shared_expert_suffix(suffix, &blk) {
        return Some(name);
    }

    // --- Layer norms ---
    if let Some(name) = map_norm_suffix(suffix, &blk) {
        return Some(name);
    }

    // Unknown — tag it.
    Some(format!("unk.{}", hf_name))
}

/// Map a per-expert tensor suffix → GGUF merged-expert name.
///
/// The expert index `E` is encoded in the suffix but the GGUF name uses a
/// single merged tensor (P5 handles the merge).  The name here uses the
/// `_exps` convention matching llama.cpp's MoE naming.
fn map_per_expert_suffix(suffix: &str, blk: &str) -> Option<String> {
    // Pattern: "mlp.experts.{E}.{gate,up,down}_proj.weight"
    let rest = suffix.strip_prefix("mlp.experts.")?;
    let dot = rest.find('.')?;
    let _expert_idx: usize = rest[..dot].parse().ok()?;
    let proj_suffix = &rest[dot + 1..];

    let gguf = match proj_suffix {
        "gate_proj.weight" => format!("{}.ffn_gate_exps.weight", blk),
        "up_proj.weight"   => format!("{}.ffn_up_exps.weight", blk),
        "down_proj.weight" => format!("{}.ffn_down_exps.weight", blk),
        _ => return None,
    };
    Some(gguf)
}

/// Map shared-expert tensor suffix → GGUF name.
fn map_shared_expert_suffix(suffix: &str, blk: &str) -> Option<String> {
    let rest = suffix.strip_prefix("mlp.shared_expert.")?;
    let gguf = match rest {
        "gate_proj.weight" => format!("{}.ffn_gate_shexp.weight", blk),
        "up_proj.weight"   => format!("{}.ffn_up_shexp.weight", blk),
        "down_proj.weight" => format!("{}.ffn_down_shexp.weight", blk),
        _ => return None,
    };
    Some(gguf)
}

/// Map full-attention suffix → GGUF name (shared with dense).
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

/// Map linear-attention suffix → GGUF name (shared with dense).
fn map_linear_attn_suffix(suffix: &str, blk: &str) -> Option<String> {
    let rest = suffix.strip_prefix("linear_attn.")?;
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
    Some(gguf)
}

/// Map layer-norm / RMS-norm suffix → GGUF name (shared with dense).
fn map_norm_suffix(suffix: &str, blk: &str) -> Option<String> {
    match suffix {
        "input_layernorm.weight"          => Some(format!("{}.attn_norm.weight", blk)),
        "post_attention_layernorm.weight" => Some(format!("{}.ffn_norm.weight", blk)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Expert merge (P5 stub)
// ---------------------------------------------------------------------------

/// Stack per-expert weight tensors into a single merged GGUF tensor.
///
/// # Phase stub (P5)
///
/// The merge body — stacking N expert tensors along dim 0 to produce
/// `[num_experts, out_features, in_features]` — is scheduled for P5
/// (ADR-012 Decision 9).
///
/// # Signature contract
///
/// `expert_tensors`: ordered by expert index 0..N.
/// All tensors must have the same shape and dtype.
///
/// Returns a single `TensorRef` whose first dimension is `num_experts`.
///
/// Callers MUST check the error return.
#[allow(dead_code)]
pub fn merge_expert_tensors(
    expert_tensors: &[TensorRef],
) -> Result<TensorRef, ConvertError> {
    // Validate early even though we don't implement yet — so the caller
    // knows the contract is being checked.
    if expert_tensors.is_empty() {
        return Err(ConvertError::PhaseStub {
            phase: "P5",
            what: "merge_expert_tensors: called with empty slice",
        });
    }
    Err(ConvertError::PhaseStub {
        phase: "P5",
        what: "merge_expert_tensors body — stacks N expert tensors along dim 0",
    })
}

// ---------------------------------------------------------------------------
// Metadata emission (P4 stub)
// ---------------------------------------------------------------------------

/// Write Qwen3.5-MoE–specific GGUF metadata keys.
///
/// # Phase stub (P4)
///
/// The body — writing `qwen35moe.*` keys including `expert_count`,
/// `expert_used_count`, `expert_feed_forward_length`,
/// `expert_shared_feed_forward_length` — is scheduled for P4.
///
/// Callers MUST check the error return.
#[allow(dead_code)]
pub fn emit_metadata_moe(
    _ctx: &Qwen35ConvertContext,
) -> Result<(), ConvertError> {
    Err(ConvertError::PhaseStub {
        phase: "P4",
        what: "emit_metadata_moe body — writes qwen35moe.* GGUF metadata keys",
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen35::{Qwen35Arch, Qwen35ConvertContext};

    fn moe_ctx() -> Qwen35ConvertContext {
        let layer_types: Vec<String> = (0..40_usize)
            .map(|i| {
                if (i + 1) % 4 == 0 {
                    "full_attention".to_string()
                } else {
                    "linear_attention".to_string()
                }
            })
            .collect();

        Qwen35ConvertContext {
            arch: Qwen35Arch::Moe,
            layer_types,
            num_layers: 40,
            num_attention_heads: 40,
            num_kv_heads: 8,
            head_dim: 256,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 128,
            linear_num_key_heads: 16,
            linear_value_head_dim: 128,
            linear_num_value_heads: 32,
            linear_num_v_per_k: 2,
            moe_intermediate_size: Some(2048),
            shared_expert_intermediate_size: Some(512),
            num_experts: Some(256),
            top_k_experts: Some(8),
            intermediate_size: None,
        }
    }

    #[test]
    fn embed_tokens_maps_correctly() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.embed_tokens.weight", &ctx),
            Some("token_embd.weight".to_string())
        );
    }

    #[test]
    fn router_maps_correctly() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.0.mlp.gate.weight", &ctx),
            Some("blk.0.ffn_gate_inp.weight".to_string())
        );
    }

    #[test]
    fn per_expert_gate_proj_maps_correctly() {
        let ctx = moe_ctx();
        // Expert 0 gate_proj
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.5.mlp.experts.0.gate_proj.weight", &ctx),
            Some("blk.5.ffn_gate_exps.weight".to_string())
        );
        // Expert 255 gate_proj (max expert index)
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.5.mlp.experts.255.gate_proj.weight", &ctx),
            Some("blk.5.ffn_gate_exps.weight".to_string())
        );
    }

    #[test]
    fn per_expert_up_proj_maps_correctly() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.0.mlp.experts.7.up_proj.weight", &ctx),
            Some("blk.0.ffn_up_exps.weight".to_string())
        );
    }

    #[test]
    fn per_expert_down_proj_maps_correctly() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.2.mlp.experts.128.down_proj.weight", &ctx),
            Some("blk.2.ffn_down_exps.weight".to_string())
        );
    }

    #[test]
    fn shared_expert_maps_correctly() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe(
                "model.layers.1.mlp.shared_expert.gate_proj.weight",
                &ctx
            ),
            Some("blk.1.ffn_gate_shexp.weight".to_string())
        );
        assert_eq!(
            hf_tensor_name_to_gguf_moe(
                "model.layers.1.mlp.shared_expert.up_proj.weight",
                &ctx
            ),
            Some("blk.1.ffn_up_shexp.weight".to_string())
        );
        assert_eq!(
            hf_tensor_name_to_gguf_moe(
                "model.layers.1.mlp.shared_expert.down_proj.weight",
                &ctx
            ),
            Some("blk.1.ffn_down_shexp.weight".to_string())
        );
    }

    #[test]
    fn full_attn_q_proj_maps_correctly() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.3.self_attn.q_proj.weight", &ctx),
            Some("blk.3.attn_q.weight".to_string())
        );
    }

    #[test]
    fn linear_attn_out_proj_maps_correctly() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe(
                "model.layers.0.linear_attn.out_proj.weight",
                &ctx
            ),
            Some("blk.0.lin_attn_out_proj.weight".to_string())
        );
    }

    #[test]
    fn merge_expert_tensors_returns_phase_stub_empty() {
        let err = merge_expert_tensors(&[])
            .expect_err("merge_expert_tensors([]) should return PhaseStub");
        assert!(
            matches!(
                err,
                crate::models::qwen35::ConvertError::PhaseStub { phase: "P5", .. }
            ),
            "expected PhaseStub(P5), got: {err}"
        );
    }

    #[test]
    fn merge_expert_tensors_returns_phase_stub_non_empty() {
        use crate::ir::DType;
        let t = TensorRef {
            name: "test_expert".to_string(),
            shape: vec![2048, 4096],
            dtype: DType::F32,
            data: vec![0u8; 2048 * 4096 * 4],
        };
        let err = merge_expert_tensors(&[t])
            .expect_err("merge_expert_tensors should return PhaseStub");
        assert!(
            matches!(
                err,
                crate::models::qwen35::ConvertError::PhaseStub { phase: "P5", .. }
            ),
            "expected PhaseStub(P5), got: {err}"
        );
    }

    #[test]
    fn emit_metadata_moe_returns_phase_stub() {
        let ctx = moe_ctx();
        let err = emit_metadata_moe(&ctx).expect_err("emit_metadata_moe should return PhaseStub");
        assert!(
            matches!(
                err,
                crate::models::qwen35::ConvertError::PhaseStub { phase: "P4", .. }
            ),
            "expected PhaseStub(P4), got: {err}"
        );
    }
}
