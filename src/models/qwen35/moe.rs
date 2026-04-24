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

    // --- MoE router (sparse experts) ---
    // tensor_mapping.py:443 + llama-arch.cpp:393: mlp.gate.weight → blk.N.ffn_gate_inp.weight
    if suffix == "mlp.gate.weight" {
        return Some(format!("{}.ffn_gate_inp.weight", blk));
    }

    // --- Shared-expert scalar gate ---
    // tensor_mapping.py:447 + llama-arch.cpp:394: mlp.shared_expert_gate → blk.N.ffn_gate_inp_shexp.weight
    // llama-model.cpp:7596: layer.ffn_gate_inp_shexp = create_tensor(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", i)
    if suffix == "mlp.shared_expert_gate.weight" {
        return Some(format!("{}.ffn_gate_inp_shexp.weight", blk));
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
///
/// Same mapping table as dense — identical linear-attention architecture in
/// both variants.  See `dense.rs:map_linear_attn_suffix` for full citation chain.
///
/// | HF suffix             | GGUF suffix        | Source |
/// |---|---|---|
/// | in_proj_qkv.weight    | attn_qkv.weight    | llama-arch.cpp:382 |
/// | in_proj_z.weight      | attn_gate.weight   | llama-arch.cpp:370 |
/// | in_proj_a.weight      | ssm_alpha.weight   | llama-arch.cpp:400 |
/// | in_proj_b.weight      | ssm_beta.weight    | llama-arch.cpp:416 |
/// | out_proj.weight       | ssm_out.weight     | llama-arch.cpp:402 |
/// | A_log                 | ssm_a              | llama-arch.cpp:395 |
/// | dt_bias               | ssm_dt.bias        | llama-arch.cpp:397 + convert_hf_to_gguf.py:4791 |
/// | conv1d.weight         | ssm_conv1d.weight  | llama-arch.cpp:396 |
/// | norm.weight           | ssm_norm.weight    | llama-arch.cpp:401 |
fn map_linear_attn_suffix(suffix: &str, blk: &str) -> Option<String> {
    let rest = suffix.strip_prefix("linear_attn.")?;
    let gguf = match rest {
        // llama-arch.cpp:382 LLM_TENSOR_ATTN_QKV → "blk.%d.attn_qkv"
        "in_proj_qkv.weight" => format!("{}.attn_qkv.weight", blk),
        // llama-arch.cpp:370 LLM_TENSOR_ATTN_GATE → "blk.%d.attn_gate"
        "in_proj_z.weight"   => format!("{}.attn_gate.weight", blk),
        // llama-arch.cpp:400 LLM_TENSOR_SSM_ALPHA → "blk.%d.ssm_alpha"
        "in_proj_a.weight"   => format!("{}.ssm_alpha.weight", blk),
        // llama-arch.cpp:416 LLM_TENSOR_SSM_BETA → "blk.%d.ssm_beta"
        "in_proj_b.weight"   => format!("{}.ssm_beta.weight", blk),
        // llama-arch.cpp:402 LLM_TENSOR_SSM_OUT → "blk.%d.ssm_out"
        "out_proj.weight"    => format!("{}.ssm_out.weight", blk),
        // llama-arch.cpp:395 LLM_TENSOR_SSM_A_NOSCAN → "blk.%d.ssm_a"
        "A_log"              => format!("{}.ssm_a", blk),
        // convert_hf_to_gguf.py:4791 dt_bias → dt_proj.bias; llama-arch.cpp:397 → "blk.%d.ssm_dt.bias"
        "dt_bias"            => format!("{}.ssm_dt.bias", blk),
        // llama-arch.cpp:396 LLM_TENSOR_SSM_CONV1D → "blk.%d.ssm_conv1d"
        "conv1d.weight"      => format!("{}.ssm_conv1d.weight", blk),
        "conv1d.bias"        => format!("{}.ssm_conv1d.bias", blk),
        // llama-arch.cpp:401 LLM_TENSOR_SSM_NORM → "blk.%d.ssm_norm"
        "norm.weight"        => format!("{}.ssm_norm.weight", blk),
        _ => return None,
    };
    Some(gguf)
}

/// Map layer-norm / RMS-norm suffix → GGUF name (shared with dense).
///
/// # post_attention_layernorm verdict (ADR-012 P4 audit)
///
/// Source: llama-model.cpp:7564-7565 (qwen35moe tensor load):
///   `layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), …)`
///   `layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), …)`
///
/// Source: llama-arch.cpp:367:
///   `{ LLM_TENSOR_ATTN_POST_NORM, "blk.%d.post_attention_norm" }`
///
/// Qwen3.5-MoE uses `attn_post_norm` (GGUF: `post_attention_norm`), NOT `ffn_norm`.
/// The MoE variant follows the exact same norm pattern as dense — same field names,
/// same GGUF key.  `ffn_norm` would cause llama.cpp to reject the file.
fn map_norm_suffix(suffix: &str, blk: &str) -> Option<String> {
    match suffix {
        "input_layernorm.weight" => Some(format!("{}.attn_norm.weight", blk)),
        // llama-model.cpp:7565 loads layer.attn_post_norm via LLM_TENSOR_ATTN_POST_NORM
        // llama-arch.cpp:367: LLM_TENSOR_ATTN_POST_NORM → "blk.%d.post_attention_norm"
        "post_attention_layernorm.weight" => {
            Some(format!("{}.post_attention_norm.weight", blk))
        }
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
// Metadata emission (P4 — implemented)
// ---------------------------------------------------------------------------

/// Validate the Qwen3.5-MoE convert context and confirm all required
/// metadata hparams for the MoE variant are present.
///
/// # P4 implementation (ADR-012 Decision 7)
///
/// The actual GGUF key-value emission for `qwen35moe.*` keys is performed by
/// `src/backends/gguf.rs::build_metadata_qwen35moe`.  This function validates
/// the context; callers then emit KVs inline.
///
/// ## Keys emitted by the caller (all prefixed `qwen35moe.*`):
/// Shared keys (same as qwen35 dense, arch prefix changes):
/// - `qwen35moe.block_count`, `.context_length`, `.embedding_length`
/// - `qwen35moe.attention.head_count`, `.head_count_kv`, `.key_length`, `.value_length`
/// - `qwen35moe.attention.layer_norm_rms_epsilon`  — llama-model.cpp:2837 (mandatory)
/// - `qwen35moe.rope.freq_base`, `.dimension_count`, `.dimension_sections`
///   — llama-model.cpp:2839 (mandatory via get_key_or_arr)
/// - `qwen35moe.full_attention_interval`            — llama-model.cpp:2851 (optional, default 4)
/// - `qwen35moe.ssm.conv_kernel`                    — llama-model.cpp:2842 (mandatory)
/// - `qwen35moe.ssm.inner_size`                     — llama-model.cpp:2843 (mandatory)
/// - `qwen35moe.ssm.state_size`                     — llama-model.cpp:2844 (mandatory)
/// - `qwen35moe.ssm.time_step_rank`                 — llama-model.cpp:2845 (mandatory)
/// - `qwen35moe.ssm.group_count`                    — llama-model.cpp:2846 (mandatory)
/// - `qwen35moe.nextn_predict_layers`               — llama-arch.cpp:194 (optional, default 0)
///
/// MoE-only keys:
/// - `qwen35moe.expert_count`                       — llama-arch.cpp:182 (LLM_KV_EXPERT_COUNT)
/// - `qwen35moe.expert_used_count`                  — llama-arch.cpp:183 (LLM_KV_EXPERT_USED_COUNT)
/// - `qwen35moe.expert_feed_forward_length`         — llama-arch.cpp:175 / llama-model.cpp:2835 (optional)
/// - `qwen35moe.expert_shared_feed_forward_length`  — llama-arch.cpp:176 / llama-model.cpp:2836 (optional)
pub fn emit_metadata_moe(ctx: &Qwen35ConvertContext) -> Result<(), ConvertError> {
    // Validate MoE-specific required fields
    if ctx.num_experts.is_none() {
        return Err(ConvertError::MissingHparam {
            field: "num_experts (required for qwen35moe expert_count)",
        });
    }
    if ctx.top_k_experts.is_none() {
        return Err(ConvertError::MissingHparam {
            field: "top_k_experts (required for qwen35moe expert_used_count)",
        });
    }
    if ctx.moe_intermediate_size.is_none() {
        return Err(ConvertError::MissingHparam {
            field: "moe_intermediate_size (required for qwen35moe expert_feed_forward_length)",
        });
    }
    Ok(())
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
            Some("blk.0.ssm_out.weight".to_string()) // llama-arch.cpp:402 LLM_TENSOR_SSM_OUT
        );
    }

    /// dt_bias maps to ssm_dt.bias in GGUF (ADR-012 Decision 8 P4 implementation).
    ///
    /// convert_hf_to_gguf.py:4791 renames ".dt_bias" → ".dt_proj.bias" before GGUF mapping.
    /// llama-arch.cpp:397 LLM_TENSOR_SSM_DT → "blk.%d.ssm_dt" with suffix "bias".
    /// Full GGUF key: "blk.N.ssm_dt.bias".
    #[test]
    fn dt_bias_maps_to_ssm_dt_bias_moe() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.0.linear_attn.dt_bias", &ctx),
            Some("blk.0.ssm_dt.bias".to_string()),
            "dt_bias must map to ssm_dt.bias (llama-arch.cpp:397, convert_hf_to_gguf.py:4791)"
        );
    }

    /// post_attention_layernorm maps to post_attention_norm (ADR-012 P4 verdict).
    ///
    /// llama-model.cpp:7565 loads layer.attn_post_norm via LLM_TENSOR_ATTN_POST_NORM.
    /// llama-arch.cpp:367: LLM_TENSOR_ATTN_POST_NORM → "blk.%d.post_attention_norm".
    #[test]
    fn post_attention_layernorm_maps_to_post_attention_norm_moe() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.3.post_attention_layernorm.weight", &ctx),
            Some("blk.3.post_attention_norm.weight".to_string()),
            "post_attention_layernorm must map to post_attention_norm (llama-arch.cpp:367)"
        );
    }

    /// in_proj_qkv maps to attn_qkv (ADR-012 Decision 8).
    #[test]
    fn in_proj_qkv_maps_to_attn_qkv_moe() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.0.linear_attn.in_proj_qkv.weight", &ctx),
            Some("blk.0.attn_qkv.weight".to_string()),
        );
    }

    /// in_proj_z maps to attn_gate (ADR-012 Decision 8).
    #[test]
    fn in_proj_z_maps_to_attn_gate_moe() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.0.linear_attn.in_proj_z.weight", &ctx),
            Some("blk.0.attn_gate.weight".to_string()),
        );
    }

    /// shared_expert_gate maps to ffn_gate_inp_shexp (ADR-012 Decision 8).
    /// tensor_mapping.py:447 + llama-arch.cpp:394
    #[test]
    fn shared_expert_gate_maps_correctly() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.0.mlp.shared_expert_gate.weight", &ctx),
            Some("blk.0.ffn_gate_inp_shexp.weight".to_string()),
            "shared_expert_gate must map to ffn_gate_inp_shexp (tensor_mapping.py:447)"
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

    /// emit_metadata_moe returns Ok when context is valid (P4 implementation).
    #[test]
    fn emit_metadata_moe_returns_ok_when_valid() {
        let ctx = moe_ctx();
        assert!(
            emit_metadata_moe(&ctx).is_ok(),
            "emit_metadata_moe should return Ok for a valid MoE context with all required fields"
        );
    }

    /// emit_metadata_moe returns error when num_experts is absent.
    #[test]
    fn emit_metadata_moe_errors_when_missing_num_experts() {
        let mut ctx = moe_ctx();
        ctx.num_experts = None;
        let err = emit_metadata_moe(&ctx)
            .expect_err("emit_metadata_moe should return error when num_experts is None");
        assert!(
            matches!(err, crate::models::qwen35::ConvertError::MissingHparam { .. }),
            "expected MissingHparam, got: {err}"
        );
    }

    /// emit_metadata_moe returns error when moe_intermediate_size is absent.
    #[test]
    fn emit_metadata_moe_errors_when_missing_moe_ff_size() {
        let mut ctx = moe_ctx();
        ctx.moe_intermediate_size = None;
        let err = emit_metadata_moe(&ctx)
            .expect_err("emit_metadata_moe should return error when moe_intermediate_size is None");
        assert!(
            matches!(err, crate::models::qwen35::ConvertError::MissingHparam { .. }),
            "expected MissingHparam, got: {err}"
        );
    }
}
