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
///
/// # GGUF tensor name mapping (ADR-012 Decision 8, P3 contract)
///
/// HF `linear_attn.*` → GGUF `blk.N.*` mapping per llama-arch.cpp and
/// the convert_hf_to_gguf.py Qwen3Next/Qwen3_5 class hierarchy:
///
/// | HF suffix             | GGUF suffix        | Source |
/// |---|---|---|
/// | in_proj_qkv.weight    | attn_qkv.weight    | llama-arch.cpp:382 (LLM_TENSOR_ATTN_QKV) |
/// | in_proj_z.weight      | attn_gate.weight   | llama-arch.cpp:370 (LLM_TENSOR_ATTN_GATE) |
/// | in_proj_a.weight      | ssm_alpha.weight   | llama-arch.cpp:400 (LLM_TENSOR_SSM_ALPHA) |
/// | in_proj_b.weight      | ssm_beta.weight    | llama-arch.cpp:416 (LLM_TENSOR_SSM_BETA) |
/// | out_proj.weight       | ssm_out.weight     | llama-arch.cpp:402 (LLM_TENSOR_SSM_OUT) |
/// | A_log                 | ssm_a              | llama-arch.cpp:395 (LLM_TENSOR_SSM_A_NOSCAN) |
/// | dt_bias               | ssm_dt.bias        | llama-arch.cpp:397 (LLM_TENSOR_SSM_DT "bias") |
/// |                       |                    | convert_hf_to_gguf.py:4791 renames dt_bias→dt_proj.bias first |
/// | conv1d.weight         | ssm_conv1d.weight  | llama-arch.cpp:396 (LLM_TENSOR_SSM_CONV1D) |
/// | norm.weight           | ssm_norm.weight    | llama-arch.cpp:401 (LLM_TENSOR_SSM_NORM) |
///
/// Note: `dt_proj.weight` does NOT appear as a separate HF tensor in qwen35;
/// the convert script synthesises it from dt_bias only.  If encountered, mapped
/// to `ssm_dt.weight` for forward-compat (llama.cpp may ignore it).
fn map_linear_attn_suffix(suffix: &str, blk: &str) -> Option<String> {
    if let Some(rest) = suffix.strip_prefix("linear_attn.") {
        let gguf = match rest {
            // llama-model.cpp:7641  layer.wqkv  → LLM_TENSOR_ATTN_QKV "weight"
            // llama-arch.cpp:382   LLM_TENSOR_ATTN_QKV → "blk.%d.attn_qkv"
            "in_proj_qkv.weight" => format!("{}.attn_qkv.weight", blk),
            // llama-model.cpp:7642  layer.wqkv_gate → LLM_TENSOR_ATTN_GATE "weight"
            // llama-arch.cpp:370   LLM_TENSOR_ATTN_GATE → "blk.%d.attn_gate"
            "in_proj_z.weight"   => format!("{}.attn_gate.weight", blk),
            // llama-model.cpp:7647  layer.ssm_alpha → LLM_TENSOR_SSM_ALPHA "weight"
            // llama-arch.cpp:400   LLM_TENSOR_SSM_ALPHA → "blk.%d.ssm_alpha"
            "in_proj_a.weight"   => format!("{}.ssm_alpha.weight", blk),
            // llama-model.cpp:7646  layer.ssm_beta → LLM_TENSOR_SSM_BETA "weight"
            // llama-arch.cpp:416   LLM_TENSOR_SSM_BETA → "blk.%d.ssm_beta"
            "in_proj_b.weight"   => format!("{}.ssm_beta.weight", blk),
            // llama-model.cpp:7649  layer.ssm_out → LLM_TENSOR_SSM_OUT "weight"
            // llama-arch.cpp:402   LLM_TENSOR_SSM_OUT → "blk.%d.ssm_out"
            "out_proj.weight"    => format!("{}.ssm_out.weight", blk),
            // llama-model.cpp:7645  layer.ssm_a → LLM_TENSOR_SSM_A_NOSCAN (no suffix)
            // llama-arch.cpp:395   LLM_TENSOR_SSM_A_NOSCAN → "blk.%d.ssm_a"
            "A_log"              => format!("{}.ssm_a", blk),
            // convert_hf_to_gguf.py:4791 renames ".dt_bias" → ".dt_proj.bias" before mapping.
            // llama-model.cpp:7644  layer.ssm_dt → LLM_TENSOR_SSM_DT "bias"
            // llama-arch.cpp:397   LLM_TENSOR_SSM_DT → "blk.%d.ssm_dt"  → full: "blk.%d.ssm_dt.bias"
            // P3 contract: dt_bias → dt_proj.bias name-map (ADR-012 Decision 8)
            "dt_bias"            => format!("{}.ssm_dt.bias", blk),
            // llama-model.cpp:7643  layer.ssm_conv1d → LLM_TENSOR_SSM_CONV1D "weight"
            // llama-arch.cpp:396   LLM_TENSOR_SSM_CONV1D → "blk.%d.ssm_conv1d"
            "conv1d.weight"      => format!("{}.ssm_conv1d.weight", blk),
            "conv1d.bias"        => format!("{}.ssm_conv1d.bias", blk),
            // llama-model.cpp:7648  layer.ssm_norm → LLM_TENSOR_SSM_NORM "weight"
            // llama-arch.cpp:401   LLM_TENSOR_SSM_NORM → "blk.%d.ssm_norm"
            "norm.weight"        => format!("{}.ssm_norm.weight", blk),
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
///
/// # post_attention_layernorm verdict (ADR-012 P4 audit)
///
/// Source: llama-model.cpp:7627-7628 (qwen35 dense tensor load):
///   `layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), …)`
///   `layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), …)`
///
/// Source: llama-arch.cpp:367:
///   `{ LLM_TENSOR_ATTN_POST_NORM, "blk.%d.post_attention_norm" }`
///
/// Qwen3.5 Dense uses `attn_post_norm` (GGUF: `post_attention_norm`), NOT `ffn_norm`.
/// There is NO separate `LLM_TENSOR_FFN_NORM` load for qwen35 — the FFN pre-norm role
/// is played by `attn_post_norm`.  Using `ffn_norm` here would cause llama.cpp to
/// reject the GGUF with "missing tensor: blk.N.post_attention_norm".
fn map_norm_suffix(suffix: &str, blk: &str) -> Option<String> {
    match suffix {
        "input_layernorm.weight" => Some(format!("{}.attn_norm.weight", blk)),
        // llama-model.cpp:7628 loads layer.attn_post_norm via LLM_TENSOR_ATTN_POST_NORM
        // llama-arch.cpp:367: LLM_TENSOR_ATTN_POST_NORM → "blk.%d.post_attention_norm"
        "post_attention_layernorm.weight" => Some(format!("{}.post_attention_norm.weight", blk)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Metadata emission (P4 — implemented)
// ---------------------------------------------------------------------------

/// Validate the Qwen3.5 Dense convert context and confirm all required
/// metadata hparams are present.
///
/// # P4 implementation (ADR-012 Decision 7)
///
/// The actual GGUF key-value emission for `qwen35.*` keys is performed by
/// `src/backends/gguf.rs::build_metadata_qwen35` which calls this function
/// to validate the context before emitting keys.  This function returns `Ok(())`
/// when the context is valid; callers then proceed to emit KVs inline.
///
/// ## Keys emitted by the caller (all prefixed `qwen35.*`):
/// - `qwen35.block_count`                        — llama-arch.cpp:172 (LLM_KV_BLOCK_COUNT)
/// - `qwen35.context_length`                     — llama-arch.cpp:167 (LLM_KV_CONTEXT_LENGTH)
/// - `qwen35.embedding_length`                   — llama-arch.cpp:168 (LLM_KV_EMBEDDING_LENGTH)
/// - `qwen35.feed_forward_length`                — llama-arch.cpp:174 (LLM_KV_FEED_FORWARD_LENGTH)
/// - `qwen35.attention.head_count`               — llama-arch.cpp:213 (LLM_KV_ATTENTION_HEAD_COUNT)
/// - `qwen35.attention.head_count_kv`            — llama-arch.cpp:214 (LLM_KV_ATTENTION_HEAD_COUNT_KV)
/// - `qwen35.attention.key_length`               — llama-arch.cpp:217 (LLM_KV_ATTENTION_KEY_LENGTH)
/// - `qwen35.attention.value_length`             — llama-arch.cpp:218 (LLM_KV_ATTENTION_VALUE_LENGTH)
/// - `qwen35.attention.layer_norm_rms_epsilon`   — llama-arch.cpp:220 / llama-model.cpp:2807 (mandatory)
/// - `qwen35.rope.freq_base`                     — llama-arch.cpp:249 (LLM_KV_ROPE_FREQ_BASE)
/// - `qwen35.rope.dimension_count`               — llama-arch.cpp:246 (LLM_KV_ROPE_DIMENSION_COUNT)
/// - `qwen35.rope.dimension_sections`            — llama-arch.cpp:248 / llama-model.cpp:2808 (mandatory)
/// - `qwen35.full_attention_interval`            — llama-arch.cpp:211 / llama-model.cpp:2820 (default 4)
/// - `qwen35.ssm.conv_kernel`                    — llama-arch.cpp:268 / llama-model.cpp:2811 (mandatory)
/// - `qwen35.ssm.inner_size`                     — llama-arch.cpp:269 / llama-model.cpp:2812 (mandatory)
/// - `qwen35.ssm.state_size`                     — llama-arch.cpp:270 / llama-model.cpp:2813 (mandatory)
/// - `qwen35.ssm.time_step_rank`                 — llama-arch.cpp:271 / llama-model.cpp:2814 (mandatory)
/// - `qwen35.ssm.group_count`                    — llama-arch.cpp:272 / llama-model.cpp:2815 (mandatory)
/// - `qwen35.nextn_predict_layers`               — llama-arch.cpp:194 (optional, default 0)
pub fn emit_metadata_dense(ctx: &Qwen35ConvertContext) -> Result<(), ConvertError> {
    // Validate dense-specific required field
    if ctx.intermediate_size.is_none() {
        return Err(ConvertError::MissingHparam {
            field: "intermediate_size (required for qwen35 dense feed_forward_length)",
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

    /// P3 test updated in P4: in_proj_qkv maps to attn_qkv (not lin_attn_in_proj_qkv).
    /// llama-arch.cpp:382 LLM_TENSOR_ATTN_QKV → "blk.%d.attn_qkv".
    #[test]
    fn linear_attn_in_proj_qkv_maps_correctly() {
        let ctx = dense_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_dense(
                "model.layers.0.linear_attn.in_proj_qkv.weight",
                &ctx
            ),
            Some("blk.0.attn_qkv.weight".to_string())
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

    /// post_attention_layernorm maps to post_attention_norm (ADR-012 P4 verdict).
    ///
    /// llama-model.cpp:7628 loads layer.attn_post_norm via LLM_TENSOR_ATTN_POST_NORM.
    /// llama-arch.cpp:367: LLM_TENSOR_ATTN_POST_NORM → "blk.%d.post_attention_norm".
    /// Previous P3 stub mapping to ffn_norm was incorrect.
    #[test]
    fn post_attention_layernorm_maps_to_post_attention_norm_dense() {
        let ctx = dense_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_dense("model.layers.1.post_attention_layernorm.weight", &ctx),
            Some("blk.1.post_attention_norm.weight".to_string()),
            "post_attention_layernorm must map to post_attention_norm, not ffn_norm \
             (llama-arch.cpp:367, llama-model.cpp:7628)"
        );
    }

    /// dt_bias maps to ssm_dt.bias (ADR-012 Decision 8 P4 implementation).
    ///
    /// convert_hf_to_gguf.py:4791 renames ".dt_bias" → ".dt_proj.bias" before GGUF mapping.
    /// llama-arch.cpp:397 LLM_TENSOR_SSM_DT → "blk.%d.ssm_dt" with suffix "bias".
    /// Full GGUF key: "blk.N.ssm_dt.bias".
    #[test]
    fn dt_bias_maps_to_ssm_dt_bias_dense() {
        let ctx = dense_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_dense("model.layers.0.linear_attn.dt_bias", &ctx),
            Some("blk.0.ssm_dt.bias".to_string()),
            "dt_bias must map to ssm_dt.bias (llama-arch.cpp:397, convert_hf_to_gguf.py:4791)"
        );
    }

    /// in_proj_qkv maps to attn_qkv (ADR-012 Decision 8).
    /// llama-arch.cpp:382 LLM_TENSOR_ATTN_QKV → "blk.%d.attn_qkv".
    #[test]
    fn in_proj_qkv_maps_to_attn_qkv_dense() {
        let ctx = dense_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_dense("model.layers.0.linear_attn.in_proj_qkv.weight", &ctx),
            Some("blk.0.attn_qkv.weight".to_string()),
            "in_proj_qkv must map to attn_qkv (llama-arch.cpp:382)"
        );
    }

    /// in_proj_z (gate) maps to attn_gate (ADR-012 Decision 8).
    /// llama-arch.cpp:370 LLM_TENSOR_ATTN_GATE → "blk.%d.attn_gate".
    #[test]
    fn in_proj_z_maps_to_attn_gate_dense() {
        let ctx = dense_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_dense("model.layers.3.linear_attn.in_proj_z.weight", &ctx),
            Some("blk.3.attn_gate.weight".to_string()),
            "in_proj_z must map to attn_gate (llama-arch.cpp:370)"
        );
    }

    /// in_proj_a maps to ssm_alpha (ADR-012 Decision 8).
    /// llama-arch.cpp:400 LLM_TENSOR_SSM_ALPHA → "blk.%d.ssm_alpha".
    #[test]
    fn in_proj_a_maps_to_ssm_alpha_dense() {
        let ctx = dense_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_dense("model.layers.0.linear_attn.in_proj_a.weight", &ctx),
            Some("blk.0.ssm_alpha.weight".to_string()),
            "in_proj_a must map to ssm_alpha (llama-arch.cpp:400)"
        );
    }

    /// in_proj_b maps to ssm_beta (ADR-012 Decision 8).
    /// llama-arch.cpp:416 LLM_TENSOR_SSM_BETA → "blk.%d.ssm_beta".
    #[test]
    fn in_proj_b_maps_to_ssm_beta_dense() {
        let ctx = dense_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_dense("model.layers.0.linear_attn.in_proj_b.weight", &ctx),
            Some("blk.0.ssm_beta.weight".to_string()),
            "in_proj_b must map to ssm_beta (llama-arch.cpp:416)"
        );
    }

    /// out_proj maps to ssm_out (ADR-012 Decision 8).
    /// llama-arch.cpp:402 LLM_TENSOR_SSM_OUT → "blk.%d.ssm_out".
    #[test]
    fn out_proj_maps_to_ssm_out_dense() {
        let ctx = dense_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_dense("model.layers.1.linear_attn.out_proj.weight", &ctx),
            Some("blk.1.ssm_out.weight".to_string()),
            "out_proj must map to ssm_out (llama-arch.cpp:402)"
        );
    }

    /// A_log maps to ssm_a (ADR-012 Decision 8).
    /// llama-arch.cpp:395 LLM_TENSOR_SSM_A_NOSCAN → "blk.%d.ssm_a".
    #[test]
    fn a_log_maps_to_ssm_a_dense() {
        let ctx = dense_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_dense("model.layers.0.linear_attn.A_log", &ctx),
            Some("blk.0.ssm_a".to_string()),
            "A_log must map to ssm_a (llama-arch.cpp:395)"
        );
    }

    /// conv1d maps to ssm_conv1d (ADR-012 Decision 8).
    /// llama-arch.cpp:396 LLM_TENSOR_SSM_CONV1D → "blk.%d.ssm_conv1d".
    #[test]
    fn conv1d_maps_to_ssm_conv1d_dense() {
        let ctx = dense_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_dense("model.layers.0.linear_attn.conv1d.weight", &ctx),
            Some("blk.0.ssm_conv1d.weight".to_string()),
            "conv1d must map to ssm_conv1d (llama-arch.cpp:396)"
        );
    }

    /// norm maps to ssm_norm (ADR-012 Decision 8).
    /// llama-arch.cpp:401 LLM_TENSOR_SSM_NORM → "blk.%d.ssm_norm".
    #[test]
    fn norm_maps_to_ssm_norm_dense() {
        let ctx = dense_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_dense("model.layers.0.linear_attn.norm.weight", &ctx),
            Some("blk.0.ssm_norm.weight".to_string()),
            "linear_attn.norm must map to ssm_norm (llama-arch.cpp:401)"
        );
    }

    /// emit_metadata_dense returns Ok when context is valid (P4 implementation).
    #[test]
    fn emit_metadata_dense_returns_ok_when_valid() {
        let ctx = dense_ctx();
        assert!(
            emit_metadata_dense(&ctx).is_ok(),
            "emit_metadata_dense should return Ok for a valid dense context with intermediate_size"
        );
    }

    /// emit_metadata_dense returns error when intermediate_size is absent.
    #[test]
    fn emit_metadata_dense_errors_when_missing_ff_size() {
        let mut ctx = dense_ctx();
        ctx.intermediate_size = None;
        let err = emit_metadata_dense(&ctx)
            .expect_err("emit_metadata_dense should return error when intermediate_size is None");
        assert!(
            matches!(err, crate::models::qwen35::ConvertError::MissingHparam { .. }),
            "expected MissingHparam, got: {err}"
        );
    }
}
