//! Storage admission derived from the existing native loaders.

pub(crate) fn tensor_incompatibility(
    arch: &str,
    name: &str,
    ggml_type: mlx_native::ops::quantized_matmul_ggml::GgmlType,
) -> Option<String> {
    use crate::inference::models::qwen35::forward_gpu::qwen35_native_embedding_type_supported;
    use crate::inference::models::qwen35::weight_loader::{
        qwen35_dense_ffn_type_supported, qwen35_moe_expert_type_supported,
        qwen35_native_projection_type_supported,
    };
    use crate::inference::models::qwen3vl_text::weights::qwen3vl_projection_type_supported;

    if matches!(arch, "qwen35" | "qwen35moe") {
        let supported = if name == "token_embd.weight" {
            qwen35_native_embedding_type_supported(ggml_type)
        } else if qwen35_dense_ffn_name(name) {
            qwen35_dense_ffn_type_supported(ggml_type)
        } else if qwen35_moe_expert_name(name) {
            qwen35_moe_expert_type_supported(ggml_type)
        } else if qwen35_native_projection_name(name) {
            qwen35_native_projection_type_supported(ggml_type)
        } else {
            true
        };
        if !supported {
            return Some(format!(
                "{name} uses unsupported {ggml_type:?} storage for {arch}"
            ));
        }
    } else if (arch == "qwen3_vl" || arch == "qwen3vl")
        && qwen3vl_projection_name(name)
        && !qwen3vl_projection_type_supported(ggml_type)
    {
        return Some(format!(
            "{name} uses unsupported {ggml_type:?} storage for {arch}"
        ));
    }
    None
}

fn qwen35_native_projection_name(name: &str) -> bool {
    name == "output.weight"
        || [
            ".attn_q.weight",
            ".attn_k.weight",
            ".attn_v.weight",
            ".attn_output.weight",
            ".attn_qkv.weight",
            ".attn_gate.weight",
            ".ssm_alpha.weight",
            ".ssm_beta.weight",
            ".ssm_out.weight",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

fn qwen35_dense_ffn_name(name: &str) -> bool {
    [".ffn_gate.weight", ".ffn_up.weight", ".ffn_down.weight"]
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

fn qwen35_moe_expert_name(name: &str) -> bool {
    [
        ".ffn_gate_exps.weight",
        ".ffn_up_exps.weight",
        ".ffn_down_exps.weight",
    ]
    .iter()
    .any(|suffix| name.ends_with(suffix))
}

fn qwen3vl_projection_name(name: &str) -> bool {
    [
        ".attn_q.weight",
        ".attn_k.weight",
        ".attn_v.weight",
        ".attn_output.weight",
        ".ffn_gate.weight",
        ".ffn_up.weight",
        ".ffn_down.weight",
    ]
    .iter()
    .any(|suffix| name.ends_with(suffix))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_native::GgmlType;

    #[test]
    fn storage_admission_uses_the_loader_for_each_tensor_role() {
        assert!(tensor_incompatibility("qwen35", "token_embd.weight", GgmlType::Q3_K).is_some());
        assert!(
            tensor_incompatibility("qwen35", "blk.0.ffn_gate.weight", GgmlType::Q5_1).is_some()
        );
        assert!(
            tensor_incompatibility("qwen35moe", "blk.0.ffn_gate_exps.weight", GgmlType::Q3_K)
                .is_none()
        );
        assert!(tensor_incompatibility("qwen35", "blk.0.attn_q.weight", GgmlType::F16).is_some());
        assert!(
            tensor_incompatibility("qwen35", "blk.0.attn_q.weight", GgmlType::IQ4_XS).is_none()
        );
    }
}
