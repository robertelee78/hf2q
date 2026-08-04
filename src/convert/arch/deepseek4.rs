//! Strict DeepSeek-V4 official-HF to canonical GGUF tensor mapping.

use super::qwen35moe::{ExpertKind, MappedTensor};

fn direct(name: impl Into<String>) -> Option<MappedTensor> {
    Some(MappedTensor::Direct(name.into()))
}

/// Map one official checkpoint tensor. Unknown names return `None` and
/// the driver turns that into `ConvertError::UnmappedTensor`.
pub fn map_tensor_name(hf_name: &str) -> Option<MappedTensor> {
    let root = match hf_name {
        "embed.weight" => Some("token_embd.weight"),
        "norm.weight" => Some("output_norm.weight"),
        "head.weight" => Some("output.weight"),
        "hc_head_fn" => Some("output_hc_fn.weight"),
        "hc_head_base" => Some("output_hc_base.weight"),
        "hc_head_scale" => Some("output_hc_scale.weight"),
        _ => None,
    };
    if let Some(name) = root {
        return direct(name);
    }

    let stripped = hf_name.strip_prefix("layers.")?;
    let (layer_raw, rest) = stripped.split_once('.')?;
    let layer: usize = layer_raw.parse().ok()?;
    if layer.to_string() != layer_raw {
        return None;
    }

    let suffix = match rest {
        "hc_attn_fn" => Some("hc_attn_fn.weight"),
        "hc_attn_base" => Some("hc_attn_base.weight"),
        "hc_attn_scale" => Some("hc_attn_scale.weight"),
        "hc_ffn_fn" => Some("hc_ffn_fn.weight"),
        "hc_ffn_base" => Some("hc_ffn_base.weight"),
        "hc_ffn_scale" => Some("hc_ffn_scale.weight"),
        "attn.attn_sink" => Some("attn_sinks.weight"),
        "attn.wq_a.weight" => Some("attn_q_a.weight"),
        "attn.wq_b.weight" => Some("attn_q_b.weight"),
        "attn.q_norm.weight" => Some("attn_q_a_norm.weight"),
        "attn.wkv.weight" => Some("attn_kv.weight"),
        "attn.kv_norm.weight" => Some("attn_kv_a_norm.weight"),
        "attn.wo_a.weight" => Some("attn_output_a.weight"),
        "attn.wo_b.weight" => Some("attn_output_b.weight"),
        "attn.compressor.ape" => Some("attn_compressor_ape.weight"),
        "attn.compressor.wkv.weight" => Some("attn_compressor_kv.weight"),
        "attn.compressor.wgate.weight" => Some("attn_compressor_gate.weight"),
        "attn.compressor.norm.weight" => Some("attn_compressor_norm.weight"),
        "attn.indexer.wq_b.weight" => Some("indexer.attn_q_b.weight"),
        "attn.indexer.weights_proj.weight" => Some("indexer.proj.weight"),
        "attn.indexer.compressor.ape" => Some("indexer_compressor_ape.weight"),
        "attn.indexer.compressor.wkv.weight" => Some("indexer_compressor_kv.weight"),
        "attn.indexer.compressor.wgate.weight" => Some("indexer_compressor_gate.weight"),
        "attn.indexer.compressor.norm.weight" => Some("indexer_compressor_norm.weight"),
        "attn_norm.weight" => Some("attn_norm.weight"),
        "ffn_norm.weight" => Some("ffn_norm.weight"),
        "ffn.gate.weight" => Some("ffn_gate_inp.weight"),
        "ffn.gate.bias" => Some("exp_probs_b.bias"),
        "ffn.gate.tid2eid" => Some("ffn_gate_tid2eid.weight"),
        "ffn.shared_experts.w1.weight" => Some("ffn_gate_shexp.weight"),
        "ffn.shared_experts.w2.weight" => Some("ffn_down_shexp.weight"),
        "ffn.shared_experts.w3.weight" => Some("ffn_up_shexp.weight"),
        _ => None,
    };
    if let Some(suffix) = suffix {
        return direct(format!("blk.{layer}.{suffix}"));
    }

    let expert = rest.strip_prefix("ffn.experts.")?;
    let (expert_raw, projection) = expert.split_once('.')?;
    let expert_index: usize = expert_raw.parse().ok()?;
    if expert_index.to_string() != expert_raw {
        return None;
    }
    let (kind, suffix) = match projection {
        "w1.weight" => (ExpertKind::Gate, "ffn_gate_exps.weight"),
        "w2.weight" => (ExpertKind::Down, "ffn_down_exps.weight"),
        "w3.weight" => (ExpertKind::Up, "ffn_up_exps.weight"),
        // Sidecar scales are consumed by the Rust source reader.
        "w1.scale" | "w2.scale" | "w3.scale" => return Some(MappedTensor::Drop),
        _ => return None,
    };
    Some(MappedTensor::ExpertGroup {
        gguf_name: format!("blk.{layer}.{suffix}"),
        layer,
        expert_index,
        kind,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_official_roots_attention_hash_and_experts() {
        assert_eq!(map_tensor_name("embed.weight"), direct("token_embd.weight"));
        assert_eq!(
            map_tensor_name("layers.4.attn.indexer.weights_proj.weight"),
            direct("blk.4.indexer.proj.weight")
        );
        assert_eq!(
            map_tensor_name("layers.0.ffn.gate.tid2eid"),
            direct("blk.0.ffn_gate_tid2eid.weight")
        );
        assert!(matches!(
            map_tensor_name("layers.7.ffn.experts.12.w2.weight"),
            Some(MappedTensor::ExpertGroup {
                layer: 7,
                expert_index: 12,
                kind: ExpertKind::Down,
                ..
            })
        ));
    }

    #[test]
    fn malformed_or_unknown_names_fail_closed() {
        assert_eq!(map_tensor_name("layers.01.attn_norm.weight"), None);
        assert_eq!(map_tensor_name("layers.0.ffn.experts.-1.w1.weight"), None);
        assert_eq!(map_tensor_name("layers.0.made_up.weight"), None);
    }

    #[test]
    fn covers_every_emitted_official_base_index_name_family() {
        // Normalized from the 72,317-name official 0731 index. `.scale`
        // sidecars are deliberately absent: HfModelSource consumes them
        // inline, and `mtp.*` belongs to the separate draft artifact.
        let names = [
            "embed.weight",
            "hc_head_base",
            "hc_head_fn",
            "hc_head_scale",
            "head.weight",
            "norm.weight",
            "layers.0.attn.attn_sink",
            "layers.0.attn.compressor.ape",
            "layers.0.attn.compressor.norm.weight",
            "layers.0.attn.compressor.wgate.weight",
            "layers.0.attn.compressor.wkv.weight",
            "layers.0.attn.indexer.compressor.ape",
            "layers.0.attn.indexer.compressor.norm.weight",
            "layers.0.attn.indexer.compressor.wgate.weight",
            "layers.0.attn.indexer.compressor.wkv.weight",
            "layers.0.attn.indexer.weights_proj.weight",
            "layers.0.attn.indexer.wq_b.weight",
            "layers.0.attn.kv_norm.weight",
            "layers.0.attn.q_norm.weight",
            "layers.0.attn.wkv.weight",
            "layers.0.attn.wo_a.weight",
            "layers.0.attn.wo_b.weight",
            "layers.0.attn.wq_a.weight",
            "layers.0.attn.wq_b.weight",
            "layers.0.attn_norm.weight",
            "layers.0.ffn.experts.255.w1.weight",
            "layers.0.ffn.experts.255.w2.weight",
            "layers.0.ffn.experts.255.w3.weight",
            "layers.0.ffn.gate.bias",
            "layers.0.ffn.gate.tid2eid",
            "layers.0.ffn.gate.weight",
            "layers.0.ffn.shared_experts.w1.weight",
            "layers.0.ffn.shared_experts.w2.weight",
            "layers.0.ffn.shared_experts.w3.weight",
            "layers.0.ffn_norm.weight",
            "layers.0.hc_attn_base",
            "layers.0.hc_attn_fn",
            "layers.0.hc_attn_scale",
            "layers.0.hc_ffn_base",
            "layers.0.hc_ffn_fn",
            "layers.0.hc_ffn_scale",
        ];

        for name in names {
            assert!(
                map_tensor_name(name).is_some(),
                "unmapped official family: {name}"
            );
        }
    }
}
