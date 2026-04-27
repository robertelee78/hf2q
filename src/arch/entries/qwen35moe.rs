//! qwen35moe (Qwen3.5-MoE / Qwen3.6-35B-A3B) registry entry.
//!
//! Tensor catalog hand-transcribed from `/opt/llama.cpp/src/llama-arch.cpp`
//! and the P4-shipped mapping in `src/models/qwen35/moe.rs`.

use crate::arch::catalog::{LayerScope, TensorCatalog, TensorCatalogEntry, TensorDtype};
use crate::arch::registry::{ArchEntry, EvalCorpus, QualityThresholds};

/// MoE catalog. Shares linear-attn and full-attn tensors with dense,
/// diverges at the FFN: router + merged experts + shared expert.
const MOE_CATALOG: TensorCatalog = TensorCatalog {
    entries: &[
        // Global.
        TensorCatalogEntry {
            name_template: "token_embd.weight",
            scope: LayerScope::Global,
            dtype: TensorDtype::F16,
            citation: "src/models/qwen35/moe.rs:49",
        },
        TensorCatalogEntry {
            name_template: "output_norm.weight",
            scope: LayerScope::Global,
            dtype: TensorDtype::F32,
            citation: "src/models/qwen35/moe.rs:53",
        },
        TensorCatalogEntry {
            name_template: "output.weight",
            scope: LayerScope::Global,
            dtype: TensorDtype::F16,
            citation: "src/models/qwen35/moe.rs:57",
        },
        // Per-block norms.
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_norm.weight",
            scope: LayerScope::AllLayers,
            dtype: TensorDtype::F32,
            citation: "src/models/qwen35/moe.rs:218",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.post_attention_norm.weight",
            scope: LayerScope::AllLayers,
            dtype: TensorDtype::F32,
            citation: "llama-arch.cpp:367; src/models/qwen35/moe.rs:221",
        },
        // Full-attention.
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_q.weight",
            scope: LayerScope::FullAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "src/models/qwen35/moe.rs:149",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_k.weight",
            scope: LayerScope::FullAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "src/models/qwen35/moe.rs:150",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_v.weight",
            scope: LayerScope::FullAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "src/models/qwen35/moe.rs:151",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_output.weight",
            scope: LayerScope::FullAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "src/models/qwen35/moe.rs:152",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_q_norm.weight",
            scope: LayerScope::FullAttentionLayersOnly,
            dtype: TensorDtype::F32,
            citation: "src/models/qwen35/moe.rs:153",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_k_norm.weight",
            scope: LayerScope::FullAttentionLayersOnly,
            dtype: TensorDtype::F32,
            citation: "src/models/qwen35/moe.rs:154",
        },
        // Linear-attention (shared with dense).
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_qkv.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "llama-arch.cpp:382; src/models/qwen35/moe.rs:179",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_gate.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "llama-arch.cpp:370; src/models/qwen35/moe.rs:181",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_alpha.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "llama-arch.cpp:400; src/models/qwen35/moe.rs:183",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_beta.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "llama-arch.cpp:416; src/models/qwen35/moe.rs:185",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_out.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "llama-arch.cpp:402; src/models/qwen35/moe.rs:187",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_a",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::F32,
            citation: "llama-arch.cpp:395; src/models/qwen35/moe.rs:189",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_dt.bias",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::F32,
            citation: "llama-arch.cpp:397 + convert_hf_to_gguf.py:4791",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_conv1d.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::F32,
            citation: "llama-arch.cpp:396",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_norm.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::F32,
            citation: "llama-arch.cpp:401",
        },
        // MoE router (one per MoE block; every block has the router in qwen35moe).
        TensorCatalogEntry {
            name_template: "blk.{L}.ffn_gate_inp.weight",
            scope: LayerScope::MoeRouterPerLayer,
            dtype: TensorDtype::F32,
            citation: "llama-arch.cpp:393 LLM_TENSOR_FFN_GATE_INP; src/models/qwen35/moe.rs:83",
        },
        // Merged per-expert tensors (N experts merged into one 3-D stack per projection).
        TensorCatalogEntry {
            name_template: "blk.{L}.ffn_gate_exps.weight",
            scope: LayerScope::AllLayers, // Once per block after merge — not once per expert.
            dtype: TensorDtype::Quantized,
            citation: "src/models/qwen35/moe.rs:126 (merged stack of 256 experts)",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ffn_up_exps.weight",
            scope: LayerScope::AllLayers,
            dtype: TensorDtype::Quantized,
            citation: "src/models/qwen35/moe.rs:127",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ffn_down_exps.weight",
            scope: LayerScope::AllLayers,
            dtype: TensorDtype::Quantized,
            citation: "src/models/qwen35/moe.rs:128",
        },
        // Shared-expert tensors (per-block, one copy).
        TensorCatalogEntry {
            name_template: "blk.{L}.ffn_gate_shexp.weight",
            scope: LayerScope::MoeSharedExpertPerLayer,
            dtype: TensorDtype::Quantized,
            citation: "src/models/qwen35/moe.rs:138",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ffn_up_shexp.weight",
            scope: LayerScope::MoeSharedExpertPerLayer,
            dtype: TensorDtype::Quantized,
            citation: "src/models/qwen35/moe.rs:139",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ffn_down_shexp.weight",
            scope: LayerScope::MoeSharedExpertPerLayer,
            dtype: TensorDtype::Quantized,
            citation: "src/models/qwen35/moe.rs:140",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ffn_gate_inp_shexp.weight",
            scope: LayerScope::MoeSharedExpertPerLayer,
            dtype: TensorDtype::F32,
            citation: "llama-arch.cpp:394; src/models/qwen35/moe.rs:90",
        },
        // MTP tensors.
        TensorCatalogEntry {
            name_template: "blk.{L}.nextn.enorm.weight",
            scope: LayerScope::MtpLayers,
            dtype: TensorDtype::F32,
            citation: "llama-arch.cpp:449",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.nextn.hnorm.weight",
            scope: LayerScope::MtpLayers,
            dtype: TensorDtype::F32,
            citation: "llama-arch.cpp:450",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.nextn.embed_tokens.weight",
            scope: LayerScope::MtpLayers,
            dtype: TensorDtype::F16,
            citation: "llama-arch.cpp:448",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.nextn.eh_proj.weight",
            scope: LayerScope::MtpLayers,
            dtype: TensorDtype::Quantized,
            citation: "llama-arch.cpp:447",
        },
    ],
};

/// The qwen35moe arch entry.
pub const ENTRY: ArchEntry = ArchEntry {
    arch: "qwen35moe",
    // Both HF architecture aliases — see src/arch/entries/qwen35.rs for
    // rationale. Registry must list both so get_by_hf_architecture
    // matches arch_gguf_name's four-alias acceptance.
    hf_architectures: &["Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"],
    tensor_catalog: &MOE_CATALOG,
    has_mtp: true,
    // The Robert-named 35B-A3B MoE target dropped vision_config from config.json
    // (ADR-012 Decision 18 note); --emit-vision-tower silently skips for this repo.
    // has_vision==true advertises the *possibility*; the converter fast-paths to skip
    // when config.json has no vision_config, so no Gemma-style regression is incurred.
    has_vision: false,
    smoke_prompts: &["The quick brown fox"],
    ppl_corpus: EvalCorpus {
        id: "wikitext2",
        token_count: 512,
        sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
    },
    quality_thresholds: QualityThresholds::ADR_012_DEFAULT,
    disk_floor_gb: 150, // src/input/hf_download.rs Decision 14 floor for MoE
    hf_repos: &["jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated"],
    // ADR-014 P8 Decision 18: no per-arch override yet — fall through
    // to the Decision-18 routing table (MoE → dwq-4-{6,8} based on RAM).
    auto_override: None,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::catalog::CatalogExpansion;

    #[test]
    fn moe_catalog_has_expected_entry_count() {
        // 3 global + 2 per-layer norms + 6 full-attn + 9 linear-attn
        // + 1 router + 3 merged_exps + 4 shared_expert(+inp_shexp) + 4 MTP = 32
        assert_eq!(MOE_CATALOG.entries.len(), 32);
    }

    #[test]
    fn qwen36_35ba3b_tensor_count_folds_correctly() {
        // 40 layers, full_attention_interval 4 → 10 full, 30 linear.
        let exp = CatalogExpansion {
            num_hidden_layers: 40,
            num_full_attention_layers: 10,
            num_linear_attention_layers: 30,
            num_experts: 256,
            has_shared_expert: true,
            mtp_num_hidden_layers: 1,
        };
        let count = ENTRY.expected_tensor_count(exp);
        // globals(3) + per_block_norms(2*40=80) + full_attn(6*10=60) + linear_attn(9*30=270)
        //   + router(1*40=40) + merged_exps(3*40=120) + shexp(4*40=160) + mtp(4*1=4) = 737
        assert_eq!(count, 3 + 80 + 60 + 270 + 40 + 120 + 160 + 4);
        assert_eq!(count, 737);
    }

    #[test]
    fn hf_architectures_routes_to_moe_entry() {
        assert_eq!(
            ENTRY.hf_architectures,
            &["Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"]
        );
        assert_eq!(ENTRY.arch, "qwen35moe");
    }

    #[test]
    fn has_mtp_true_has_vision_false_for_moe_robert_target() {
        assert!(ENTRY.has_mtp);
        // Robert's jenerallee78/Qwen3.6-35B-A3B-... dropped vision_config.
        assert!(!ENTRY.has_vision);
    }
}
