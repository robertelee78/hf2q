//! qwen35 (dense Qwen3.5-family, including Qwen3.6 and Qwen3.8) registry entry.
//!
//! The active mapping is `src/convert/arch/qwen35_dense.rs`; the native
//! consumer is `src/inference/models/qwen35`.

use crate::arch::catalog::{LayerScope, TensorCatalog, TensorCatalogEntry, TensorDtype};
use crate::arch::registry::{ArchEntry, EvalCorpus, QualityThresholds};

/// Tensor templates emitted by the dense qwen35 convert path.
///
/// Every entry cites the active hf2q mapper or native loader that owns the
/// tensor contract. Per-layer tensors use `{L}`.
///
/// NOTE: linear-attn tensors are emitted on linear-attention layers
/// only; full-attn tensors on full-attention layers only. The
/// expansion helper uses `LayerScope::{LinearAttention,FullAttention}LayersOnly`
/// so the expected count folds `full_attention_interval` correctly.
const DENSE_CATALOG: TensorCatalog = TensorCatalog {
    entries: &[
        // Global.
        TensorCatalogEntry {
            name_template: "token_embd.weight",
            scope: LayerScope::Global,
            dtype: TensorDtype::F16,
            citation:
                "src/convert/arch/qwen35_dense.rs (model.embed_tokens.weight → token_embd.weight)",
        },
        TensorCatalogEntry {
            name_template: "output_norm.weight",
            scope: LayerScope::Global,
            dtype: TensorDtype::F32,
            citation: "src/convert/arch/qwen35_dense.rs (model.norm.weight → output_norm.weight)",
        },
        TensorCatalogEntry {
            name_template: "output.weight",
            scope: LayerScope::Global,
            dtype: TensorDtype::F16,
            citation: "src/convert/arch/qwen35_dense.rs (lm_head.weight → output.weight)",
        },
        // Per-block norms (present on every block).
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_norm.weight",
            scope: LayerScope::AllLayersIncludingMtp,
            dtype: TensorDtype::F32,
            citation: "src/convert/arch/qwen35_dense.rs (input_layernorm → attn_norm)",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.post_attention_norm.weight",
            scope: LayerScope::AllLayersIncludingMtp,
            dtype: TensorDtype::F32,
            citation: "src/convert/arch/qwen35_dense.rs (post_attention_layernorm mapping)",
        },
        // Full-attention block tensors.
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_q.weight",
            scope: LayerScope::FullAttentionAndMtpLayers,
            dtype: TensorDtype::Quantized,
            citation: "src/convert/arch/qwen35_dense.rs (self_attn.q_proj → attn_q)",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_k.weight",
            scope: LayerScope::FullAttentionAndMtpLayers,
            dtype: TensorDtype::Quantized,
            citation: "src/convert/arch/qwen35_dense.rs",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_v.weight",
            scope: LayerScope::FullAttentionAndMtpLayers,
            dtype: TensorDtype::Quantized,
            citation: "src/convert/arch/qwen35_dense.rs",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_output.weight",
            scope: LayerScope::FullAttentionAndMtpLayers,
            dtype: TensorDtype::Quantized,
            citation: "src/convert/arch/qwen35_dense.rs",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_q_norm.weight",
            scope: LayerScope::FullAttentionAndMtpLayers,
            dtype: TensorDtype::F32,
            citation: "src/convert/arch/qwen35_dense.rs",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_k_norm.weight",
            scope: LayerScope::FullAttentionAndMtpLayers,
            dtype: TensorDtype::F32,
            citation: "src/convert/arch/qwen35_dense.rs",
        },
        // Linear-attention block tensors.
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_qkv.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "src/convert/arch/qwen35_dense.rs (linear in_proj_qkv)",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_gate.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "src/convert/arch/qwen35_dense.rs (linear in_proj_z)",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_alpha.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "src/convert/arch/qwen35_dense.rs (linear in_proj_a)",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_beta.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "src/convert/arch/qwen35_dense.rs (linear in_proj_b)",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_out.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "src/convert/arch/qwen35_dense.rs (linear out_proj)",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_a",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::F32,
            citation: "src/convert/arch/qwen35_dense.rs (linear A_log)",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_dt.bias",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::F32,
            citation: "src/convert/arch/qwen35_dense.rs (linear dt_bias)",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_conv1d.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::F32,
            citation: "src/convert/arch/qwen35_dense.rs (linear conv1d)",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_norm.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::F32,
            citation: "src/convert/arch/qwen35_dense.rs (linear norm)",
        },
        // Dense FFN.
        TensorCatalogEntry {
            name_template: "blk.{L}.ffn_gate.weight",
            scope: LayerScope::AllLayersIncludingMtp,
            dtype: TensorDtype::Quantized,
            citation: "src/convert/arch/qwen35_dense.rs (mlp.gate_proj → ffn_gate)",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ffn_up.weight",
            scope: LayerScope::AllLayersIncludingMtp,
            dtype: TensorDtype::Quantized,
            citation: "src/convert/arch/qwen35_dense.rs",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ffn_down.weight",
            scope: LayerScope::AllLayersIncludingMtp,
            dtype: TensorDtype::Quantized,
            citation: "src/convert/arch/qwen35_dense.rs",
        },
        // MTP tensors (emitted when mtp_num_hidden_layers > 0).
        TensorCatalogEntry {
            name_template: "blk.{L}.nextn.enorm.weight",
            scope: LayerScope::MtpLayers,
            dtype: TensorDtype::F32,
            citation: "src/convert/arch/qwen35_dense.rs (MTP embedding norm)",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.nextn.hnorm.weight",
            scope: LayerScope::MtpLayers,
            dtype: TensorDtype::F32,
            citation: "src/convert/arch/qwen35_dense.rs (MTP hidden norm)",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.nextn.shared_head_norm.weight",
            scope: LayerScope::MtpLayers,
            dtype: TensorDtype::F32,
            citation: "src/convert/arch/qwen35_dense.rs (mtp.norm → nextn.shared_head_norm)",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.nextn.eh_proj.weight",
            scope: LayerScope::MtpLayers,
            dtype: TensorDtype::Quantized,
            citation: "src/convert/arch/qwen35_dense.rs (MTP embedding-hidden projection)",
        },
    ],
};

/// The qwen35 arch entry.
pub const ENTRY: ArchEntry = ArchEntry {
    arch: "qwen35",
    // Both HF architecture aliases — *ForCausalLM is the text-only form,
    // *ForConditionalGeneration ships on multimodal Qwen3.5 checkpoints.
    // `arch_gguf_name` (src/backends/gguf.rs:2585) accepts both; the
    // registry must too so `get_by_hf_architecture` matches what convert
    // sees in config.json.
    hf_architectures: &["Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"],
    tensor_catalog: &DENSE_CATALOG,
    has_mtp: true,
    // ConditionalGeneration checkpoints expose a separate Qwen vision tower.
    // hf2q converts that tower to an mmproj and serves its embeddings through
    // the same bounded SlotAware path as text prefill. CausalLM checkpoints
    // remain usable without a projector; this flag advertises family
    // capability, not a requirement that every checkpoint include vision.
    has_vision: true,
    smoke_prompts: &["The quick brown fox"],
    ppl_corpus: EvalCorpus {
        id: "wikitext2",
        token_count: 512,
        sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
    },
    quality_thresholds: QualityThresholds::ADR_012_DEFAULT,
    disk_floor_gb: 100, // src/input/hf_download.rs Decision 14 floor
    hf_repos: &["Qwen/Qwen3.6-27B", "Qwen/Qwen3.8-27B"],
    // ADR-014 P8 Decision 18: no per-arch override yet — fall through
    // to the Decision-18 routing table (dense ≤30B → imatrix-q4_k_m).
    auto_override: None,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::catalog::CatalogExpansion;

    #[test]
    fn dense_catalog_has_expected_entry_count() {
        // The 11 MTP inner-block templates are shared with verifier layers;
        // four NextN-specific templates bring the declarative total to 27.
        assert_eq!(DENSE_CATALOG.entries.len(), 27);
    }

    #[test]
    fn qwen36_27b_tensor_count_folds_correctly() {
        // Qwen3.6-27B: 64 layers, full_attention_interval 4 → 16 full, 48 linear.
        let exp = CatalogExpansion {
            num_hidden_layers: 64,
            num_full_attention_layers: 16,
            num_linear_attention_layers: 48,
            num_experts: 0,
            has_shared_expert: false,
            mtp_num_hidden_layers: 1,
        };
        let count = ENTRY.expected_tensor_count(exp);
        // globals(3) + per_block_norms(2*64=128) + full_attn(6*16=96) + linear_attn(9*48=432)
        //            + dense_ffn(3*64=192) + mtp(15*1=15) = 866
        assert_eq!(count, 3 + 128 + 96 + 432 + 192 + 15);
        assert_eq!(count, 866);
    }

    #[test]
    fn hf_architectures_routes_to_dense_entry() {
        assert_eq!(
            ENTRY.hf_architectures,
            &["Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"]
        );
        assert_eq!(ENTRY.arch, "qwen35");
    }

    #[test]
    fn quality_thresholds_are_adr012_defaults() {
        assert_eq!(ENTRY.quality_thresholds, QualityThresholds::ADR_012_DEFAULT);
    }

    #[test]
    fn has_mtp_and_vision_capabilities_are_advertised() {
        // Dense conditional-generation checkpoints carry both one MTP layer
        // and a separately converted vision tower.
        assert!(ENTRY.has_mtp);
        assert!(ENTRY.has_vision);
    }
}
