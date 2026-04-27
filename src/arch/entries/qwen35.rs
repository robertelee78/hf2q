//! qwen35 (dense Qwen3.5 / Qwen3.6-27B) registry entry.
//!
//! Tensor catalog hand-transcribed from `/opt/llama.cpp/src/llama-arch.cpp`
//! and the P4-shipped mapping in `src/models/qwen35/dense.rs`.

use crate::arch::catalog::{LayerScope, TensorCatalog, TensorCatalogEntry, TensorDtype};
use crate::arch::registry::{ArchEntry, EvalCorpus, QualityThresholds};

/// Tensor templates emitted by the dense qwen35 convert path.
///
/// Every entry cites either llama-arch.cpp:{line} or the P4-shipped
/// mapper line that emits the tensor. Per-layer tensors use `{L}`.
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
            citation: "src/models/qwen35/dense.rs:50 (model.embed_tokens.weight → token_embd.weight)",
        },
        TensorCatalogEntry {
            name_template: "output_norm.weight",
            scope: LayerScope::Global,
            dtype: TensorDtype::F32,
            citation: "src/models/qwen35/dense.rs:54 (model.norm.weight → output_norm.weight)",
        },
        TensorCatalogEntry {
            name_template: "output.weight",
            scope: LayerScope::Global,
            dtype: TensorDtype::F16,
            citation: "src/models/qwen35/dense.rs:58 (lm_head.weight → output.weight)",
        },
        // Per-block norms (present on every block).
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_norm.weight",
            scope: LayerScope::AllLayers,
            dtype: TensorDtype::F32,
            citation: "llama-arch.cpp:LLM_TENSOR_ATTN_NORM; src/models/qwen35/dense.rs:198",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.post_attention_norm.weight",
            scope: LayerScope::AllLayers,
            dtype: TensorDtype::F32,
            citation: "llama-arch.cpp:367 LLM_TENSOR_ATTN_POST_NORM; src/models/qwen35/dense.rs:201",
        },
        // Full-attention block tensors.
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_q.weight",
            scope: LayerScope::FullAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "src/models/qwen35/dense.rs:98 (self_attn.q_proj → attn_q)",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_k.weight",
            scope: LayerScope::FullAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "src/models/qwen35/dense.rs:99",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_v.weight",
            scope: LayerScope::FullAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "src/models/qwen35/dense.rs:100",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_output.weight",
            scope: LayerScope::FullAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "src/models/qwen35/dense.rs:101",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_q_norm.weight",
            scope: LayerScope::FullAttentionLayersOnly,
            dtype: TensorDtype::F32,
            citation: "src/models/qwen35/dense.rs:102",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_k_norm.weight",
            scope: LayerScope::FullAttentionLayersOnly,
            dtype: TensorDtype::F32,
            citation: "src/models/qwen35/dense.rs:103",
        },
        // Linear-attention block tensors.
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_qkv.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "llama-arch.cpp:382 LLM_TENSOR_ATTN_QKV; src/models/qwen35/dense.rs:136",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.attn_gate.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "llama-arch.cpp:370 LLM_TENSOR_ATTN_GATE; src/models/qwen35/dense.rs:139",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_alpha.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "llama-arch.cpp:400 LLM_TENSOR_SSM_ALPHA; src/models/qwen35/dense.rs:142",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_beta.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "llama-arch.cpp:416 LLM_TENSOR_SSM_BETA; src/models/qwen35/dense.rs:145",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_out.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::Quantized,
            citation: "llama-arch.cpp:402 LLM_TENSOR_SSM_OUT; src/models/qwen35/dense.rs:148",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_a",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::F32,
            citation: "llama-arch.cpp:395 LLM_TENSOR_SSM_A_NOSCAN; src/models/qwen35/dense.rs:151",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_dt.bias",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::F32,
            citation: "llama-arch.cpp:397 + convert_hf_to_gguf.py:4791; src/models/qwen35/dense.rs:156",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_conv1d.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::F32,
            citation: "llama-arch.cpp:396 LLM_TENSOR_SSM_CONV1D; src/models/qwen35/dense.rs:159",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ssm_norm.weight",
            scope: LayerScope::LinearAttentionLayersOnly,
            dtype: TensorDtype::F32,
            citation: "llama-arch.cpp:401 LLM_TENSOR_SSM_NORM; src/models/qwen35/dense.rs:163",
        },
        // Dense FFN.
        TensorCatalogEntry {
            name_template: "blk.{L}.ffn_gate.weight",
            scope: LayerScope::AllLayers,
            dtype: TensorDtype::Quantized,
            citation: "src/models/qwen35/dense.rs:174 (mlp.gate_proj → ffn_gate)",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ffn_up.weight",
            scope: LayerScope::AllLayers,
            dtype: TensorDtype::Quantized,
            citation: "src/models/qwen35/dense.rs:175",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.ffn_down.weight",
            scope: LayerScope::AllLayers,
            dtype: TensorDtype::Quantized,
            citation: "src/models/qwen35/dense.rs:176",
        },
        // MTP tensors (emitted when mtp_num_hidden_layers > 0).
        TensorCatalogEntry {
            name_template: "blk.{L}.nextn.enorm.weight",
            scope: LayerScope::MtpLayers,
            dtype: TensorDtype::F32,
            citation: "llama-arch.cpp:449 LLM_TENSOR_NEXTN_ENORM",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.nextn.hnorm.weight",
            scope: LayerScope::MtpLayers,
            dtype: TensorDtype::F32,
            citation: "llama-arch.cpp:450 LLM_TENSOR_NEXTN_HNORM",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.nextn.embed_tokens.weight",
            scope: LayerScope::MtpLayers,
            dtype: TensorDtype::F16,
            citation: "llama-arch.cpp:448 LLM_TENSOR_NEXTN_EMBED_TOKENS",
        },
        TensorCatalogEntry {
            name_template: "blk.{L}.nextn.eh_proj.weight",
            scope: LayerScope::MtpLayers,
            dtype: TensorDtype::Quantized,
            citation: "llama-arch.cpp:447 LLM_TENSOR_NEXTN_EH_PROJ",
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
    has_vision: true, // Qwen3.6-27B ships a vision_config; --emit-vision-tower honored when present.
    smoke_prompts: &["The quick brown fox"],
    ppl_corpus: EvalCorpus {
        id: "wikitext2",
        token_count: 512,
        sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
    },
    quality_thresholds: QualityThresholds::ADR_012_DEFAULT,
    disk_floor_gb: 100, // src/input/hf_download.rs Decision 14 floor
    hf_repos: &["Qwen/Qwen3.6-27B"],
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
        // 3 global + 2 per-layer norms + 6 full-attn + 9 linear-attn + 3 dense-ffn + 4 MTP = 27
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
        //            + dense_ffn(3*64=192) + mtp(4*1=4) = 855
        assert_eq!(count, 3 + 128 + 96 + 432 + 192 + 4);
        assert_eq!(count, 855);
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
        assert_eq!(
            ENTRY.quality_thresholds,
            QualityThresholds::ADR_012_DEFAULT
        );
    }

    #[test]
    fn has_mtp_and_has_vision_are_true_for_qwen35() {
        // Qwen3.5 dense has MTP (mtp_num_hidden_layers: 1) and vision_config
        // (Qwen3.6-27B ships a ViT that needs --emit-vision-tower).
        assert!(ENTRY.has_mtp);
        assert!(ENTRY.has_vision);
    }
}
