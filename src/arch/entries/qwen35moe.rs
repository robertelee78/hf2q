//! `qwen35moe` (Qwen3.5/3.6 MoE) registry entry.
//!
//! Tensor catalog hand-transcribed from `/opt/llama.cpp/src/models/qwen35moe.cpp`
//! and `/opt/llama.cpp/src/llama-arch.cpp` (LLM_TENSOR_* table).  Per the
//! sovereignty rule, the catalog is derived by reading llama.cpp source, not
//! by parsing a GGUF produced externally.

use crate::arch::catalog::{TensorCatalog, TensorSpec};
use crate::arch::registry::{ArchEntry, EvalCorpus, QualityThresholds};

// ---------------------------------------------------------------------------
// Tensor catalog
// ---------------------------------------------------------------------------

const GLOBAL: &[TensorSpec] = &[
    TensorSpec::global("token_embd.weight"),
    TensorSpec::global("output_norm.weight"),
    TensorSpec::global("output.weight"),
];

const FULL_ATTENTION_LAYER: &[TensorSpec] = &[
    TensorSpec::per_layer("blk.{L}.attn_norm.weight"),
    TensorSpec::per_layer("blk.{L}.post_attention_norm.weight"),
    TensorSpec::per_layer("blk.{L}.attn_q.weight"),
    TensorSpec::per_layer("blk.{L}.attn_k.weight"),
    TensorSpec::per_layer("blk.{L}.attn_v.weight"),
    TensorSpec::per_layer("blk.{L}.attn_output.weight"),
    TensorSpec::per_layer("blk.{L}.attn_q_norm.weight"),
    TensorSpec::per_layer("blk.{L}.attn_k_norm.weight"),
    // attn_output_gate=true (Decision 7).
    TensorSpec::per_layer("blk.{L}.attn_gate.weight"),
    // MoE FFN — router + per-expert merged + shared experts (Decision 9).
    TensorSpec::per_layer("blk.{L}.ffn_gate_inp.weight"),
    TensorSpec::per_layer("blk.{L}.ffn_gate_exps.weight"),
    TensorSpec::per_layer("blk.{L}.ffn_up_exps.weight"),
    TensorSpec::per_layer("blk.{L}.ffn_down_exps.weight"),
    TensorSpec::per_layer("blk.{L}.ffn_gate_shexp.weight"),
    TensorSpec::per_layer("blk.{L}.ffn_up_shexp.weight"),
    TensorSpec::per_layer("blk.{L}.ffn_down_shexp.weight"),
];

const LINEAR_ATTENTION_LAYER: &[TensorSpec] = &[
    TensorSpec::per_layer("blk.{L}.attn_norm.weight"),
    TensorSpec::per_layer("blk.{L}.post_attention_norm.weight"),
    // Linear-attention QKV/Z fusion.
    TensorSpec::per_layer("blk.{L}.attn_qkv.weight"),
    TensorSpec::per_layer("blk.{L}.attn_gate.weight"),
    // SSM state.
    TensorSpec::per_layer("blk.{L}.ssm_a"),
    TensorSpec::per_layer("blk.{L}.ssm_dt.bias"),
    TensorSpec::per_layer("blk.{L}.ssm_dt.weight"),
    TensorSpec::per_layer("blk.{L}.ssm_out.weight"),
    TensorSpec::per_layer("blk.{L}.ssm_conv1d.weight"),
    TensorSpec::per_layer("blk.{L}.ssm_norm.weight"),
    // MoE FFN, same shape as full-attn layers.
    TensorSpec::per_layer("blk.{L}.ffn_gate_inp.weight"),
    TensorSpec::per_layer("blk.{L}.ffn_gate_exps.weight"),
    TensorSpec::per_layer("blk.{L}.ffn_up_exps.weight"),
    TensorSpec::per_layer("blk.{L}.ffn_down_exps.weight"),
    TensorSpec::per_layer("blk.{L}.ffn_gate_shexp.weight"),
    TensorSpec::per_layer("blk.{L}.ffn_up_shexp.weight"),
    TensorSpec::per_layer("blk.{L}.ffn_down_shexp.weight"),
];

const MTP: &[TensorSpec] = &[
    TensorSpec::optional_per_layer("blk.{N}.nextn.eh_proj.weight"),
    TensorSpec::optional_per_layer("blk.{N}.nextn.embed_tokens.weight"),
    TensorSpec::optional_per_layer("blk.{N}.nextn.enorm.weight"),
    TensorSpec::optional_per_layer("blk.{N}.nextn.hnorm.weight"),
];

const CATALOG: TensorCatalog = TensorCatalog {
    global: GLOBAL,
    full_attention_layer: FULL_ATTENTION_LAYER,
    linear_attention_layer: LINEAR_ATTENTION_LAYER,
    mtp: MTP,
};

const SMOKE_PROMPTS: &[&str] = &[
    "The quick brown fox",
    "Once upon a time",
];

const HF_REPOS: &[&str] = &[
    // Robert's primary MoE deliverable (ADR-012 §Business problem).
    "jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated",
];

pub static ENTRY: &ArchEntry = &ArchEntry {
    arch: "qwen35moe",
    hf_architectures: &["Qwen3_5MoeForCausalLM"],
    tensor_catalog: &CATALOG,
    has_mtp: true,
    // The MoE target dropped vision_config — only the dense 27B carries it.
    has_vision: false,
    // 35B MoE convert peak (per ADR-012 P7): ~70 GB weights + ~73 GB DWQ
    // intermediate + ~10 GB margin = 150 GB floor.
    disk_floor_gb: 150,
    smoke_prompts: SMOKE_PROMPTS,
    hf_repos: HF_REPOS,
    ppl_corpus: EvalCorpus::WIKITEXT2_512,
    quality_thresholds: QualityThresholds::ADR_012_DEFAULTS,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_arch_string() {
        assert_eq!(ENTRY.arch, "qwen35moe");
    }

    #[test]
    fn entry_hf_architectures_present() {
        assert!(ENTRY.hf_architectures.contains(&"Qwen3_5MoeForCausalLM"));
    }

    #[test]
    fn catalog_full_attn_includes_router_and_experts() {
        let names: Vec<&str> = CATALOG
            .full_attention_layer
            .iter()
            .map(|t| t.name)
            .collect();
        for required in &[
            "blk.{L}.ffn_gate_inp.weight",
            "blk.{L}.ffn_gate_exps.weight",
            "blk.{L}.ffn_up_exps.weight",
            "blk.{L}.ffn_down_exps.weight",
        ] {
            assert!(names.contains(required), "missing {required}");
        }
    }

    #[test]
    fn catalog_full_attn_includes_shared_experts() {
        // ADR-012 Decision 9: shared experts emit as `_shexp` singletons,
        // not merged.  Missing them = silent loss of always-on capacity.
        let names: Vec<&str> = CATALOG
            .full_attention_layer
            .iter()
            .map(|t| t.name)
            .collect();
        for required in &[
            "blk.{L}.ffn_gate_shexp.weight",
            "blk.{L}.ffn_up_shexp.weight",
            "blk.{L}.ffn_down_shexp.weight",
        ] {
            assert!(names.contains(required), "missing {required}");
        }
    }

    #[test]
    fn linear_attn_includes_ssm_state() {
        let names: Vec<&str> = CATALOG
            .linear_attention_layer
            .iter()
            .map(|t| t.name)
            .collect();
        for required in &["blk.{L}.ssm_a", "blk.{L}.ssm_conv1d.weight"] {
            assert!(names.contains(required), "missing {required}");
        }
    }

    #[test]
    fn disk_floor_above_dense() {
        // Sanity: MoE floor must exceed dense floor (35B > 27B + experts).
        assert!(ENTRY.disk_floor_gb > super::super::qwen35::ENTRY.disk_floor_gb);
    }
}
