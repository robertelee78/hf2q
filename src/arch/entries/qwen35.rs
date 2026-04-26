//! `qwen35` (Qwen3.5/3.6 dense) registry entry.
//!
//! Tensor catalog hand-transcribed from `/opt/llama.cpp/src/models/qwen35.cpp`
//! and `/opt/llama.cpp/src/llama-arch.cpp` (LLM_TENSOR_* table).  Per the
//! sovereignty rule, the catalog is derived by reading llama.cpp source, not
//! by parsing a GGUF produced externally.

use crate::arch::catalog::{TensorCatalog, TensorSpec};
use crate::arch::registry::{ArchEntry, EvalCorpus, QualityThresholds};

// ---------------------------------------------------------------------------
// Tensor catalog
// ---------------------------------------------------------------------------

const GLOBAL: &[TensorSpec] = &[
    // Token embedding + LM head + output norm.  llama-arch.cpp tensor table
    // (LLM_TENSOR_TOKEN_EMBD / LLM_TENSOR_OUTPUT_NORM / LLM_TENSOR_OUTPUT).
    TensorSpec::global("token_embd.weight"),
    TensorSpec::global("output_norm.weight"),
    TensorSpec::global("output.weight"),
];

const FULL_ATTENTION_LAYER: &[TensorSpec] = &[
    // Pre/post norms.  llama-arch.cpp:367 (LLM_TENSOR_ATTN_NORM /
    // LLM_TENSOR_POST_ATTENTION_NORM — Qwen3.5 uses Gemma-style post-norm
    // per ADR-012 P4 verdict).
    TensorSpec::per_layer("blk.{L}.attn_norm.weight"),
    TensorSpec::per_layer("blk.{L}.post_attention_norm.weight"),
    // Q/K/V/O projections (LLM_TENSOR_ATTN_Q/K/V/OUTPUT).
    TensorSpec::per_layer("blk.{L}.attn_q.weight"),
    TensorSpec::per_layer("blk.{L}.attn_k.weight"),
    TensorSpec::per_layer("blk.{L}.attn_v.weight"),
    TensorSpec::per_layer("blk.{L}.attn_output.weight"),
    // Q/K norms (LLM_TENSOR_ATTN_Q_NORM / LLM_TENSOR_ATTN_K_NORM).
    TensorSpec::per_layer("blk.{L}.attn_q_norm.weight"),
    TensorSpec::per_layer("blk.{L}.attn_k_norm.weight"),
    // Output gate (LLM_TENSOR_ATTN_GATE, llama-arch.cpp:370).
    TensorSpec::per_layer("blk.{L}.attn_gate.weight"),
    // Dense FFN (LLM_TENSOR_FFN_GATE/UP/DOWN).
    TensorSpec::per_layer("blk.{L}.ffn_gate.weight"),
    TensorSpec::per_layer("blk.{L}.ffn_up.weight"),
    TensorSpec::per_layer("blk.{L}.ffn_down.weight"),
];

const LINEAR_ATTENTION_LAYER: &[TensorSpec] = &[
    // Pre/post norms (shared with full-attention).
    TensorSpec::per_layer("blk.{L}.attn_norm.weight"),
    TensorSpec::per_layer("blk.{L}.post_attention_norm.weight"),
    // Linear-attention QKV/Z fusion.  llama-arch.cpp:382 (LLM_TENSOR_ATTN_QKV)
    // for the qkv-fused stream; :370 (LLM_TENSOR_ATTN_GATE) for the z gate.
    TensorSpec::per_layer("blk.{L}.attn_qkv.weight"),
    TensorSpec::per_layer("blk.{L}.attn_gate.weight"),
    // SSM / DeltaNet state tensors.  llama-arch.cpp:395-402:
    //   LLM_TENSOR_SSM_A → "blk.%d.ssm_a"
    //   LLM_TENSOR_SSM_DT → "blk.%d.ssm_dt"  (carries the .bias variant)
    //   LLM_TENSOR_SSM_OUT → "blk.%d.ssm_out"
    //   LLM_TENSOR_SSM_CONV1D → "blk.%d.ssm_conv1d"
    //   LLM_TENSOR_SSM_NORM → "blk.%d.ssm_norm"
    TensorSpec::per_layer("blk.{L}.ssm_a"),
    TensorSpec::per_layer("blk.{L}.ssm_dt.bias"),
    TensorSpec::per_layer("blk.{L}.ssm_dt.weight"),
    TensorSpec::per_layer("blk.{L}.ssm_out.weight"),
    TensorSpec::per_layer("blk.{L}.ssm_conv1d.weight"),
    TensorSpec::per_layer("blk.{L}.ssm_norm.weight"),
    // Dense FFN follows linear-attn block (Qwen3.5 dense has FFN every layer).
    TensorSpec::per_layer("blk.{L}.ffn_gate.weight"),
    TensorSpec::per_layer("blk.{L}.ffn_up.weight"),
    TensorSpec::per_layer("blk.{L}.ffn_down.weight"),
];

const MTP: &[TensorSpec] = &[
    // llama-arch.cpp:447-450 (LLM_TENSOR_NEXTN_*).  Optional because emission
    // is gated on `mtp_num_hidden_layers > 0`.
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

// ---------------------------------------------------------------------------
// Smoke prompts
// ---------------------------------------------------------------------------

const SMOKE_PROMPTS: &[&str] = &[
    "The quick brown fox",
    "Once upon a time",
];

// ---------------------------------------------------------------------------
// Canonical HF repos
// ---------------------------------------------------------------------------

const HF_REPOS: &[&str] = &[
    // Robert's primary dense target (ADR-012 §Business problem).
    "Qwen/Qwen3.6-27B",
];

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

pub static ENTRY: &ArchEntry = &ArchEntry {
    arch: "qwen35",
    hf_architectures: &["Qwen3_5ForCausalLM"],
    tensor_catalog: &CATALOG,
    has_mtp: true,
    has_vision: true,
    // 27B dense + DWQ intermediate + output ≈ 55 GB floor per ADR-012 P7.
    disk_floor_gb: 55,
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
        assert_eq!(ENTRY.arch, "qwen35");
    }

    #[test]
    fn entry_hf_architectures_present() {
        assert!(ENTRY.hf_architectures.contains(&"Qwen3_5ForCausalLM"));
    }

    #[test]
    fn catalog_includes_ssm_state_tensors() {
        // ADR-012 P3 verdict: SSM state tensors are mandatory on
        // linear-attention layers; missing one = silent inference failure.
        let names: Vec<&str> = CATALOG
            .linear_attention_layer
            .iter()
            .map(|t| t.name)
            .collect();
        for required in &[
            "blk.{L}.ssm_a",
            "blk.{L}.ssm_dt.bias",
            "blk.{L}.ssm_conv1d.weight",
            "blk.{L}.ssm_out.weight",
        ] {
            assert!(
                names.contains(required),
                "missing SSM tensor {required} from qwen35 linear-attn catalog"
            );
        }
    }

    #[test]
    fn catalog_full_attn_includes_attn_gate() {
        // ADR-012 P4 correction: attn_output_gate=true makes attn_gate
        // mandatory on full-attention blocks.
        let names: Vec<&str> = CATALOG
            .full_attention_layer
            .iter()
            .map(|t| t.name)
            .collect();
        assert!(names.contains(&"blk.{L}.attn_gate.weight"));
    }

    #[test]
    fn mtp_tensors_marked_optional() {
        for spec in CATALOG.mtp {
            assert!(spec.optional, "MTP {} should be optional", spec.name);
        }
    }
}
