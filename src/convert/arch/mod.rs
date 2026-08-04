//! Per-architecture HF→GGUF tensor-name mappers + metadata builders.
//!
//! ADR-033 §P0 "Per-arch convert-side mapping at `src/convert/arch/<arch>.rs`".
//! v1 ships only `llama3` (dense decoder test fixture for the convert
//! matrix); extending to `{gemma4, qwen35moe, qwen3vl, gemma4_mmproj,
//! bert, nomic_bert, minimax_m2}` is iter-23+ work per the ADR.
//!
//! Per [[feedback-no-backwards-compat-2026-05-18]]: no compat shims, no
//! per-arch fallback. Adding a new arch is an explicit code change that
//! adds a new file under this module — `ArchName` is closed enum.
//!
//! Per [[feedback-no-loop-suppression-2026-05-17]]: `map_tensor_name`
//! returning `None` is a signal the caller MUST surface (e.g. as
//! `ConvertError::UnmappedTensor { hf_name }`). Never silently skip.

pub mod bake;
pub mod bert;
pub mod deepseek4;
pub mod deepseek4_metadata;
pub mod gemma4;
pub mod gemma4_mmproj;
pub mod gemma4_vision_mmproj;
pub mod llama3;
pub mod minimax_m2;
pub mod nomic_bert;
pub mod qwen35moe;
pub mod qwen35moe_full;
pub mod qwen3vl_text;

/// Per-arch "source-tensor drop list" lookup.
///
/// HF safetensors checkpoints often carry non-weight buffers (e.g.
/// BERT's `embeddings.position_ids` is an I64 `[0..max_pos]` lookup
/// table — not a learnable weight; canonical's `bert.py:filter_tensors`
/// drops it explicitly at lines 82-92). hf2q's source reader normally
/// errors on unsupported dtypes (I64 / I32 / U8) because they'd
/// otherwise silently slip through. The drop list lets the reader
/// SKIP known non-weight tensors before the dtype check, preserving
/// strictness on truly unexpected tensors.
///
/// Mirrors canonical's per-class `filter_tensors` overrides
/// (`/opt/llama.cpp/conversion/bert.py:82-92` for BERT; nomic-bert
/// adds `mlp.experts.bias` per `bert.py:362-365`). Other arches
/// (Gemma 4, Qwen 3.5, Llama 3, MiniMax-M2) have empty drop lists
/// — their canonical converters don't override `filter_tensors`.
///
/// Called by `source_reader::HfModelSource::open` via `model_type`
/// from `config.json`. Returns `true` if the (stripped) tensor name
/// should be silently filtered out of the source meta list.
pub fn should_drop_source_tensor(model_type: &str, hf_name: &str) -> bool {
    // Canonical `bert.py:71-72` strips a leading `bert.` prefix
    // before applying the filter rules.
    let stripped = hf_name.strip_prefix("bert.").unwrap_or(hf_name);

    match model_type {
        "bert" => {
            // /opt/llama.cpp/conversion/bert.py:82-92 — exact drop list.
            // `position_ids` is I64; pooler is non-embedding-relevant.
            if matches!(
                stripped,
                "embeddings.position_ids" | "pooler.dense.weight" | "pooler.dense.bias"
            ) {
                return true;
            }
            if stripped.starts_with("cls.predictions") || stripped.starts_with("cls.seq_relationship")
            {
                return true;
            }
        }
        "nomic_bert" => {
            // /opt/llama.cpp/conversion/bert.py:362-365 — nomic-bert
            // additionally drops `mlp.experts.bias`. Falls through to
            // the BERT base list since NomicBertModel extends BertModel.
            if stripped.contains("mlp.experts.bias") {
                return true;
            }
            if matches!(
                stripped,
                "embeddings.position_ids" | "pooler.dense.weight" | "pooler.dense.bias"
            ) {
                return true;
            }
            if stripped.starts_with("cls.predictions") || stripped.starts_with("cls.seq_relationship")
            {
                return true;
            }
        }
        "deepseek_v4" => {
            // Base conversion deliberately excludes the separately
            // exported MTP/DSpark namespace. This is an explicit
            // artifact boundary, not an unknown-tensor fallback.
            if hf_name.starts_with("mtp.") {
                return true;
            }
        }
        _ => {}
    }
    false
}

#[cfg(test)]
mod drop_list_tests {
    use super::*;

    #[test]
    fn bert_drops_position_ids() {
        assert!(should_drop_source_tensor("bert", "embeddings.position_ids"));
        assert!(should_drop_source_tensor(
            "bert",
            "bert.embeddings.position_ids"
        ));
    }

    #[test]
    fn bert_drops_pooler() {
        assert!(should_drop_source_tensor("bert", "pooler.dense.weight"));
        assert!(should_drop_source_tensor("bert", "pooler.dense.bias"));
    }

    #[test]
    fn bert_drops_cls_heads() {
        assert!(should_drop_source_tensor(
            "bert",
            "cls.predictions.transform.dense.weight"
        ));
        assert!(should_drop_source_tensor(
            "bert",
            "cls.seq_relationship.weight"
        ));
    }

    #[test]
    fn bert_keeps_real_weights() {
        assert!(!should_drop_source_tensor(
            "bert",
            "embeddings.word_embeddings.weight"
        ));
        assert!(!should_drop_source_tensor(
            "bert",
            "encoder.layer.0.attention.self.query.weight"
        ));
    }

    #[test]
    fn nomic_bert_drops_expert_bias() {
        assert!(should_drop_source_tensor(
            "nomic_bert",
            "encoder.layer.0.mlp.experts.bias"
        ));
    }

    #[test]
    fn non_bert_drops_nothing() {
        assert!(!should_drop_source_tensor(
            "llama",
            "embeddings.position_ids"
        ));
        assert!(!should_drop_source_tensor(
            "gemma4",
            "model.layers.0.self_attn.q_proj.weight"
        ));
        assert!(!should_drop_source_tensor("qwen3_5_moe", "anything"));
    }
}
