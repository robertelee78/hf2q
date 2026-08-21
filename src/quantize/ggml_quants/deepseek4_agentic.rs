//! DeepSeek-V4 agentic mixed-quant policy.
//!
//! Routed experts account for almost all model bytes, while the much smaller
//! attention, compressor, hyper-connection, and indexer projections determine
//! whether sparse attention keeps selecting the right context. A plain Q2_K
//! body is therefore a poor trade for long, tool-heavy sessions: it saves
//! little by quantizing the discrimination path aggressively and can become
//! trapped in repeated structures.
//!
//! This policy is an overlay on peer-compatible `MostlyQ2_K`:
//!
//! - expert gate/up projections remain Q2_K;
//! - expert down projections inherit standard Q2_K's Q3_K promotion;
//! - token embeddings, output, attention/compressor, hyper-connection, and
//!   indexer matrices use Q8_0;
//! - norms, biases, router scalars, and integer hash tables are handled by the
//!   orchestrator's existing F32/I32 gates before this overlay runs.
//!
//! The layout is based on independently published DeepSeek-V4 Q2_K-XXL/XL
//! recipes and verified against their GGUF tensor metadata. It is deliberately
//! architecture-specific rather than a silent change to the standard Q2_K
//! policy used by every other model family.

use super::{ArchName, GgmlType, TensorRef};

/// Canonical CLI/receipt spelling for this architecture-specific profile.
pub const DEEPSEEK4_AGENTIC_Q2_NAME: &str = "deepseek4-agentic-q2";
/// GGUF metadata key that preserves the mixed profile's identity for serving
/// diagnostics instead of reducing it to whichever tensor type has most files.
pub const DEEPSEEK4_AGENTIC_Q2_METADATA_KEY: &str = "hf2q.quantization.profile";

/// Stateless overlay applied after the standard `MostlyQ2_K` policy.
#[derive(Debug, Clone, Copy, Default)]
pub struct Deepseek4AgenticQ2Policy;

impl Deepseek4AgenticQ2Policy {
    pub const fn new() -> Self {
        Self
    }

    /// Promote the context-discrimination path to Q8_0 while preserving the
    /// standard Q2_K/Q3_K expert-body decision in `base_type`.
    pub fn target_for(self, tensor: &TensorRef<'_>, base_type: GgmlType) -> GgmlType {
        assert_eq!(
            tensor.arch,
            ArchName::Deepseek4,
            "DeepSeek-V4 agentic quantization cannot be applied to another architecture"
        );
        if is_q8_pinned_tensor(tensor.name) {
            GgmlType::Q8_0
        } else {
            base_type
        }
    }
}

fn is_q8_pinned_tensor(name: &str) -> bool {
    if matches!(name, "output.weight" | "token_embd.weight") {
        return true;
    }

    // Hyper-connections replace ordinary residual connections in V4. The
    // one-dimensional base/scale tensors remain F32 via the earlier keep gate;
    // this catches the two-dimensional mixers (`*_fn.weight`).
    if name.starts_with("output_hc_") || name.contains(".hc_attn_") || name.contains(".hc_ffn_") {
        return true;
    }

    // Bare `indexer` intentionally catches both `indexer.*` projections and
    // `indexer_compressor_*`. Restricting this to `indexer_` misses the large
    // dotted projections that drive sparse top-k selection.
    if name.contains("indexer") {
        return true;
    }

    name.contains(".attn_compressor_")
        || name.contains(".attn_q_a.weight")
        || name.contains(".attn_q_b.weight")
        || name.contains(".attn_kv.weight")
        || name.contains(".attn_output_a.weight")
        || name.contains(".attn_output_b.weight")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::ggml_quants::{ArchName, SourceDtype};

    fn tensor(name: &str) -> TensorRef<'_> {
        TensorRef {
            name,
            shape: &[4096, 4096],
            source_dtype: SourceDtype::F32,
            arch: ArchName::Deepseek4,
            layer_index: Some(2),
        }
    }

    #[test]
    fn pins_every_deepseek_context_discrimination_family() {
        let policy = Deepseek4AgenticQ2Policy::new();
        for name in [
            "output.weight",
            "token_embd.weight",
            "output_hc_fn.weight",
            "blk.2.hc_attn_fn.weight",
            "blk.2.hc_ffn_fn.weight",
            "blk.2.attn_compressor_gate.weight",
            "blk.2.attn_compressor_kv.weight",
            "blk.2.attn_q_a.weight",
            "blk.2.attn_q_b.weight",
            "blk.2.attn_kv.weight",
            "blk.2.attn_output_a.weight",
            "blk.2.attn_output_b.weight",
            "blk.2.indexer.attn_q_b.weight",
            "blk.2.indexer.proj.weight",
            "blk.2.indexer_compressor_gate.weight",
        ] {
            assert_eq!(
                policy.target_for(&tensor(name), GgmlType::Q2_K),
                GgmlType::Q8_0,
                "{name}"
            );
        }
    }

    #[test]
    fn preserves_routed_and_shared_expert_body_types() {
        let policy = Deepseek4AgenticQ2Policy::new();
        for (name, base) in [
            ("blk.2.ffn_gate_exps.weight", GgmlType::Q2_K),
            ("blk.2.ffn_up_exps.weight", GgmlType::Q2_K),
            ("blk.2.ffn_down_exps.weight", GgmlType::Q3_K),
            ("blk.2.ffn_gate_shexp.weight", GgmlType::Q2_K),
            ("blk.2.ffn_up_shexp.weight", GgmlType::Q2_K),
            ("blk.2.ffn_down_shexp.weight", GgmlType::Q3_K),
        ] {
            assert_eq!(policy.target_for(&tensor(name), base), base, "{name}");
        }
    }

    #[test]
    #[should_panic(expected = "cannot be applied to another architecture")]
    fn rejects_non_deepseek_architectures_in_release_and_debug_builds() {
        let tensor = TensorRef {
            name: "output.weight",
            shape: &[256, 256],
            source_dtype: SourceDtype::F32,
            arch: ArchName::Llama3,
            layer_index: None,
        };
        let _ = Deepseek4AgenticQ2Policy::new().target_for(&tensor, GgmlType::Q2_K);
    }
}
