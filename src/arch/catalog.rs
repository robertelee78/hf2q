//! TensorCatalog — hand-transcribed per-arch tensor-name/shape catalog.
//!
//! ADR-012 Decision 20. Each `ArchEntry::tensor_catalog` points at a
//! `&'static TensorCatalog` populated with the exact set of GGUF
//! tensors every converted model of that arch MUST contain. The P8
//! smoke harness (Decision 16) asserts the loaded tensor count matches
//! `tensor_catalog.expected_tensor_count(metadata)` — not a side-file.
//!
//! Per-layer tensors are emitted as templates with a `{L}` placeholder
//! that is expanded at smoke-eval time against the model's
//! `num_hidden_layers` (and per-layer kind for hybrid arches: linear
//! vs full attention, dense vs MoE FFN).

/// Tensor element dtype, matching the underlying GGUF ggml type surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDtype {
    /// f16 (ggml_type GGML_TYPE_F16).
    F16,
    /// f32 (ggml_type GGML_TYPE_F32).
    F32,
    /// bf16 (ggml_type GGML_TYPE_BF16).
    BF16,
    /// Any quantized type accepted by the converter (Q4_0, Q6_K, DWQ blocks, ...).
    Quantized,
}

/// One entry in a tensor catalog.
#[derive(Debug, Clone, Copy)]
pub struct TensorCatalogEntry {
    /// Tensor name template. `{L}` is substituted with the layer index at
    /// eval time. Names without `{L}` are emitted once (e.g. `token_embd.weight`).
    pub name_template: &'static str,
    /// Layer scope: `None` = global (one copy), `Some(LayerScope::AllLayers)` = one per block.
    pub scope: LayerScope,
    /// Declared dtype. `Quantized` accepts any quantized block type.
    pub dtype: TensorDtype,
    /// Source of truth citation (file:line in `/opt/llama.cpp/src/llama-arch.cpp`
    /// or `convert_hf_to_gguf.py`). Comment-only; not enforced at runtime but
    /// REQUIRED for every catalog entry per ADR-012 mantra.
    pub citation: &'static str,
}

/// Per-layer scope for a tensor template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerScope {
    /// Emitted once, independent of block index.
    Global,
    /// Emitted once per block index `[0, num_hidden_layers)`.
    AllLayers,
    /// Emitted only for layers whose `Qwen35LayerKind` is `FullAttention`.
    FullAttentionLayersOnly,
    /// Emitted only for layers whose `Qwen35LayerKind` is `LinearAttention`.
    LinearAttentionLayersOnly,
    /// Emitted once per MTP block (`mtp_num_hidden_layers` in HF config).
    MtpLayers,
    /// Emitted once per expert on every MoE block.
    MoeExpertsPerLayer,
    /// Emitted once per MoE block, shared expert tensors.
    MoeSharedExpertPerLayer,
    /// Emitted once per MoE block, router tensor.
    MoeRouterPerLayer,
}

/// Hand-transcribed tensor catalog for one arch.
///
/// The `entries` slice is declarative — it lists *templates*. The
/// `expected_tensor_count` helper folds in `num_hidden_layers`,
/// `num_experts`, `mtp_num_hidden_layers`, and `layer_types` to
/// return the exact expected count for a concrete model.
#[derive(Debug, Clone, Copy)]
pub struct TensorCatalog {
    pub entries: &'static [TensorCatalogEntry],
}

/// Parameters required to expand a catalog against a concrete model.
#[derive(Debug, Clone, Copy)]
pub struct CatalogExpansion {
    pub num_hidden_layers: u32,
    pub num_full_attention_layers: u32,
    pub num_linear_attention_layers: u32,
    pub num_experts: u32,
    pub has_shared_expert: bool,
    pub mtp_num_hidden_layers: u32,
}

impl TensorCatalog {
    /// Compute the expected tensor count for a concrete model's metadata.
    ///
    /// This is the number that `llama_model_load: loaded tensor 0x%x`
    /// must match in the smoke harness output. If the number differs,
    /// the conversion produced a structurally-wrong file and the smoke
    /// gate fails loudly.
    pub fn expected_tensor_count(&self, exp: CatalogExpansion) -> u64 {
        let mut total: u64 = 0;
        for e in self.entries {
            let count = match e.scope {
                LayerScope::Global => 1,
                LayerScope::AllLayers => exp.num_hidden_layers as u64,
                LayerScope::FullAttentionLayersOnly => exp.num_full_attention_layers as u64,
                LayerScope::LinearAttentionLayersOnly => exp.num_linear_attention_layers as u64,
                LayerScope::MtpLayers => exp.mtp_num_hidden_layers as u64,
                LayerScope::MoeExpertsPerLayer => {
                    (exp.num_experts as u64) * (exp.num_hidden_layers as u64)
                }
                LayerScope::MoeSharedExpertPerLayer => {
                    if exp.has_shared_expert {
                        exp.num_hidden_layers as u64
                    } else {
                        0
                    }
                }
                LayerScope::MoeRouterPerLayer => exp.num_hidden_layers as u64,
            };
            total = total.saturating_add(count);
        }
        total
    }

    /// Expand the catalog into concrete tensor names (used by MTP
    /// round-trip tests, not the smoke harness counting path).
    pub fn expand_names(&self, exp: CatalogExpansion) -> Vec<String> {
        let mut names = Vec::new();
        for e in self.entries {
            match e.scope {
                LayerScope::Global => names.push(e.name_template.to_string()),
                LayerScope::AllLayers => {
                    for l in 0..exp.num_hidden_layers {
                        names.push(e.name_template.replace("{L}", &l.to_string()));
                    }
                }
                LayerScope::FullAttentionLayersOnly
                | LayerScope::LinearAttentionLayersOnly => {
                    // Caller must filter by known layer kinds; expansion inserts indices
                    // for every block and downstream matchers filter by kind.
                    for l in 0..exp.num_hidden_layers {
                        names.push(e.name_template.replace("{L}", &l.to_string()));
                    }
                }
                LayerScope::MtpLayers => {
                    for l in 0..exp.mtp_num_hidden_layers {
                        let block = exp.num_hidden_layers + l;
                        names.push(e.name_template.replace("{L}", &block.to_string()));
                    }
                }
                LayerScope::MoeExpertsPerLayer => {
                    for l in 0..exp.num_hidden_layers {
                        for x in 0..exp.num_experts {
                            names.push(
                                e.name_template
                                    .replace("{L}", &l.to_string())
                                    .replace("{X}", &x.to_string()),
                            );
                        }
                    }
                }
                LayerScope::MoeSharedExpertPerLayer => {
                    if exp.has_shared_expert {
                        for l in 0..exp.num_hidden_layers {
                            names.push(e.name_template.replace("{L}", &l.to_string()));
                        }
                    }
                }
                LayerScope::MoeRouterPerLayer => {
                    for l in 0..exp.num_hidden_layers {
                        names.push(e.name_template.replace("{L}", &l.to_string()));
                    }
                }
            }
        }
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EMPTY: TensorCatalog = TensorCatalog { entries: &[] };

    #[test]
    fn empty_catalog_expects_zero() {
        let exp = CatalogExpansion {
            num_hidden_layers: 40,
            num_full_attention_layers: 10,
            num_linear_attention_layers: 30,
            num_experts: 256,
            has_shared_expert: true,
            mtp_num_hidden_layers: 1,
        };
        assert_eq!(EMPTY.expected_tensor_count(exp), 0);
    }

    #[test]
    fn scope_allocations_are_linear_in_their_scope() {
        const CAT: TensorCatalog = TensorCatalog {
            entries: &[
                TensorCatalogEntry {
                    name_template: "token_embd.weight",
                    scope: LayerScope::Global,
                    dtype: TensorDtype::F16,
                    citation: "test",
                },
                TensorCatalogEntry {
                    name_template: "blk.{L}.attn_q.weight",
                    scope: LayerScope::AllLayers,
                    dtype: TensorDtype::Quantized,
                    citation: "test",
                },
                TensorCatalogEntry {
                    name_template: "blk.{L}.nextn.embed.weight",
                    scope: LayerScope::MtpLayers,
                    dtype: TensorDtype::F16,
                    citation: "test",
                },
            ],
        };
        let exp = CatalogExpansion {
            num_hidden_layers: 4,
            num_full_attention_layers: 1,
            num_linear_attention_layers: 3,
            num_experts: 0,
            has_shared_expert: false,
            mtp_num_hidden_layers: 1,
        };
        // 1 global + 4 all-layers + 1 MTP = 6
        assert_eq!(CAT.expected_tensor_count(exp), 6);
        let names = CAT.expand_names(exp);
        assert_eq!(names.len(), 6);
        assert!(names.contains(&"token_embd.weight".to_string()));
        assert!(names.contains(&"blk.0.attn_q.weight".to_string()));
        assert!(names.contains(&"blk.3.attn_q.weight".to_string()));
        // MTP layer is at block N == num_hidden_layers
        assert!(names.contains(&"blk.4.nextn.embed.weight".to_string()));
    }

    #[test]
    fn moe_expert_scope_multiplies_experts_by_layers() {
        const CAT: TensorCatalog = TensorCatalog {
            entries: &[TensorCatalogEntry {
                name_template: "blk.{L}.ffn_gate.{X}.weight",
                scope: LayerScope::MoeExpertsPerLayer,
                dtype: TensorDtype::Quantized,
                citation: "test",
            }],
        };
        let exp = CatalogExpansion {
            num_hidden_layers: 40,
            num_full_attention_layers: 10,
            num_linear_attention_layers: 30,
            num_experts: 256,
            has_shared_expert: false,
            mtp_num_hidden_layers: 0,
        };
        assert_eq!(CAT.expected_tensor_count(exp), 40 * 256);
    }
}
