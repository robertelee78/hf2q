//! Exact tensor catalog for a DeepSeek-V4 verifier GGUF.
//!
//! The catalog is checked before allocating model weights. It deliberately
//! excludes the `mtp.*` DSpark draft stage, which is converted to a separate
//! artifact rather than being mistaken for verifier layers.

use std::collections::{HashMap, HashSet};

use mlx_native::gguf::GgufFile;
use thiserror::Error;

use super::Deepseek4Config;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorRole {
    /// Token table consumed by an artifact-native embedding gather.
    EmbeddingMatrix,
    /// Ordinary two-dimensional projection consumed by dense matmul.
    DenseMatrix,
    /// Output-A matrix reinterpreted as one batch per output group.
    GroupedMatrix,
    /// Three-dimensional expert stack consumed through selected expert IDs.
    ExpertStack,
    /// Elementwise state expanded to F32 when it becomes resident.
    ElementwiseF32,
    /// Hash-routing table kept as canonical signed I32 values.
    IntegerLookupI32,
}

impl TensorRole {
    pub(super) fn is_native_matrix(self) -> bool {
        matches!(
            self,
            Self::EmbeddingMatrix | Self::DenseMatrix | Self::GroupedMatrix | Self::ExpertStack
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorSpec {
    pub name: String,
    /// Outer-first shape, matching `mlx_native::gguf::TensorInfo::shape`.
    pub shape: Vec<usize>,
    pub role: TensorRole,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum WeightCatalogError {
    #[error("missing required DeepSeek-V4 tensor '{name}'")]
    Missing { name: String },
    #[error("unexpected tensor '{name}' in verifier artifact")]
    Unexpected { name: String },
    #[error("tensor '{name}' has shape {actual:?}, expected {expected:?}")]
    Shape {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}

fn spec(name: impl Into<String>, shape: impl Into<Vec<usize>>, role: TensorRole) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        shape: shape.into(),
        role,
    }
}

fn matrix(name: impl Into<String>, rows: usize, columns: usize) -> TensorSpec {
    spec(name, vec![rows, columns], TensorRole::DenseMatrix)
}

fn embedding(name: impl Into<String>, rows: usize, columns: usize) -> TensorSpec {
    spec(name, vec![rows, columns], TensorRole::EmbeddingMatrix)
}

fn grouped_matrix(name: impl Into<String>, rows: usize, columns: usize) -> TensorSpec {
    spec(name, vec![rows, columns], TensorRole::GroupedMatrix)
}

fn state(name: impl Into<String>, shape: impl Into<Vec<usize>>) -> TensorSpec {
    spec(name, shape, TensorRole::ElementwiseF32)
}

/// Generate the complete tensor contract for the verifier artifact described
/// by `cfg`. Shapes follow the official 0731 reference implementation.
pub fn required_tensor_specs(cfg: &Deepseek4Config) -> Vec<TensorSpec> {
    let hidden = cfg.hidden_size as usize;
    let vocab = cfg.vocab_size as usize;
    let hc = cfg.hyper_connection_count as usize;
    let heads = cfg.num_attention_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_rank = cfg.q_lora_rank as usize;
    let o_rank = cfg.o_lora_rank as usize;
    let o_groups = cfg.output_groups as usize;
    let experts = cfg.num_experts as usize;
    let expert_dim = cfg.expert_intermediate_size as usize;
    let top_k = cfg.num_experts_per_tok as usize;
    let mix_hc = (2 + hc) * hc;
    let hc_hidden = hc * hidden;

    let mut specs = vec![
        embedding("token_embd.weight", vocab, hidden),
        state("output_norm.weight", vec![hidden]),
        matrix("output.weight", vocab, hidden),
        matrix("output_hc_fn.weight", hc, hc_hidden),
        state("output_hc_base.weight", vec![hc]),
        state("output_hc_scale.weight", vec![1]),
    ];

    for layer in 0..cfg.num_hidden_layers as usize {
        let prefix = format!("blk.{layer}");
        for sublayer in ["attn", "ffn"] {
            specs.push(matrix(
                format!("{prefix}.hc_{sublayer}_fn.weight"),
                mix_hc,
                hc_hidden,
            ));
            specs.push(state(
                format!("{prefix}.hc_{sublayer}_base.weight"),
                vec![mix_hc],
            ));
            specs.push(state(
                format!("{prefix}.hc_{sublayer}_scale.weight"),
                vec![3],
            ));
        }

        specs.extend([
            state(format!("{prefix}.attn_sinks.weight"), vec![heads]),
            matrix(format!("{prefix}.attn_q_a.weight"), q_rank, hidden),
            state(format!("{prefix}.attn_q_a_norm.weight"), vec![q_rank]),
            matrix(
                format!("{prefix}.attn_q_b.weight"),
                heads * head_dim,
                q_rank,
            ),
            matrix(format!("{prefix}.attn_kv.weight"), head_dim, hidden),
            state(format!("{prefix}.attn_kv_a_norm.weight"), vec![head_dim]),
            grouped_matrix(
                format!("{prefix}.attn_output_a.weight"),
                o_groups * o_rank,
                heads * head_dim / o_groups,
            ),
            matrix(
                format!("{prefix}.attn_output_b.weight"),
                hidden,
                o_groups * o_rank,
            ),
            state(format!("{prefix}.attn_norm.weight"), vec![hidden]),
            state(format!("{prefix}.ffn_norm.weight"), vec![hidden]),
            matrix(format!("{prefix}.ffn_gate_inp.weight"), experts, hidden),
            matrix(
                format!("{prefix}.ffn_gate_shexp.weight"),
                expert_dim,
                hidden,
            ),
            matrix(format!("{prefix}.ffn_up_shexp.weight"), expert_dim, hidden),
            matrix(
                format!("{prefix}.ffn_down_shexp.weight"),
                hidden,
                expert_dim,
            ),
            spec(
                format!("{prefix}.ffn_gate_exps.weight"),
                vec![experts, expert_dim, hidden],
                TensorRole::ExpertStack,
            ),
            spec(
                format!("{prefix}.ffn_up_exps.weight"),
                vec![experts, expert_dim, hidden],
                TensorRole::ExpertStack,
            ),
            spec(
                format!("{prefix}.ffn_down_exps.weight"),
                vec![experts, hidden, expert_dim],
                TensorRole::ExpertStack,
            ),
        ]);

        if layer < cfg.hash_layer_count as usize {
            specs.push(spec(
                format!("{prefix}.ffn_gate_tid2eid.weight"),
                vec![vocab, top_k],
                TensorRole::IntegerLookupI32,
            ));
        } else {
            specs.push(state(format!("{prefix}.exp_probs_b.bias"), vec![experts]));
        }

        let ratio = cfg.compress_ratios[layer] as usize;
        if ratio > 0 {
            let overlap_factor = if ratio == 4 { 2 } else { 1 };
            specs.extend([
                state(
                    format!("{prefix}.attn_compressor_ape.weight"),
                    vec![ratio, overlap_factor * head_dim],
                ),
                matrix(
                    format!("{prefix}.attn_compressor_kv.weight"),
                    overlap_factor * head_dim,
                    hidden,
                ),
                matrix(
                    format!("{prefix}.attn_compressor_gate.weight"),
                    overlap_factor * head_dim,
                    hidden,
                ),
                state(
                    format!("{prefix}.attn_compressor_norm.weight"),
                    vec![head_dim],
                ),
            ]);
        }
        if ratio == 4 {
            let index_heads = cfg.index_num_heads as usize;
            let index_dim = cfg.index_head_dim as usize;
            specs.extend([
                matrix(
                    format!("{prefix}.indexer.attn_q_b.weight"),
                    index_heads * index_dim,
                    q_rank,
                ),
                matrix(format!("{prefix}.indexer.proj.weight"), index_heads, hidden),
                state(
                    format!("{prefix}.indexer_compressor_ape.weight"),
                    vec![ratio, 2 * index_dim],
                ),
                matrix(
                    format!("{prefix}.indexer_compressor_kv.weight"),
                    2 * index_dim,
                    hidden,
                ),
                matrix(
                    format!("{prefix}.indexer_compressor_gate.weight"),
                    2 * index_dim,
                    hidden,
                ),
                state(
                    format!("{prefix}.indexer_compressor_norm.weight"),
                    vec![index_dim],
                ),
            ]);
        }
    }
    specs
}

/// Check that a GGUF contains exactly the verifier tensor set and that every
/// outer-first shape matches the model metadata.
pub fn validate_tensor_catalog(
    gguf: &GgufFile,
    cfg: &Deepseek4Config,
) -> Result<Vec<TensorSpec>, WeightCatalogError> {
    let specs = required_tensor_specs(cfg);
    let by_name: HashMap<&str, &TensorSpec> = specs
        .iter()
        .map(|entry| (entry.name.as_str(), entry))
        .collect();
    for entry in &specs {
        let info = gguf
            .tensor_info(&entry.name)
            .ok_or_else(|| WeightCatalogError::Missing {
                name: entry.name.clone(),
            })?;
        if info.shape != entry.shape {
            return Err(WeightCatalogError::Shape {
                name: entry.name.clone(),
                expected: entry.shape.clone(),
                actual: info.shape.clone(),
            });
        }
    }
    let expected: HashSet<&str> = by_name.keys().copied().collect();
    if let Some(name) = gguf
        .tensor_names()
        .into_iter()
        .filter(|name| !expected.contains(name))
        .min()
    {
        return Err(WeightCatalogError::Unexpected {
            name: name.to_string(),
        });
    }
    Ok(specs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn official_cfg() -> Deepseek4Config {
        Deepseek4Config {
            num_hidden_layers: 43,
            hidden_size: 4096,
            hidden_size_out: 16384,
            max_position_embeddings: 1_048_576,
            vocab_size: 129280,
            num_attention_heads: 64,
            num_key_value_heads: 1,
            head_dim: 512,
            rope_head_dim: 64,
            rope_theta: 10000.0,
            rope_factor: 16.0,
            original_context_length: 65536,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            q_lora_rank: 1024,
            o_lora_rank: 1024,
            output_groups: 8,
            sliding_window: 128,
            compress_ratios: (0..43)
                .map(|layer| {
                    if layer < 2 {
                        0
                    } else if layer % 2 == 0 {
                        4
                    } else {
                        128
                    }
                })
                .collect(),
            compress_rope_theta: 160000.0,
            index_num_heads: 64,
            index_head_dim: 128,
            index_top_k: 512,
            rms_norm_eps: 1e-6,
            num_experts: 256,
            num_experts_per_tok: 6,
            num_shared_experts: 1,
            expert_intermediate_size: 2048,
            route_scale: 1.5,
            normalize_topk: true,
            swiglu_clamp_experts: vec![10.0; 43],
            swiglu_clamp_shared: vec![10.0; 43],
            hyper_connection_count: 4,
            hyper_connection_sinkhorn_iterations: 20,
            hyper_connection_epsilon: 1e-6,
            hash_layer_count: 3,
        }
    }

    #[test]
    fn official_catalog_covers_hash_compressed_and_indexed_variants() {
        let specs = required_tensor_specs(&official_cfg());
        let by_name: HashMap<_, _> = specs.iter().map(|s| (s.name.as_str(), s)).collect();
        assert_eq!(
            by_name["blk.0.ffn_gate_tid2eid.weight"].shape,
            vec![129280, 6]
        );
        assert!(!by_name.contains_key("blk.0.exp_probs_b.bias"));
        assert!(by_name.contains_key("blk.3.attn_compressor_ape.weight"));
        assert_eq!(
            by_name["blk.3.attn_compressor_ape.weight"].role,
            TensorRole::ElementwiseF32
        );
        assert!(!by_name.contains_key("blk.3.indexer.attn_q_b.weight"));
        assert_eq!(
            by_name["blk.4.indexer.attn_q_b.weight"].shape,
            vec![8192, 1024]
        );
        assert_eq!(
            by_name["blk.4.indexer_compressor_ape.weight"].role,
            TensorRole::ElementwiseF32
        );
        assert_eq!(
            by_name["blk.0.ffn_gate_tid2eid.weight"].role,
            TensorRole::IntegerLookupI32
        );
        assert_eq!(
            by_name["blk.42.ffn_down_exps.weight"].shape,
            vec![256, 4096, 2048]
        );
        assert!(by_name.keys().all(|name| !name.starts_with("mtp.")));
    }
}
