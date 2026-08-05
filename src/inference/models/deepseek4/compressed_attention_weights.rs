//! Borrowed compressor and indexer tensor bundles for compressed attention.

use anyhow::{bail, Result};
use mlx_native::MlxBuffer;

use super::attention_weights::AttentionWeightsRef;
use super::residency::{Deepseek4Weights, RawMatrixRef};

pub(super) struct CompressorWeightsRef<'a> {
    pub kv: RawMatrixRef<'a>,
    pub gate: RawMatrixRef<'a>,
    pub ape: &'a MlxBuffer,
    pub norm: &'a MlxBuffer,
}

pub(super) struct IndexerWeightsRef<'a> {
    pub q_b: RawMatrixRef<'a>,
    pub projection: RawMatrixRef<'a>,
    pub compressor: CompressorWeightsRef<'a>,
}

pub(super) struct CompressedAttentionWeightsRef<'a> {
    pub attention: AttentionWeightsRef<'a>,
    pub compressor: CompressorWeightsRef<'a>,
    pub indexer: Option<IndexerWeightsRef<'a>>,
}

impl<'a> CompressedAttentionWeightsRef<'a> {
    pub(super) fn get(weights: &'a Deepseek4Weights, layer: usize, ratio: u32) -> Result<Self> {
        if !matches!(ratio, 4 | 128) {
            bail!("DeepSeek-V4 compressed attention requires ratio 4 or 128, got {ratio}");
        }
        let prefix = format!("blk.{layer}");
        let compressor = CompressorWeightsRef {
            kv: weights.raw_matrix_ref(&format!("{prefix}.attn_compressor_kv.weight"))?,
            gate: weights.raw_matrix_ref(&format!("{prefix}.attn_compressor_gate.weight"))?,
            ape: weights.f32_state(&format!("{prefix}.attn_compressor_ape.weight"))?,
            norm: weights.f32_state(&format!("{prefix}.attn_compressor_norm.weight"))?,
        };
        let indexer = if ratio == 4 {
            Some(IndexerWeightsRef {
                q_b: weights.raw_matrix_ref(&format!("{prefix}.indexer.attn_q_b.weight"))?,
                projection: weights.raw_matrix_ref(&format!("{prefix}.indexer.proj.weight"))?,
                compressor: CompressorWeightsRef {
                    kv: weights
                        .raw_matrix_ref(&format!("{prefix}.indexer_compressor_kv.weight"))?,
                    gate: weights
                        .raw_matrix_ref(&format!("{prefix}.indexer_compressor_gate.weight"))?,
                    ape: weights.f32_state(&format!("{prefix}.indexer_compressor_ape.weight"))?,
                    norm: weights.f32_state(&format!("{prefix}.indexer_compressor_norm.weight"))?,
                },
            })
        } else {
            None
        };
        Ok(Self {
            attention: AttentionWeightsRef::get(weights, layer)?,
            compressor,
            indexer,
        })
    }
}
