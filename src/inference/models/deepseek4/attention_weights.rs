//! Borrowed per-layer attention tensor bundle.

use anyhow::Result;
use mlx_native::MlxBuffer;

use super::residency::{Deepseek4Weights, RawMatrixRef};

pub(super) struct AttentionWeightsRef<'a> {
    pub hc_fn: RawMatrixRef<'a>,
    pub hc_base: &'a MlxBuffer,
    pub hc_scale: &'a MlxBuffer,
    pub attn_norm: &'a MlxBuffer,
    pub q_a: RawMatrixRef<'a>,
    pub q_a_norm: &'a MlxBuffer,
    pub q_b: RawMatrixRef<'a>,
    pub kv: RawMatrixRef<'a>,
    pub kv_norm: &'a MlxBuffer,
    pub sinks: &'a MlxBuffer,
    pub output_a: RawMatrixRef<'a>,
    pub output_b: RawMatrixRef<'a>,
}

impl<'a> AttentionWeightsRef<'a> {
    pub(super) fn get(weights: &'a Deepseek4Weights, layer: usize) -> Result<Self> {
        let prefix = format!("blk.{layer}");
        Ok(Self {
            hc_fn: weights.raw_matrix_ref(&format!("{prefix}.hc_attn_fn.weight"))?,
            hc_base: weights.f32_state(&format!("{prefix}.hc_attn_base.weight"))?,
            hc_scale: weights.f32_state(&format!("{prefix}.hc_attn_scale.weight"))?,
            attn_norm: weights.f32_state(&format!("{prefix}.attn_norm.weight"))?,
            q_a: weights.raw_matrix_ref(&format!("{prefix}.attn_q_a.weight"))?,
            q_a_norm: weights.f32_state(&format!("{prefix}.attn_q_a_norm.weight"))?,
            q_b: weights.raw_matrix_ref(&format!("{prefix}.attn_q_b.weight"))?,
            kv: weights.raw_matrix_ref(&format!("{prefix}.attn_kv.weight"))?,
            kv_norm: weights.f32_state(&format!("{prefix}.attn_kv_a_norm.weight"))?,
            sinks: weights.f32_state(&format!("{prefix}.attn_sinks.weight"))?,
            output_a: weights.raw_matrix_ref(&format!("{prefix}.attn_output_a.weight"))?,
            output_b: weights.raw_matrix_ref(&format!("{prefix}.attn_output_b.weight"))?,
        })
    }
}
