//! Bounded, reusable DFlash2 decode workspace.

use super::config::DFlash2Config;
use mlx_native::{DType, MlxBuffer, MlxDevice, MlxError};

#[derive(Debug, thiserror::Error)]
pub enum DFlash2WorkspaceError {
    #[error("invalid DFlash2 workspace: {0}")]
    Invalid(String),
    #[error("DFlash2 workspace allocation: {0}")]
    Mlx(#[from] MlxError),
}

pub struct DFlash2Workspace {
    pub max_lanes: usize,
    pub rows_per_lane: usize,
    pub row_tokens: MlxBuffer,
    pub captured_target_hidden: MlxBuffer,
    pub hidden: [MlxBuffer; 2],
    pub normalized: MlxBuffer,
    pub af4_table: MlxBuffer,
    pub af4_sums: MlxBuffer,
    pub dynamic: MlxBuffer,
    pub convolved: MlxBuffer,
    pub residual: MlxBuffer,
    pub qkv: MlxBuffer,
    pub query: MlxBuffer,
    pub key: MlxBuffer,
    pub value: MlxBuffer,
    pub query_normalized: MlxBuffer,
    pub key_normalized: MlxBuffer,
    pub attention: MlxBuffer,
    pub packed_query: MlxBuffer,
    pub packed_key: MlxBuffer,
    pub packed_value: MlxBuffer,
    pub attention_bf16: MlxBuffer,
    pub attention_mask: MlxBuffer,
    pub ring_keys: Vec<MlxBuffer>,
    pub ring_values: Vec<MlxBuffer>,
    pub cache_lengths: MlxBuffer,
    pub write_positions: MlxBuffer,
    pub rope_positions: MlxBuffer,
    pub rope_params: MlxBuffer,
    pub norm_params: MlxBuffer,
    pub head_norm_params: MlxBuffer,
    pub silu_params: MlxBuffer,
    pub softcap_params: MlxBuffer,
    pub head_meta: MlxBuffer,
    pub projected: MlxBuffer,
    pub gate_up: MlxBuffer,
    pub gate: MlxBuffer,
    pub up: MlxBuffer,
    pub intermediate: MlxBuffer,
    pub final_hidden: MlxBuffer,
    pub logits: MlxBuffer,
    pub selector_hidden_f32: MlxBuffer,
    pub selector_hidden_bf16: MlxBuffer,
    pub uniforms: MlxBuffer,
    pub partial_ids: MlxBuffer,
    pub partial_values: MlxBuffer,
    pub candidates: MlxBuffer,
    pub unary: MlxBuffer,
    pub edges: MlxBuffer,
    pub proposed_tokens: MlxBuffer,
    pub proposal_probabilities: MlxBuffer,
    pub confidence: MlxBuffer,
    pub verify_rows: MlxBuffer,
    allocated_bytes: usize,
}

fn elements(shape: &[usize]) -> Result<usize, DFlash2WorkspaceError> {
    shape.iter().try_fold(1usize, |count, dim| {
        count.checked_mul(*dim).ok_or_else(|| {
            DFlash2WorkspaceError::Invalid(format!("shape product overflows: {shape:?}"))
        })
    })
}

fn allocate(
    device: &MlxDevice,
    dtype: DType,
    shape: &[usize],
    bytes: &mut usize,
) -> Result<MlxBuffer, DFlash2WorkspaceError> {
    let size = elements(shape)?
        .checked_mul(dtype.size_of())
        .ok_or_else(|| DFlash2WorkspaceError::Invalid(format!("byte size overflows: {shape:?}")))?;
    *bytes = bytes
        .checked_add(size)
        .ok_or_else(|| DFlash2WorkspaceError::Invalid("workspace total overflows".into()))?;
    Ok(device.alloc_buffer(size, dtype, shape.to_vec())?)
}

impl DFlash2Workspace {
    pub fn new(
        device: &MlxDevice,
        config: &DFlash2Config,
        max_lanes: usize,
    ) -> Result<Self, DFlash2WorkspaceError> {
        if max_lanes == 0 || max_lanes > 4 {
            return Err(DFlash2WorkspaceError::Invalid(format!(
                "max_lanes must be in 1..=4, got {max_lanes}"
            )));
        }
        let lanes = max_lanes;
        let block = config.block_size;
        let positions = config.proposal_tokens();
        let rows = lanes.checked_mul(block).ok_or_else(|| {
            DFlash2WorkspaceError::Invalid("workspace row count overflows".into())
        })?;
        let hidden = config.hidden_size;
        let query = config.num_attention_heads * config.head_dim;
        let key_value = config.num_key_value_heads * config.head_dim;
        let intermediate = config.intermediate_size;
        let packed = config.sliding_window - 1 + block;
        let max_projection_input = config
            .fc_input_dim()
            .max(hidden)
            .max(query)
            .max(intermediate);
        let mut allocated_bytes = 0usize;

        let row_tokens = allocate(device, DType::U32, &[lanes, block], &mut allocated_bytes)?;
        let captured_target_hidden = allocate(
            device,
            DType::F32,
            &[rows, config.fc_input_dim()],
            &mut allocated_bytes,
        )?;
        let hidden_a = allocate(device, DType::F32, &[rows, hidden], &mut allocated_bytes)?;
        let hidden_b = allocate(device, DType::F32, &[rows, hidden], &mut allocated_bytes)?;
        let normalized = allocate(device, DType::F32, &[rows, hidden], &mut allocated_bytes)?;
        let af4_table = allocate(
            device,
            DType::BF16,
            &[rows.div_ceil(8), 8, max_projection_input],
            &mut allocated_bytes,
        )?;
        let af4_sums = allocate(
            device,
            DType::F32,
            &[rows.div_ceil(8), max_projection_input / 64, 8],
            &mut allocated_bytes,
        )?;
        let dynamic = allocate(
            device,
            DType::F32,
            &[rows, config.dynamic_projection_width()],
            &mut allocated_bytes,
        )?;
        let convolved = allocate(device, DType::F32, &[rows, hidden], &mut allocated_bytes)?;
        let residual = allocate(device, DType::F32, &[rows, hidden], &mut allocated_bytes)?;
        let qkv = allocate(
            device,
            DType::F32,
            &[rows, query + 2 * key_value],
            &mut allocated_bytes,
        )?;
        let query_buffer = allocate(device, DType::F32, &[rows, query], &mut allocated_bytes)?;
        let key = allocate(device, DType::F32, &[rows, key_value], &mut allocated_bytes)?;
        let value = allocate(device, DType::F32, &[rows, key_value], &mut allocated_bytes)?;
        let query_normalized = allocate(device, DType::F32, &[rows, query], &mut allocated_bytes)?;
        let key_normalized =
            allocate(device, DType::F32, &[rows, key_value], &mut allocated_bytes)?;
        let attention = allocate(device, DType::F32, &[rows, query], &mut allocated_bytes)?;
        let packed_query = allocate(
            device,
            DType::BF16,
            &[lanes, query / config.head_dim, block, config.head_dim],
            &mut allocated_bytes,
        )?;
        let packed_key = allocate(
            device,
            DType::BF16,
            &[lanes, key_value / config.head_dim, packed, config.head_dim],
            &mut allocated_bytes,
        )?;
        let packed_value = allocate(
            device,
            DType::BF16,
            &[lanes, key_value / config.head_dim, packed, config.head_dim],
            &mut allocated_bytes,
        )?;
        let attention_bf16 = allocate(
            device,
            DType::BF16,
            &[lanes, query / config.head_dim, block, config.head_dim],
            &mut allocated_bytes,
        )?;
        let attention_mask = allocate(
            device,
            DType::BF16,
            &[lanes, query / config.head_dim, block, packed],
            &mut allocated_bytes,
        )?;
        let mut ring_keys = Vec::with_capacity(config.num_hidden_layers);
        let mut ring_values = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            ring_keys.push(allocate(
                device,
                DType::BF16,
                &[
                    lanes,
                    key_value / config.head_dim,
                    config.sliding_window,
                    config.head_dim,
                ],
                &mut allocated_bytes,
            )?);
            ring_values.push(allocate(
                device,
                DType::BF16,
                &[
                    lanes,
                    key_value / config.head_dim,
                    config.sliding_window,
                    config.head_dim,
                ],
                &mut allocated_bytes,
            )?);
        }
        let cache_lengths = allocate(device, DType::U32, &[lanes], &mut allocated_bytes)?;
        let write_positions = allocate(device, DType::U32, &[lanes], &mut allocated_bytes)?;
        let rope_positions = allocate(device, DType::U32, &[rows], &mut allocated_bytes)?;
        let rope_params = allocate(device, DType::F32, &[4], &mut allocated_bytes)?;
        let mut norm_params = allocate(device, DType::F32, &[2], &mut allocated_bytes)?;
        norm_params
            .as_mut_slice::<f32>()?
            .copy_from_slice(&[config.rms_norm_eps, config.hidden_size as f32]);
        let mut head_norm_params = allocate(device, DType::F32, &[2], &mut allocated_bytes)?;
        head_norm_params
            .as_mut_slice::<f32>()?
            .copy_from_slice(&[config.rms_norm_eps, config.head_dim as f32]);
        let mut rope_params = rope_params;
        rope_params.as_mut_slice::<f32>()?.copy_from_slice(&[
            config.rope_theta as f32,
            config.head_dim as f32,
            config.head_dim as f32,
            0.0,
        ]);
        let mut silu_params = allocate(device, DType::U32, &[1], &mut allocated_bytes)?;
        silu_params.as_mut_slice::<u32>()?[0] = u32::try_from(rows * intermediate)
            .map_err(|_| DFlash2WorkspaceError::Invalid("SiLU element count exceeds u32".into()))?;
        let mut softcap_params = allocate(device, DType::F32, &[2], &mut allocated_bytes)?;
        softcap_params.as_mut_slice::<f32>()?.copy_from_slice(&[
            config.final_logit_softcapping.unwrap_or(1.0),
            f32::from_bits(u32::try_from(rows * config.vocab_size).map_err(|_| {
                DFlash2WorkspaceError::Invalid("logit element count exceeds u32".into())
            })?),
        ]);
        let head_meta = allocate(device, DType::U32, &[4], &mut allocated_bytes)?;
        let projected = allocate(device, DType::F32, &[rows, hidden], &mut allocated_bytes)?;
        let gate_up = allocate(
            device,
            DType::F32,
            &[rows, 2 * intermediate],
            &mut allocated_bytes,
        )?;
        let gate = allocate(
            device,
            DType::F32,
            &[rows, intermediate],
            &mut allocated_bytes,
        )?;
        let up = allocate(
            device,
            DType::F32,
            &[rows, intermediate],
            &mut allocated_bytes,
        )?;
        let intermediate_buffer = allocate(
            device,
            DType::F32,
            &[rows, intermediate],
            &mut allocated_bytes,
        )?;
        let final_hidden = allocate(device, DType::F32, &[rows, hidden], &mut allocated_bytes)?;
        let logits = allocate(
            device,
            DType::F32,
            &[lanes, block, config.vocab_size],
            &mut allocated_bytes,
        )?;
        let selector_hidden_f32 = allocate(
            device,
            DType::F32,
            &[lanes, block, config.selector_rank],
            &mut allocated_bytes,
        )?;
        let selector_hidden_bf16 = allocate(
            device,
            DType::BF16,
            &[lanes, block, config.selector_rank],
            &mut allocated_bytes,
        )?;
        let uniforms = allocate(
            device,
            DType::F32,
            &[lanes, positions],
            &mut allocated_bytes,
        )?;
        let partial_ids = allocate(
            device,
            DType::U32,
            &[lanes, positions, 8, 16],
            &mut allocated_bytes,
        )?;
        let partial_values = allocate(
            device,
            DType::F32,
            &[lanes, positions, 8, 16],
            &mut allocated_bytes,
        )?;
        let candidates = allocate(
            device,
            DType::U32,
            &[lanes, positions, 16],
            &mut allocated_bytes,
        )?;
        let unary = allocate(
            device,
            DType::F32,
            &[lanes, positions, 16],
            &mut allocated_bytes,
        )?;
        let edges = allocate(
            device,
            DType::F32,
            &[lanes, positions, 16, 16],
            &mut allocated_bytes,
        )?;
        let proposed_tokens = allocate(
            device,
            DType::U32,
            &[lanes, positions],
            &mut allocated_bytes,
        )?;
        let proposal_probabilities = allocate(
            device,
            DType::F32,
            &[lanes, positions, 16],
            &mut allocated_bytes,
        )?;
        let confidence = allocate(
            device,
            DType::F32,
            &[lanes, positions],
            &mut allocated_bytes,
        )?;
        let verify_rows = allocate(device, DType::U32, &[lanes], &mut allocated_bytes)?;

        Ok(Self {
            max_lanes,
            rows_per_lane: block,
            row_tokens,
            captured_target_hidden,
            hidden: [hidden_a, hidden_b],
            normalized,
            af4_table,
            af4_sums,
            dynamic,
            convolved,
            residual,
            qkv,
            query: query_buffer,
            key,
            value,
            query_normalized,
            key_normalized,
            attention,
            projected,
            gate_up,
            gate,
            up,
            packed_query,
            packed_key,
            packed_value,
            attention_bf16,
            attention_mask,
            ring_keys,
            ring_values,
            cache_lengths,
            write_positions,
            rope_positions,
            rope_params,
            norm_params,
            head_norm_params,
            silu_params,
            intermediate: intermediate_buffer,
            softcap_params,
            head_meta,
            final_hidden,
            logits,
            selector_hidden_f32,
            selector_hidden_bf16,
            uniforms,
            partial_ids,
            partial_values,
            candidates,
            unary,
            edges,
            proposed_tokens,
            proposal_probabilities,
            confidence,
            verify_rows,
            allocated_bytes,
        })
    }

    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_product_rejects_overflow() {
        let error = elements(&[usize::MAX, 2]).unwrap_err();
        assert!(error.to_string().contains("overflows"));
    }
}
