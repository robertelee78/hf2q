//! Bounded, reusable DFlash2 decode workspace.

use super::{
    config::DFlash2Config,
    staged::DFlash2StagedScratch,
};
use mlx_native::{
    ops::{
        dflash2_af4g64_staged::dflash2_af4g64_staged_splits,
        dflash2_selector::DFLASH2_MAX_LANES,
    },
    DType, MlxBuffer, MlxDevice, MlxError,
};

mod memory;
pub use memory::DFlash2WorkspaceMemory;
pub(super) use memory::{add_bytes, buffer_bytes, WorkspaceGeometry};

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
    /// Packed AF4 decode table; contiguous BF16 input during context FC/KV and QKV.
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
    /// Phase-exclusive F32 alias for context FC/KV and QKV, over `packed_key`'s root.
    pub context_partials: MlxBuffer,
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
    /// Fixed-width target token tile composed on GPU from anchors and proposals.
    pub target_tokens: MlxBuffer,
    pub proposal_probabilities: MlxBuffer,
    pub confidence: MlxBuffer,
    pub verify_rows: MlxBuffer,
    pub retained_counts: MlxBuffer,
    pub accepted_counts: MlxBuffer,
    /// Present when the staged draft route is selected.
    pub staged: Option<DFlash2StagedScratch>,
    allocated_bytes: usize,
}

fn elements(shape: &[usize]) -> Result<usize, DFlash2WorkspaceError> {
    shape.iter().try_fold(1usize, |count, dim| {
        count.checked_mul(*dim).ok_or_else(|| {
            DFlash2WorkspaceError::Invalid(format!("shape product overflows: {shape:?}"))
        })
    })
}

fn context_partial_elements(
    rows: usize,
    output: usize,
    input: usize,
) -> Result<usize, DFlash2WorkspaceError> {
    let output = u32::try_from(output)
        .map_err(|_| DFlash2WorkspaceError::Invalid("context output exceeds u32".into()))?;
    let input = u32::try_from(input)
        .map_err(|_| DFlash2WorkspaceError::Invalid("context input exceeds u32".into()))?;
    // Dense BF16 workspaces also use this geometry. Sub-tile projections
    // cannot use AF4G64's split tile and must not enter its N/64 cost law.
    if output < 64 || input < 64 {
        return Ok(0);
    }
    let splits = dflash2_af4g64_staged_splits(output, input);
    if splits == 1 {
        return Ok(0);
    }
    // Prefill tail tiles keep compact real-row partials, not padded row tiles.
    // Use the native shape-only split law, but check every sizing product.
    elements(&[splits as usize, rows, output as usize])
}

pub(super) fn allocate(
    device: &MlxDevice,
    dtype: DType,
    shape: &[usize],
    bytes: &mut usize,
) -> Result<MlxBuffer, DFlash2WorkspaceError> {
    let size = buffer_bytes(dtype, shape)?;
    add_bytes(bytes, size)?;
    Ok(device.alloc_buffer(size, dtype, shape.to_vec())?)
}

impl DFlash2Workspace {
    pub fn new(
        device: &MlxDevice,
        config: &DFlash2Config,
        max_lanes: usize,
        staged: bool,
    ) -> Result<Self, DFlash2WorkspaceError> {
        let geometry = WorkspaceGeometry::new(config, max_lanes)?;
        let lanes = max_lanes;
        let block = config.block_size;
        let positions = config.proposal_tokens();
        let rows = geometry.rows;
        let hidden = config.hidden_size;
        let query = geometry.query;
        let key_value = geometry.key_value;
        let intermediate = config.intermediate_size;
        let packed = geometry.packed;
        let max_projection_input = geometry.widest_input;
        let mut allocated_bytes = 0usize;

        let row_tokens = allocate(device, DType::U32, &[lanes, block], &mut allocated_bytes)?;
        let captured_target_hidden = allocate(
            device,
            DType::F32,
            &[rows, geometry.fc_input],
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
            &[rows, geometry.dynamic],
            &mut allocated_bytes,
        )?;
        let convolved = allocate(device, DType::F32, &[rows, hidden], &mut allocated_bytes)?;
        let residual = allocate(device, DType::F32, &[rows, hidden], &mut allocated_bytes)?;
        let qkv = allocate(
            device,
            DType::F32,
            &[rows, geometry.qkv],
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
        let packed_key_shape = [lanes, key_value / config.head_dim, packed, config.head_dim];
        let packed_key_elements = geometry.packed_key_elements;
        let context_partials_bytes = geometry.context_partials_bytes;
        // QKV's partial reduction finishes before attention packs packed_key;
        // existing barriers order those accesses. Context commits/ingest never
        // encode attention and write separate persistent ring roots. Runtime
        // settles prefetch before context and directly binds tracked storage
        // when a draft follows a detached context command (runtime.rs).
        // No scratch role retains data across phases. Grow this same root for
        // tiny geometries and charge its capacity once.
        let packed_key_root = allocate(
            device,
            DType::BF16,
            &[geometry.packed_key_root_bytes / DType::BF16.size_of()],
            &mut allocated_bytes,
        )?;
        let context_partials = packed_key_root
            .slice_view(0, context_partials_bytes / DType::BF16.size_of())
            .view_dtype(DType::F32)?;
        let packed_key = packed_key_root
            .slice_view(0, packed_key_elements)
            .with_shape(packed_key_shape.to_vec())?;
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
            &[rows, geometry.gate_up],
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
        // Target verification always uses the canonical width-32 tile, even
        // when the worker owns fewer than four physical draft lanes. The
        // composition kernel pads unused logical lanes with lane 0's anchor.
        let target_tokens = allocate(
            device,
            DType::U32,
            &[DFLASH2_MAX_LANES * block],
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
        let retained_counts = allocate(device, DType::U32, &[lanes], &mut allocated_bytes)?;
        let accepted_counts = allocate(device, DType::U32, &[lanes], &mut allocated_bytes)?;
        let staged = staged
            .then(|| DFlash2StagedScratch::allocate(device, config, lanes, &geometry, &mut allocated_bytes))
            .transpose()?;

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
            context_partials,
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
            target_tokens,
            proposal_probabilities,
            confidence,
            verify_rows,
            retained_counts,
            accepted_counts,
            staged,
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

    #[test]
    fn context_partials_follow_prefill_splits_with_checked_products() {
        assert_eq!(
            context_partial_elements(32, 2048, 16384).unwrap(),
            8 * 32 * 2048
        );
        assert_eq!(
            context_partial_elements(32, 2048, 2048).unwrap(),
            4 * 32 * 2048
        );
        assert_eq!(context_partial_elements(3, 2048, 16384).unwrap(), 8 * 3 * 2048);
        assert_eq!(context_partial_elements(17, 2048, 2048).unwrap(), 4 * 17 * 2048);
        assert_eq!(context_partial_elements(32, 128, 128).unwrap(), 0);
        assert!(context_partial_elements(usize::MAX, 2048, 16384).is_err());
        assert!(context_partial_elements(32, usize::MAX, 128).is_err());
        assert!(context_partial_elements(32, 128, usize::MAX).is_err());
    }

    #[test]
    fn tiny_context_grows_only_the_packed_key_root_and_charges_it_once() {
        let device = MlxDevice::new().unwrap();
        let mut config =
            DFlash2Config::from_json_str(super::super::config::tests::QWEN38_CONFIG).unwrap();
        config.hidden_size = 128;
        config.intermediate_size = 128;
        config.num_hidden_layers = 1;
        config.num_attention_heads = 1;
        config.num_key_value_heads = 1;
        config.head_dim = 128;
        config.vocab_size = 16;
        config.sliding_window = 1;
        config.target_layer_ids = vec![0];
        let baseline = DFlash2Workspace::new(&device, &config, 4, false).unwrap();
        let baseline_plan = DFlash2Workspace::required_memory(&config, 4, false).unwrap();
        assert_eq!(baseline_plan.allocated_bytes, baseline.allocated_bytes());
        config.target_layer_ids = (0..32).collect();
        let mut grown = DFlash2Workspace::new(&device, &config, 4, false).unwrap();
        let grown_plan = DFlash2Workspace::required_memory(&config, 4, false).unwrap();
        assert_eq!(grown_plan.allocated_bytes, grown.allocated_bytes());
        assert!(grown_plan.ordinary_bytes >= grown.packed_key.byte_len());
        let packed_bytes = 4 * 128 * 8 * DType::BF16.size_of();
        let partial_bytes =
            context_partial_elements(32, 128, 4096).unwrap() * DType::F32.size_of();
        assert!(partial_bytes > packed_bytes);
        assert_eq!(grown.packed_key.shape(), &[4, 1, 8, 128]);
        assert_eq!(grown.packed_key.data_byte_len(), packed_bytes);
        assert_eq!(grown.context_partials.data_byte_len(), partial_bytes);
        assert_eq!(grown.packed_key.byte_len(), partial_bytes);
        assert_eq!(grown.context_partials.byte_len(), partial_bytes);
        let input_growth = grown.captured_target_hidden.data_byte_len()
            - baseline.captured_target_hidden.data_byte_len()
            + grown.af4_table.data_byte_len()
            - baseline.af4_table.data_byte_len()
            + grown.af4_sums.data_byte_len()
            - baseline.af4_sums.data_byte_len();
        assert_eq!(
            grown.allocated_bytes() - baseline.allocated_bytes(),
            input_growth + partial_bytes - packed_bytes
        );
        grown.packed_key.as_mut_slice::<u16>().unwrap()[..2]
            .copy_from_slice(&[0x3f80, 0x3f80]);
        assert_eq!(
            grown.context_partials.as_slice::<f32>().unwrap()[0].to_bits(),
            0x3f803f80
        );
    }

    #[test]
    fn dense_subtile_workspace_keeps_rectangular_attention_and_allocation_plan() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().unwrap();
        let mut config =
            DFlash2Config::from_json_str(super::super::config::tests::QWEN38_CONFIG).unwrap();
        config.hidden_size = 32;
        config.intermediate_size = 128;
        config.num_hidden_layers = 1;
        config.layer_types.truncate(1);
        config.num_attention_heads = 1;
        config.num_key_value_heads = 1;
        config.head_dim = 128;
        config.vocab_size = 16;
        config.sliding_window = 1;
        config.target_layer_ids = vec![0];
        let plan = DFlash2Workspace::required_memory(&config, 4, false).unwrap();
        let workspace = DFlash2Workspace::new(&device, &config, 4, false).unwrap();
        assert_eq!(workspace.query.element_count(), 4 * config.block_size * 128);
        assert_eq!(workspace.captured_target_hidden.element_count(), 4 * config.block_size * 32);
        assert_eq!(plan.allocated_bytes, workspace.allocated_bytes());
    }
}
