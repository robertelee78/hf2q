//! Requested workspace roots, shared with construction; not physical admission.

use super::super::config;

use super::{context_partial_elements, elements, DFlash2Config, DFlash2StagedScratch, DFlash2Workspace, DFlash2WorkspaceError, DFLASH2_MAX_LANES};
use mlx_native::{DType, ops::{ActivationPrecision, dflash2_af4g64_staged::{dflash2_af4g64_decode_uses_register, dflash2_af4g64_staged_splits_with_precision}}};

/// Device-buffer payloads owned by the draft workspace, before allocation.
///
/// Prepared vocabulary heads (including a pending repack) belong to the target
/// head policy, not this owner. Checkpoint weights, target capture/history and
/// native/Metal allocation overhead are also outside this payload accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DFlash2WorkspaceMemory {
    pub ordinary_bytes: usize,
    pub ring_bytes: usize,
    pub staged_projection_bytes: usize,
    pub staged_ring_attention_bytes: usize,
    /// Initial ordinary, ring and staged allocations; equals `allocated_bytes()`.
    pub allocated_bytes: usize,
    /// Two position-buffer copies retained by detached context commit.
    /// Separate runtime owner; NOT included in `allocated_bytes`.
    pub detached_context_positions_bytes: usize,
}

pub(crate) fn buffer_bytes(
    dtype: DType,
    shape: &[usize],
) -> Result<usize, DFlash2WorkspaceError> {
    elements(shape)?
        .checked_mul(dtype.size_of())
        .ok_or_else(|| DFlash2WorkspaceError::Invalid(format!("byte size overflows: {shape:?}")))
}

pub(crate) fn add_bytes(total: &mut usize, bytes: usize) -> Result<(), DFlash2WorkspaceError> {
    *total = total
        .checked_add(bytes)
        .ok_or_else(|| DFlash2WorkspaceError::Invalid("workspace total overflows".into()))?;
    Ok(())
}

/// Checked dimensions shared by allocation and preallocation accounting.
pub(crate) struct WorkspaceGeometry {
    pub rows: usize,
    pub query: usize,
    pub key_value: usize,
    pub qkv: usize,
    pub gate_up: usize,
    pub packed: usize,
    pub fc_input: usize,
    pub dynamic: usize,
    pub widest_input: usize,
    pub packed_key_elements: usize,
    pub context_partials_bytes: usize,
    pub packed_key_root_bytes: usize,
}

impl WorkspaceGeometry {
    pub(crate) fn new(
        config: &DFlash2Config,
        lanes: usize,
    ) -> Result<Self, DFlash2WorkspaceError> {
        let invalid = |message: &str| DFlash2WorkspaceError::Invalid(message.into());
        if !(1..=DFLASH2_MAX_LANES).contains(&lanes) {
            return Err(invalid("max_lanes must be in 1..=4"));
        }
        if !(2..=config::MAX_DFLASH2_BLOCK_ROWS).contains(&config.block_size)
            || config.hidden_size == 0
            || config.intermediate_size == 0
            || config.num_hidden_layers == 0
            || config.num_attention_heads == 0
            || config.num_key_value_heads == 0
            || config.head_dim == 0
            || config.vocab_size == 0
            || config.sliding_window == 0
            || config.target_layer_ids.is_empty()
            || config.conv_group_size == 0
            || config.conv_kernel_size != 2
            || config.hidden_size % config.conv_group_size != 0
            || config.num_attention_heads % config.num_key_value_heads != 0
            || config.selector_rank != config::DFLASH2_SELECTOR_RANK
            || config.selector_top_k != config::DFLASH2_SELECTOR_TOP_K
        {
            return Err(invalid("unsupported workspace geometry"));
        }
        let rows = elements(&[lanes, config.block_size])?;
        let query = elements(&[config.num_attention_heads, config.head_dim])?;
        let key_value = elements(&[config.num_key_value_heads, config.head_dim])?;
        let twice_kv = elements(&[2, key_value])?;
        let qkv = query
            .checked_add(twice_kv)
            .ok_or_else(|| invalid("QKV width overflows"))?;
        let gate_up = elements(&[2, config.intermediate_size])?;
        let packed = (config.sliding_window - 1)
            .checked_add(config.block_size)
            .ok_or_else(|| invalid("packed window overflows"))?;
        let fc_input = elements(&[config.hidden_size, config.target_layer_ids.len()])?;
        let dynamic = elements(&[
            config::DFLASH2_CONVOLUTION_STAGES,
            config.conv_kernel_size,
            config.hidden_size / config.conv_group_size,
        ])?;
        // Command parameters and native projection geometry use u32.
        for width in [
            rows, query, key_value, qkv, gate_up, packed, fc_input, dynamic,
            config.hidden_size, config.intermediate_size, config.vocab_size,
            config.sliding_window, config.head_dim,
            elements(&[rows, config.intermediate_size])?,
            elements(&[rows, config.vocab_size])?,
        ] {
            u32::try_from(width).map_err(|_| invalid("workspace geometry exceeds u32"))?;
        }
        let packed_key_elements = elements(&[lanes, config.num_key_value_heads, packed, config.head_dim])?;
        let packed_key_bytes = buffer_bytes(DType::BF16, &[packed_key_elements])?;
        let native_rows = rows as u32;
        let native_qkv = qkv as u32;
        let qkv_partials = if config.hidden_size >= 64 && rows % 8 == 0
            && !dflash2_af4g64_decode_uses_register(native_rows, native_qkv)
        {
            let splits = dflash2_af4g64_staged_splits_with_precision(
                native_qkv, config.hidden_size as u32, ActivationPrecision::Bf16,
            );
            if splits == 1 { 0 } else { elements(&[splits as usize, rows, qkv])? }
        } else { 0 };
        let context_partials_elements = context_partial_elements(rows, config.hidden_size, fc_input)?
            .max(context_partial_elements(rows, twice_kv, config.hidden_size)?)
            .max(qkv_partials);
        let context_partials_bytes = buffer_bytes(DType::F32, &[context_partials_elements])?;
        let packed_key_root_bytes = packed_key_bytes.max(context_partials_bytes);
        Ok(Self {
            rows,
            query,
            key_value,
            qkv,
            gate_up,
            packed,
            fc_input,
            dynamic,
            widest_input: fc_input.max(config.hidden_size).max(query).max(config.intermediate_size),
            packed_key_elements,
            context_partials_bytes,
            packed_key_root_bytes,
        })
    }
}

impl DFlash2Workspace {
    /// Size the exact selected route without a device or checkpoint upload.
    ///
    /// `staged` is the already-resolved staged capability, not an environment
    /// default. BF16 and register-AF4 routes own the same ordinary buffers.
    pub fn required_memory(
        config: &DFlash2Config,
        max_lanes: usize,
        staged: bool,
    ) -> Result<DFlash2WorkspaceMemory, DFlash2WorkspaceError> {
        let g = WorkspaceGeometry::new(config, max_lanes)?;
        Self::memory_for_geometry(config, max_lanes, staged, &g)
    }

    fn memory_for_geometry(
        config: &DFlash2Config,
        max_lanes: usize,
        staged: bool,
        g: &WorkspaceGeometry,
    ) -> Result<DFlash2WorkspaceMemory, DFlash2WorkspaceError> {
        let lanes = max_lanes;
        let block = config.block_size;
        let positions = config.proposal_tokens();
        let rows = g.rows;
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let mut ordinary_bytes = 0;
        // One entry per ordinary root allocation in `new`, in allocation order.
        for (dtype, shape) in [
            (DType::U32, &[lanes, block][..]), // row_tokens
            (DType::F32, &[rows, g.fc_input]), // captured_target_hidden: only draft rows
            (DType::F32, &[rows, hidden]), // hidden[0]
            (DType::F32, &[rows, hidden]), // hidden[1]
            (DType::F32, &[rows, hidden]), // normalized
            (DType::BF16, &[rows.div_ceil(8), 8, g.widest_input]),
            (DType::F32, &[rows.div_ceil(8), g.widest_input / 64, 8]),
            (DType::F32, &[rows, g.dynamic]),
            (DType::F32, &[rows, hidden]), // convolved
            (DType::F32, &[rows, hidden]), // residual
            (DType::F32, &[rows, g.qkv]),
            (DType::F32, &[rows, g.query]), // query
            (DType::F32, &[rows, g.key_value]), // key
            (DType::F32, &[rows, g.key_value]), // value
            (DType::F32, &[rows, g.query]), // query_normalized
            (DType::F32, &[rows, g.key_value]), // key_normalized
            (DType::F32, &[rows, g.query]), // attention
            (DType::BF16, &[lanes, config.num_attention_heads, block, config.head_dim]),
            (DType::BF16, &[g.packed_key_root_bytes / DType::BF16.size_of()]), // one packed/context/QKV root
            (DType::BF16, &[lanes, config.num_key_value_heads, g.packed, config.head_dim]),
            (DType::BF16, &[lanes, config.num_attention_heads, block, config.head_dim]),
            (DType::BF16, &[lanes, config.num_attention_heads, block, g.packed]),
            (DType::U32, &[lanes]), // cache_lengths
            (DType::U32, &[lanes]), // write_positions
            (DType::U32, &[rows]), // rope_positions
            (DType::F32, &[4]), // rope_params
            (DType::F32, &[2]), // norm_params
            (DType::F32, &[2]), // head_norm_params
            (DType::U32, &[1]), // silu_params
            (DType::F32, &[2]), // softcap_params
            (DType::U32, &[4]), // head_meta
            (DType::F32, &[rows, hidden]), // projected
            (DType::F32, &[rows, g.gate_up]),
            (DType::F32, &[rows, intermediate]), // gate
            (DType::F32, &[rows, intermediate]), // up
            (DType::F32, &[rows, intermediate]), // intermediate
            (DType::F32, &[rows, hidden]), // final_hidden
            (DType::F32, &[lanes, block, config.vocab_size]),
            (DType::F32, &[lanes, block, config.selector_rank]),
            (DType::BF16, &[lanes, block, config.selector_rank]),
            (DType::F32, &[lanes, positions]), // uniforms
            (DType::U32, &[lanes, positions, 8, 16]), // partial_ids
            (DType::F32, &[lanes, positions, 8, 16]), // partial_values
            (DType::U32, &[lanes, positions, 16]), // candidates
            (DType::F32, &[lanes, positions, 16]), // unary
            (DType::F32, &[lanes, positions, 16, 16]), // edges
            (DType::U32, &[lanes, positions]), // proposed_tokens
            (DType::U32, &[DFLASH2_MAX_LANES * block]), // target_tokens
            (DType::F32, &[lanes, positions, 16]), // proposal_probabilities
            (DType::F32, &[lanes, positions]), // confidence
            (DType::U32, &[lanes]), // verify_rows
            (DType::U32, &[lanes]), // retained_counts
            (DType::U32, &[lanes]), // accepted_counts
        ] {
            add_bytes(&mut ordinary_bytes, buffer_bytes(dtype, shape)?)?;
        }
        let ring_bytes = elements(&[
            2, config.num_hidden_layers,
            buffer_bytes(DType::BF16, &[
                lanes, config.num_key_value_heads, config.sliding_window, config.head_dim,
            ])?,
        ])?;
        let (staged_projection_bytes, staged_ring_attention_bytes) = if staged {
            DFlash2StagedScratch::required_bytes(config, lanes, g)?
        } else {
            (0, 0)
        };
        let mut allocated_bytes = ordinary_bytes;
        for bytes in [ring_bytes, staged_projection_bytes, staged_ring_attention_bytes] {
            add_bytes(&mut allocated_bytes, bytes)?;
        }
        let mut detached_context_positions_bytes = buffer_bytes(DType::U32, &[rows])?;
        add_bytes(
            &mut detached_context_positions_bytes,
            buffer_bytes(DType::U32, &[lanes])?,
        )?;
        Ok(DFlash2WorkspaceMemory {
            ordinary_bytes,
            ring_bytes,
            staged_projection_bytes,
            staged_ring_attention_bytes,
            allocated_bytes,
            detached_context_positions_bytes,
        })
    }
}
