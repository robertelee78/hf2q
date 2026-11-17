//! Checked auxiliary scratch geometry, shared by planning and allocation.
//! Context FC/KV and QKV partials belong to the workspace phase alias, not here.

use super::{DFlash2Config, DFlash2StagedScratch, DFlash2WorkspaceError, STAGED_TILE_ROWS};
use super::super::workspace::{add_bytes, buffer_bytes, WorkspaceGeometry};
use mlx_native::{DType, ops::{
    ActivationPrecision,
    dflash2_af4g64_staged::{
        dflash2_af4g64_decode_uses_register,
        dflash2_af4g64_staged_gate_up_partial_elements,
        dflash2_af4g64_staged_partial_elements_with_precision,
    },
    dflash2_ring_attention::{
        dflash2_ring_attention_partial_elements, dflash2_ring_attention_supports,
        DFLASH2_RING_ATTENTION_CURRENT_VALUES,
    },
}};

pub(super) struct StagedGeometry {
    pub rows: usize,
    pub query: usize,
    pub widest_input: usize,
    pub partial_elements: usize,
    pub attention_partial_elements: Option<usize>,
}

impl StagedGeometry {
    pub(super) fn new(
        config: &DFlash2Config,
        lanes: usize,
        workspace: &WorkspaceGeometry,
    ) -> Result<Self, DFlash2WorkspaceError> {
        let rows = workspace.rows;
        if rows % STAGED_TILE_ROWS != 0
            || config.hidden_size % 64 != 0
            || config.intermediate_size % 64 != 0
        {
            return Err(DFlash2WorkspaceError::Invalid("unsupported staged workspace geometry".into()));
        }
        // Bound native row-indexed products before invoking the split-size API.
        // FC/context/QKV never enter this auxiliary reservation.
        for width in [workspace.dynamic, workspace.query, config.hidden_size, config.intermediate_size] {
            if width < 64 || width % 64 != 0
                || rows.checked_mul(width).and_then(|n| u32::try_from(n).ok()).is_none()
            {
                return Err(DFlash2WorkspaceError::Invalid("staged projection extent exceeds native indexing".into()));
            }
        }
        let native_rows = rows as u32;
        let native_hidden = config.hidden_size as u32;
        let mut partial_elements = 1;
        for (n, k) in [
            (workspace.dynamic, config.hidden_size),
            (config.hidden_size, workspace.query),
            (config.hidden_size, config.intermediate_size),
        ] {
            if !dflash2_af4g64_decode_uses_register(native_rows, n as u32) {
                partial_elements = partial_elements.max(
                    dflash2_af4g64_staged_partial_elements_with_precision(
                        native_rows, n as u32, k as u32, ActivationPrecision::Bf16,
                    ),
                );
            }
        }
        partial_elements = partial_elements.max(dflash2_af4g64_staged_gate_up_partial_elements(
            native_rows, config.intermediate_size as u32, native_hidden,
        ));
        let attention_partial_elements = if dflash2_ring_attention_supports(
            config.num_attention_heads, config.num_key_value_heads, config.head_dim,
            config.block_size, config.sliding_window,
        ) {
            Some(dflash2_ring_attention_partial_elements(lanes, config.num_key_value_heads)?)
        } else { None };
        Ok(Self {
            rows,
            query: workspace.query,
            widest_input: config.hidden_size.max(workspace.query).max(config.intermediate_size),
            partial_elements,
            attention_partial_elements,
        })
    }
}

impl DFlash2StagedScratch {
    /// Requested initial roots. Lazy/pending prepared heads are separate owners.
    pub(crate) fn required_bytes(
        config: &DFlash2Config,
        lanes: usize,
        workspace: &WorkspaceGeometry,
    ) -> Result<(usize, usize), DFlash2WorkspaceError> {
        let geometry = StagedGeometry::new(config, lanes, workspace)?;
        let rows = geometry.rows;
        let mut projections = buffer_bytes(DType::BF16, &[rows, geometry.widest_input])?;
        add_bytes(&mut projections, buffer_bytes(DType::BF16, &[rows, config.intermediate_size])?)?;
        add_bytes(&mut projections, buffer_bytes(DType::F32, &[geometry.partial_elements])?)?;
        let mut attention = 0;
        if let Some(partials) = geometry.attention_partial_elements {
            for (dtype, shape) in [
                (DType::BF16, &[lanes, config.num_attention_heads, config.block_size, config.head_dim][..]),
                (DType::BF16, &[lanes, config.num_key_value_heads, config.block_size, config.head_dim]),
                (DType::BF16, &[lanes, config.num_key_value_heads, DFLASH2_RING_ATTENTION_CURRENT_VALUES, config.head_dim]),
                (DType::F32, &[partials]),
                (DType::BF16, &[rows, geometry.query]),
            ] {
                add_bytes(&mut attention, buffer_bytes(dtype, shape)?)?;
            }
        }
        Ok((projections, attention))
    }
}
