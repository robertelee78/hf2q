//! Typed DeepSeek-V4 cache buffer planning, allocation, and zero-copy views.

use mlx_native::{DType, MlxBuffer, MlxDevice};

use super::cache::{CacheBufferPlan, CacheError, CacheKind};
use super::Deepseek4Config;

const REQUIRED_WINDOW: u32 = 128;

pub(super) fn completed_group_step(tokens_after: usize, ratio: u32) -> (Option<usize>, usize) {
    if ratio == 0 {
        return (None, 0);
    }
    let ratio = ratio as usize;
    let completed = tokens_after / ratio;
    let write = (tokens_after % ratio == 0).then(|| completed - 1);
    (write, completed)
}

pub(super) fn validate_request(
    cfg: &Deepseek4Config,
    context_length: usize,
) -> Result<(), CacheError> {
    if context_length == 0 {
        return Err(CacheError::EmptyContext);
    }
    let maximum = cfg.max_position_embeddings as usize;
    if context_length > maximum {
        return Err(CacheError::ContextBound {
            requested: context_length,
            maximum,
        });
    }
    if cfg.sliding_window != REQUIRED_WINDOW {
        return Err(CacheError::SlidingWindow {
            actual: cfg.sliding_window,
        });
    }
    let expected = cfg.num_hidden_layers as usize;
    if cfg.compress_ratios.len() != expected {
        return Err(CacheError::LayerCount {
            expected,
            actual: cfg.compress_ratios.len(),
        });
    }
    if let Some((layer, &ratio)) = cfg
        .compress_ratios
        .iter()
        .enumerate()
        .find(|(_, ratio)| !matches!(ratio, 0 | 4 | 128))
    {
        return Err(CacheError::CompressionRatio { layer, ratio });
    }
    Ok(())
}

pub(super) fn optional_buffer_plan(
    layer: usize,
    kind: CacheKind,
    shape: Vec<usize>,
    dtype: DType,
) -> Result<Option<CacheBufferPlan>, CacheError> {
    shape
        .iter()
        .all(|dimension| *dimension > 0)
        .then(|| buffer_plan(layer, kind, shape, dtype))
        .transpose()
}

pub(super) fn buffer_plan(
    layer: usize,
    kind: CacheKind,
    shape: Vec<usize>,
    dtype: DType,
) -> Result<CacheBufferPlan, CacheError> {
    let elements = shape.iter().try_fold(1_u64, |elements, &dimension| {
        u64::try_from(dimension)
            .ok()
            .and_then(|dimension| elements.checked_mul(dimension))
            .ok_or(CacheError::ByteOverflow { layer, kind })
    })?;
    let bytes = elements
        .checked_mul(dtype.size_of() as u64)
        .ok_or(CacheError::ByteOverflow { layer, kind })?;
    Ok(CacheBufferPlan {
        shape,
        dtype,
        bytes,
    })
}

pub(super) fn allocate_buffer(
    device: &MlxDevice,
    layer: usize,
    kind: CacheKind,
    plan: &CacheBufferPlan,
) -> Result<MlxBuffer, CacheError> {
    let bytes = usize::try_from(plan.bytes).map_err(|_| CacheError::AddressSpace {
        layer,
        kind,
        bytes: plan.bytes,
    })?;
    device
        .alloc_buffer(bytes, plan.dtype, plan.shape.clone())
        .map_err(|source| CacheError::Allocate {
            layer,
            kind,
            source,
        })
}

pub(super) fn allocate_optional(
    device: &MlxDevice,
    layer: usize,
    kind: CacheKind,
    plan: Option<&CacheBufferPlan>,
) -> Result<Option<MlxBuffer>, CacheError> {
    plan.map(|plan| allocate_buffer(device, layer, kind, plan))
        .transpose()
}

pub(super) fn compressor_state_plans(
    layer: usize,
    ratio: u32,
    head_dim: usize,
    kv_kind: CacheKind,
    score_kind: CacheKind,
) -> Result<(Option<CacheBufferPlan>, Option<CacheBufferPlan>), CacheError> {
    if ratio == 0 {
        return Ok((None, None));
    }
    let coefficient = usize::from(ratio == 4) + 1;
    let shape = vec![1, coefficient * ratio as usize, coefficient * head_dim];
    Ok((
        Some(buffer_plan(layer, kv_kind, shape.clone(), DType::F32)?),
        Some(buffer_plan(layer, score_kind, shape, DType::F32)?),
    ))
}

pub(super) fn fill_state(
    buffer: Option<&mut MlxBuffer>,
    value: f32,
    layer: usize,
    kind: CacheKind,
) -> Result<(), CacheError> {
    if let Some(buffer) = buffer {
        buffer
            .as_mut_slice::<f32>()
            .map_err(|source| CacheError::Initialize {
                layer,
                kind,
                source,
            })?
            .fill(value);
    }
    Ok(())
}

pub(super) fn view_buffer(
    parent: &MlxBuffer,
    byte_offset: u64,
    layer: usize,
    kind: CacheKind,
    plan: &CacheBufferPlan,
) -> Result<MlxBuffer, CacheError> {
    let elements = plan.shape.iter().product();
    parent
        .slice_view(byte_offset, elements)
        .with_shape(plan.shape.clone())
        .map_err(|source| CacheError::View {
            layer,
            kind,
            source,
        })
}
