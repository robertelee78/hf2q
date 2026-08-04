//! DeepSeek-V4 circular and compressed KV cache planning.

use mlx_native::{DType, MlxBuffer, MlxDevice, MlxError};
use thiserror::Error;

use super::Deepseek4Config;

const REQUIRED_WINDOW: u32 = 128;
const CACHE_DTYPE: DType = DType::F32;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CacheBufferPlan {
    pub shape: Vec<usize>,
    pub bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerCachePlan {
    pub layer_index: usize,
    pub compress_ratio: u32,
    pub window_kv: CacheBufferPlan,
    pub compressed_kv: Option<CacheBufferPlan>,
    pub indexer_kv: Option<CacheBufferPlan>,
    pub resident_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Deepseek4CachePlan {
    pub context_length: usize,
    pub layers: Vec<LayerCachePlan>,
    pub resident_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheKind {
    WindowKv,
    CompressedKv,
    IndexerKv,
}

#[derive(Debug, Error)]
pub enum CacheError {
    #[error("DeepSeek-V4 cache context must be greater than zero")]
    EmptyContext,
    #[error("requested cache context {requested} exceeds model bound {maximum}")]
    ContextBound { requested: usize, maximum: usize },
    #[error("DeepSeek-V4 cache requires a 128-token circular window, got {actual}")]
    SlidingWindow { actual: u32 },
    #[error("compression schedule has {actual} layers, expected {expected}")]
    LayerCount { expected: usize, actual: usize },
    #[error("layer {layer} has unsupported compression ratio {ratio}")]
    CompressionRatio { layer: usize, ratio: u32 },
    #[error("cache byte accounting overflowed at layer {layer} {kind:?}")]
    ByteOverflow { layer: usize, kind: CacheKind },
    #[error("layer {layer} {kind:?} cache needs {bytes} bytes, beyond this host's address space")]
    AddressSpace {
        layer: usize,
        kind: CacheKind,
        bytes: u64,
    },
    #[error("failed to allocate layer {layer} {kind:?} cache: {source}")]
    Allocate {
        layer: usize,
        kind: CacheKind,
        #[source]
        source: MlxError,
    },
    #[error("allocated cache byte accounting mismatch: planned {planned}, actual {actual}")]
    Accounting { planned: u64, actual: u64 },
}

pub struct LayerCache {
    pub window_kv: MlxBuffer,
    pub compressed_kv: Option<MlxBuffer>,
    pub indexer_kv: Option<MlxBuffer>,
}

pub struct Deepseek4Cache {
    layers: Vec<LayerCache>,
    resident_bytes: u64,
    _device: MlxDevice,
}

impl Deepseek4CachePlan {
    pub fn for_context(cfg: &Deepseek4Config, context_length: usize) -> Result<Self, CacheError> {
        validate_request(cfg, context_length)?;
        let mut layers = Vec::with_capacity(cfg.compress_ratios.len());
        let mut resident_bytes = 0_u64;
        for (layer, &ratio) in cfg.compress_ratios.iter().enumerate() {
            let window_capacity = context_length.min(cfg.sliding_window as usize);
            let window_kv = buffer_plan(
                layer,
                CacheKind::WindowKv,
                window_capacity,
                cfg.head_dim as usize,
            )?;
            let compressed_kv = if ratio == 0 {
                None
            } else {
                optional_buffer_plan(
                    layer,
                    CacheKind::CompressedKv,
                    context_length / ratio as usize,
                    cfg.head_dim as usize,
                )?
            };
            let indexer_kv = if ratio == 4 {
                optional_buffer_plan(
                    layer,
                    CacheKind::IndexerKv,
                    context_length / 4,
                    cfg.index_head_dim as usize,
                )?
            } else {
                None
            };
            let mut layer_bytes = window_kv.bytes;
            for (kind, plan) in [
                (CacheKind::CompressedKv, compressed_kv.as_ref()),
                (CacheKind::IndexerKv, indexer_kv.as_ref()),
            ] {
                if let Some(plan) = plan {
                    layer_bytes = layer_bytes
                        .checked_add(plan.bytes)
                        .ok_or(CacheError::ByteOverflow { layer, kind })?;
                }
            }
            resident_bytes =
                resident_bytes
                    .checked_add(layer_bytes)
                    .ok_or(CacheError::ByteOverflow {
                        layer,
                        kind: CacheKind::WindowKv,
                    })?;
            layers.push(LayerCachePlan {
                layer_index: layer,
                compress_ratio: ratio,
                window_kv,
                compressed_kv,
                indexer_kv,
                resident_bytes: layer_bytes,
            });
        }
        Ok(Self {
            context_length,
            layers,
            resident_bytes,
        })
    }
}

impl Deepseek4Cache {
    pub fn allocate(plan: &Deepseek4CachePlan, device: MlxDevice) -> Result<Self, CacheError> {
        let mut layers = Vec::with_capacity(plan.layers.len());
        let mut resident_bytes = 0_u64;
        for layer in &plan.layers {
            let window_kv = allocate_buffer(
                &device,
                layer.layer_index,
                CacheKind::WindowKv,
                &layer.window_kv,
            )?;
            let compressed_kv = layer
                .compressed_kv
                .as_ref()
                .map(|buffer| {
                    allocate_buffer(&device, layer.layer_index, CacheKind::CompressedKv, buffer)
                })
                .transpose()?;
            let indexer_kv = layer
                .indexer_kv
                .as_ref()
                .map(|buffer| {
                    allocate_buffer(&device, layer.layer_index, CacheKind::IndexerKv, buffer)
                })
                .transpose()?;
            resident_bytes = resident_bytes.checked_add(layer.resident_bytes).ok_or(
                CacheError::ByteOverflow {
                    layer: layer.layer_index,
                    kind: CacheKind::WindowKv,
                },
            )?;
            layers.push(LayerCache {
                window_kv,
                compressed_kv,
                indexer_kv,
            });
        }
        let actual = layers.iter().try_fold(0_u64, |total, layer| {
            std::iter::once(&layer.window_kv)
                .chain(layer.compressed_kv.iter())
                .chain(layer.indexer_kv.iter())
                .try_fold(total, |bytes, buffer| {
                    u64::try_from(buffer.byte_len())
                        .ok()
                        .and_then(|buffer_bytes| bytes.checked_add(buffer_bytes))
                })
        });
        let actual = actual.ok_or(CacheError::ByteOverflow {
            layer: layers.len().saturating_sub(1),
            kind: CacheKind::WindowKv,
        })?;
        if actual != plan.resident_bytes || resident_bytes != plan.resident_bytes {
            return Err(CacheError::Accounting {
                planned: plan.resident_bytes,
                actual,
            });
        }
        Ok(Self {
            layers,
            resident_bytes: actual,
            _device: device,
        })
    }

    pub fn layers(&self) -> &[LayerCache] {
        &self.layers
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

fn validate_request(cfg: &Deepseek4Config, context_length: usize) -> Result<(), CacheError> {
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

fn optional_buffer_plan(
    layer: usize,
    kind: CacheKind,
    capacity: usize,
    width: usize,
) -> Result<Option<CacheBufferPlan>, CacheError> {
    (capacity > 0)
        .then(|| buffer_plan(layer, kind, capacity, width))
        .transpose()
}

fn buffer_plan(
    layer: usize,
    kind: CacheKind,
    capacity: usize,
    width: usize,
) -> Result<CacheBufferPlan, CacheError> {
    let elements = u64::try_from(capacity)
        .ok()
        .and_then(|capacity| {
            u64::try_from(width)
                .ok()
                .and_then(|width| capacity.checked_mul(width))
        })
        .ok_or(CacheError::ByteOverflow { layer, kind })?;
    let bytes = elements
        .checked_mul(CACHE_DTYPE.size_of() as u64)
        .ok_or(CacheError::ByteOverflow { layer, kind })?;
    Ok(CacheBufferPlan {
        shape: vec![capacity, width],
        bytes,
    })
}

fn allocate_buffer(
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
        .alloc_buffer(bytes, CACHE_DTYPE, plan.shape.clone())
        .map_err(|source| CacheError::Allocate {
            layer,
            kind,
            source,
        })
}
