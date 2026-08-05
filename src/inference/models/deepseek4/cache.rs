//! DeepSeek-V4 circular and compressed KV cache planning.

use mlx_native::{DType, MlxBuffer, MlxDevice, MlxError};
use thiserror::Error;

use super::cache_buffers::{
    allocate_buffer, allocate_optional, buffer_plan, completed_group_step, compressor_state_plans,
    fill_state, optional_buffer_plan, validate_request, view_buffer,
};
use super::Deepseek4Config;

mod prefill;
pub use prefill::{CacheSpan, LayerCacheSpan};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CacheBufferPlan {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerCachePlan {
    pub layer_index: usize,
    pub compress_ratio: u32,
    /// Physical BF16 allocation addressed as window rows followed by
    /// compressed rows. The two fields below are zero-copy views.
    pub attention_kv: CacheBufferPlan,
    pub window_kv: CacheBufferPlan,
    pub compressed_kv: Option<CacheBufferPlan>,
    pub indexer_kv: Option<CacheBufferPlan>,
    pub main_kv_state: Option<CacheBufferPlan>,
    pub main_score_state: Option<CacheBufferPlan>,
    pub indexer_kv_state: Option<CacheBufferPlan>,
    pub indexer_score_state: Option<CacheBufferPlan>,
    pub resident_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Deepseek4CachePlan {
    pub context_length: usize,
    pub layers: Vec<LayerCachePlan>,
    pub resident_bytes: u64,
}

/// Exact cache slots and visibility bounds for one autoregressive position.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerCacheStep {
    pub layer_index: usize,
    pub window_write_slot: usize,
    pub window_start_position: usize,
    pub window_valid_after: usize,
    pub compressed_write_slot: Option<usize>,
    pub compressed_valid_after: usize,
    pub indexer_write_slot: Option<usize>,
    pub indexer_valid_after: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CacheStep {
    pub position: usize,
    pub layers: Vec<LayerCacheStep>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheKind {
    AttentionKv,
    WindowKv,
    CompressedKv,
    IndexerKv,
    MainKvState,
    MainScoreState,
    IndexerKvState,
    IndexerScoreState,
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
    #[error("failed to create layer {layer} {kind:?} cache view: {source}")]
    View {
        layer: usize,
        kind: CacheKind,
        #[source]
        source: MlxError,
    },
    #[error("failed to initialize layer {layer} {kind:?} cache state: {source}")]
    Initialize {
        layer: usize,
        kind: CacheKind,
        #[source]
        source: MlxError,
    },
    #[error("allocated cache byte accounting mismatch: planned {planned}, actual {actual}")]
    Accounting { planned: u64, actual: u64 },
    #[error("DeepSeek-V4 cache is full at context bound {maximum}")]
    ContextExhausted { maximum: usize },
    #[error("cache step committed out of order: expected position {expected}, got {actual}")]
    StepOutOfOrder { expected: usize, actual: usize },
    #[error("DeepSeek-V4 prefill requires at least one token")]
    EmptyPrefill,
    #[error("DeepSeek-V4 start-zero prefill requires an empty cache, currently at {position}")]
    PrefillNotEmpty { position: usize },
    #[error("DeepSeek-V4 prefill length {requested} exceeds the {maximum}-token native window")]
    PrefillWindow { requested: usize, maximum: usize },
    #[error("DeepSeek-V4 cache is poisoned by a partial token; reset and replay the request")]
    Poisoned,
}

pub struct LayerCache {
    /// Owns the allocation shared by `window_kv` and `compressed_kv`.
    pub attention_kv: MlxBuffer,
    pub window_kv: MlxBuffer,
    pub compressed_kv: Option<MlxBuffer>,
    pub indexer_kv: Option<MlxBuffer>,
    pub main_kv_state: Option<MlxBuffer>,
    pub main_score_state: Option<MlxBuffer>,
    pub indexer_kv_state: Option<MlxBuffer>,
    pub indexer_score_state: Option<MlxBuffer>,
}

pub struct Deepseek4Cache {
    layers: Vec<LayerCache>,
    pub(super) plan: Deepseek4CachePlan,
    pub(super) next_position: usize,
    poisoned: bool,
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
                vec![window_capacity, cfg.head_dim as usize],
                DType::BF16,
            )?;
            let compressed_kv = if ratio == 0 {
                None
            } else {
                optional_buffer_plan(
                    layer,
                    CacheKind::CompressedKv,
                    vec![context_length / ratio as usize, cfg.head_dim as usize],
                    DType::BF16,
                )?
            };
            let compressed_capacity = compressed_kv.as_ref().map_or(0, |plan| plan.shape[0]);
            let attention_kv = buffer_plan(
                layer,
                CacheKind::AttentionKv,
                vec![window_capacity + compressed_capacity, cfg.head_dim as usize],
                DType::BF16,
            )?;
            let indexer_kv = if ratio == 4 {
                optional_buffer_plan(
                    layer,
                    CacheKind::IndexerKv,
                    vec![context_length / 4, cfg.index_head_dim as usize],
                    DType::BF16,
                )?
            } else {
                None
            };
            let (main_kv_state, main_score_state) = compressor_state_plans(
                layer,
                ratio,
                cfg.head_dim as usize,
                CacheKind::MainKvState,
                CacheKind::MainScoreState,
            )?;
            let (indexer_kv_state, indexer_score_state) = if ratio == 4 {
                compressor_state_plans(
                    layer,
                    ratio,
                    cfg.index_head_dim as usize,
                    CacheKind::IndexerKvState,
                    CacheKind::IndexerScoreState,
                )?
            } else {
                (None, None)
            };
            let mut layer_bytes = attention_kv.bytes;
            for (kind, plan) in [
                (CacheKind::IndexerKv, indexer_kv.as_ref()),
                (CacheKind::MainKvState, main_kv_state.as_ref()),
                (CacheKind::MainScoreState, main_score_state.as_ref()),
                (CacheKind::IndexerKvState, indexer_kv_state.as_ref()),
                (CacheKind::IndexerScoreState, indexer_score_state.as_ref()),
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
                        kind: CacheKind::AttentionKv,
                    })?;
            layers.push(LayerCachePlan {
                layer_index: layer,
                compress_ratio: ratio,
                attention_kv,
                window_kv,
                compressed_kv,
                indexer_kv,
                main_kv_state,
                main_score_state,
                indexer_kv_state,
                indexer_score_state,
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
            let attention_kv = allocate_buffer(
                &device,
                layer.layer_index,
                CacheKind::AttentionKv,
                &layer.attention_kv,
            )?;
            let window_kv = view_buffer(
                &attention_kv,
                0,
                layer.layer_index,
                CacheKind::WindowKv,
                &layer.window_kv,
            )?;
            let compressed_kv = layer
                .compressed_kv
                .as_ref()
                .map(|buffer| {
                    view_buffer(
                        &attention_kv,
                        layer.window_kv.bytes,
                        layer.layer_index,
                        CacheKind::CompressedKv,
                        buffer,
                    )
                })
                .transpose()?;
            let indexer_kv = layer
                .indexer_kv
                .as_ref()
                .map(|buffer| {
                    allocate_buffer(&device, layer.layer_index, CacheKind::IndexerKv, buffer)
                })
                .transpose()?;
            let main_kv_state = allocate_optional(
                &device,
                layer.layer_index,
                CacheKind::MainKvState,
                layer.main_kv_state.as_ref(),
            )?;
            let main_score_state = allocate_optional(
                &device,
                layer.layer_index,
                CacheKind::MainScoreState,
                layer.main_score_state.as_ref(),
            )?;
            let indexer_kv_state = allocate_optional(
                &device,
                layer.layer_index,
                CacheKind::IndexerKvState,
                layer.indexer_kv_state.as_ref(),
            )?;
            let indexer_score_state = allocate_optional(
                &device,
                layer.layer_index,
                CacheKind::IndexerScoreState,
                layer.indexer_score_state.as_ref(),
            )?;
            resident_bytes = resident_bytes.checked_add(layer.resident_bytes).ok_or(
                CacheError::ByteOverflow {
                    layer: layer.layer_index,
                    kind: CacheKind::WindowKv,
                },
            )?;
            layers.push(LayerCache {
                attention_kv,
                window_kv,
                compressed_kv,
                indexer_kv,
                main_kv_state,
                main_score_state,
                indexer_kv_state,
                indexer_score_state,
            });
        }
        let actual = layers.iter().try_fold(0_u64, |total, layer| {
            std::iter::once(&layer.attention_kv)
                .chain(layer.indexer_kv.iter())
                .chain(layer.main_kv_state.iter())
                .chain(layer.main_score_state.iter())
                .chain(layer.indexer_kv_state.iter())
                .chain(layer.indexer_score_state.iter())
                .try_fold(total, |bytes, buffer| {
                    u64::try_from(buffer.byte_len())
                        .ok()
                        .and_then(|buffer_bytes| bytes.checked_add(buffer_bytes))
                })
        });
        let actual = actual.ok_or(CacheError::ByteOverflow {
            layer: layers.len().saturating_sub(1),
            kind: CacheKind::AttentionKv,
        })?;
        if actual != plan.resident_bytes || resident_bytes != plan.resident_bytes {
            return Err(CacheError::Accounting {
                planned: plan.resident_bytes,
                actual,
            });
        }
        let mut cache = Self {
            layers,
            plan: plan.clone(),
            next_position: 0,
            poisoned: false,
            resident_bytes: actual,
            _device: device,
        };
        cache.reset()?;
        Ok(cache)
    }

    pub fn layers(&self) -> &[LayerCache] {
        &self.layers
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn position(&self) -> usize {
        self.next_position
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Prevent retries after any submitted layer has mutated recurrent state.
    /// Resetting the request clears the poison and recurrent state together.
    pub(super) fn poison(&mut self) {
        self.poisoned = true;
    }

    /// Return the write slots and post-write visibility bounds for the next
    /// token without changing logical cache state. The caller commits only
    /// after its Metal command buffer completes successfully.
    pub fn plan_next_step(&self) -> Result<CacheStep, CacheError> {
        if self.poisoned {
            return Err(CacheError::Poisoned);
        }
        if self.next_position >= self.plan.context_length {
            return Err(CacheError::ContextExhausted {
                maximum: self.plan.context_length,
            });
        }
        let position = self.next_position;
        let tokens_after = position + 1;
        let layers = self
            .plan
            .layers
            .iter()
            .map(|layer| {
                let window_capacity = layer.window_kv.shape[0];
                let window_valid_after = tokens_after.min(window_capacity);
                let (compressed_write_slot, compressed_valid_after) =
                    completed_group_step(tokens_after, layer.compress_ratio);
                let indexer_write_slot = (layer.compress_ratio == 4)
                    .then_some(compressed_write_slot)
                    .flatten();
                let indexer_valid_after = if layer.compress_ratio == 4 {
                    compressed_valid_after
                } else {
                    0
                };
                LayerCacheStep {
                    layer_index: layer.layer_index,
                    window_write_slot: position % window_capacity,
                    window_start_position: tokens_after - window_valid_after,
                    window_valid_after,
                    compressed_write_slot,
                    compressed_valid_after,
                    indexer_write_slot,
                    indexer_valid_after,
                }
            })
            .collect();
        Ok(CacheStep { position, layers })
    }

    /// Advance logical visibility after the planned GPU writes complete.
    pub fn commit_step(&mut self, position: usize) -> Result<(), CacheError> {
        if self.poisoned {
            return Err(CacheError::Poisoned);
        }
        if position != self.next_position {
            return Err(CacheError::StepOutOfOrder {
                expected: self.next_position,
                actual: position,
            });
        }
        if self.next_position >= self.plan.context_length {
            return Err(CacheError::ContextExhausted {
                maximum: self.plan.context_length,
            });
        }
        self.next_position += 1;
        Ok(())
    }

    /// Reset logical visibility and all recurrent compressor state. KV rows
    /// remain validity-bounded, but pooling state participates in future
    /// writes and therefore must be restored exactly between requests.
    pub fn reset(&mut self) -> Result<(), CacheError> {
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            fill_state(
                layer.main_kv_state.as_mut(),
                0.0,
                layer_index,
                CacheKind::MainKvState,
            )?;
            fill_state(
                layer.main_score_state.as_mut(),
                f32::NEG_INFINITY,
                layer_index,
                CacheKind::MainScoreState,
            )?;
            fill_state(
                layer.indexer_kv_state.as_mut(),
                0.0,
                layer_index,
                CacheKind::IndexerKvState,
            )?;
            fill_state(
                layer.indexer_score_state.as_mut(),
                f32::NEG_INFINITY,
                layer_index,
                CacheKind::IndexerScoreState,
            )?;
        }
        self.next_position = 0;
        self.poisoned = false;
        Ok(())
    }
}
