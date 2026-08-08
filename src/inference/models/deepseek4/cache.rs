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
    #[error("DeepSeek-V4 cache is poisoned by a partial token; reset and replay the request")]
    Poisoned,
    #[error("DeepSeek-V4 cache snapshot does not match the live cache plan")]
    SnapshotPlanMismatch,
    #[error("DeepSeek-V4 cache snapshot layer count {actual} does not match {expected}")]
    SnapshotLayerCount { expected: usize, actual: usize },
    #[error("failed to copy layer {layer} {kind:?} cache snapshot: {source}")]
    SnapshotCopy {
        layer: usize,
        kind: CacheKind,
        #[source]
        source: MlxError,
    },
    #[error("layer {layer} {kind:?} cache snapshot shape or dtype does not match")]
    SnapshotBufferMismatch { layer: usize, kind: CacheKind },
    #[error(
        "DeepSeek-V4 cache growth requires a larger context: source {source_context}, target {target_context}"
    )]
    MigrationContext {
        source_context: usize,
        target_context: usize,
    },
    #[error(
        "DeepSeek-V4 cache growth layer count {target_layers} does not match source {source_layers}"
    )]
    MigrationLayerCount {
        source_layers: usize,
        target_layers: usize,
    },
    #[error("layer {layer} {kind:?} cache growth plan is not prefix-compatible")]
    MigrationPlanMismatch { layer: usize, kind: CacheKind },
    #[error("DeepSeek-V4 cache-growth snapshot does not match the source cache plan")]
    MigrationSnapshotPlanMismatch,
    #[error(
        "DeepSeek-V4 cache-growth snapshot position {snapshot_position} exceeds source position {source_position}"
    )]
    MigrationSnapshotPosition {
        snapshot_position: usize,
        source_position: usize,
    },
    #[error("failed to copy layer {layer} {kind:?} during cache growth: {source}")]
    MigrationCopy {
        layer: usize,
        kind: CacheKind,
        #[source]
        source: MlxError,
    },
    #[error("layer {layer} {kind:?} cache growth buffer shape or dtype does not match")]
    MigrationBufferMismatch { layer: usize, kind: CacheKind },
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

/// Exact rollback state for a DeepSeek-V4 cache at a token boundary.
///
/// Compressed and indexer KV rows are append-only: decode after the anchor can
/// only write rows at or beyond the anchor's logical position, and restoring
/// `next_position` makes those future rows invisible until replay overwrites
/// them. The 128-token circular window and recurrent compressor state *are*
/// overwritten by decode, so the snapshot owns compact, non-aliasing copies
/// of exactly those buffers. This keeps rollback exact without duplicating a
/// context-linear multi-gigabyte allocation.
pub struct Deepseek4CacheSnapshot {
    layers: Vec<LayerCacheSnapshot>,
    plan: Deepseek4CachePlan,
    next_position: usize,
    resident_bytes: u64,
}

struct LayerCacheSnapshot {
    window_kv: MlxBuffer,
    main_kv_state: Option<MlxBuffer>,
    main_score_state: Option<MlxBuffer>,
    indexer_kv_state: Option<MlxBuffer>,
    indexer_score_state: Option<MlxBuffer>,
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

    #[cfg(test)]
    pub(super) fn layers_mut(&mut self) -> &mut [LayerCache] {
        &mut self.layers
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn position(&self) -> usize {
        self.next_position
    }

    pub fn capacity(&self) -> usize {
        self.plan.context_length
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Copy a valid live prefix into a larger, prefix-compatible cache.
    ///
    /// The source remains untouched until every destination copy succeeds, so
    /// callers can discard the partially populated destination and continue
    /// serving from the old cache after an allocation or copy failure. The
    /// optional compact rollback snapshot is rebound only after all live bytes
    /// have migrated; this keeps recovery-anchor restore exact across the
    /// capacity boundary without accepting unrelated old-plan snapshots.
    pub fn migrate_from(
        &mut self,
        source: &Deepseek4Cache,
        snapshot: Option<&mut Deepseek4CacheSnapshot>,
    ) -> Result<(), CacheError> {
        if source.poisoned {
            return Err(CacheError::Poisoned);
        }
        validate_growth_plans(&source.plan, &self.plan)?;
        if let Some(snapshot) = snapshot.as_ref() {
            if snapshot.plan != source.plan {
                return Err(CacheError::MigrationSnapshotPlanMismatch);
            }
            if snapshot.next_position > source.next_position {
                return Err(CacheError::MigrationSnapshotPosition {
                    snapshot_position: snapshot.next_position,
                    source_position: source.next_position,
                });
            }
        }

        for (layer_index, (source_layer, destination_layer)) in
            source.layers.iter().zip(self.layers.iter_mut()).enumerate()
        {
            copy_buffer_prefix(
                &source_layer.attention_kv,
                &mut destination_layer.attention_kv,
                layer_index,
                CacheKind::AttentionKv,
            )?;
            copy_optional_buffer_prefix(
                source_layer.indexer_kv.as_ref(),
                destination_layer.indexer_kv.as_mut(),
                layer_index,
                CacheKind::IndexerKv,
            )?;
            copy_optional_buffer_prefix(
                source_layer.main_kv_state.as_ref(),
                destination_layer.main_kv_state.as_mut(),
                layer_index,
                CacheKind::MainKvState,
            )?;
            copy_optional_buffer_prefix(
                source_layer.main_score_state.as_ref(),
                destination_layer.main_score_state.as_mut(),
                layer_index,
                CacheKind::MainScoreState,
            )?;
            copy_optional_buffer_prefix(
                source_layer.indexer_kv_state.as_ref(),
                destination_layer.indexer_kv_state.as_mut(),
                layer_index,
                CacheKind::IndexerKvState,
            )?;
            copy_optional_buffer_prefix(
                source_layer.indexer_score_state.as_ref(),
                destination_layer.indexer_score_state.as_mut(),
                layer_index,
                CacheKind::IndexerScoreState,
            )?;
        }

        self.next_position = source.next_position;
        self.poisoned = false;
        if let Some(snapshot) = snapshot {
            snapshot.plan = self.plan.clone();
        }
        Ok(())
    }

    /// Capture exact prompt-boundary rollback state before generation mutates
    /// the circular window and recurrent compressor state.
    pub fn snapshot(&self) -> Result<Deepseek4CacheSnapshot, CacheError> {
        if self.poisoned {
            return Err(CacheError::Poisoned);
        }
        let mut layers = Vec::with_capacity(self.layers.len());
        let mut resident_bytes = 0_u64;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let snapshot = LayerCacheSnapshot {
                window_kv: snapshot_buffer(
                    &self._device,
                    &layer.window_kv,
                    layer_index,
                    CacheKind::WindowKv,
                )?,
                main_kv_state: snapshot_optional_buffer(
                    &self._device,
                    layer.main_kv_state.as_ref(),
                    layer_index,
                    CacheKind::MainKvState,
                )?,
                main_score_state: snapshot_optional_buffer(
                    &self._device,
                    layer.main_score_state.as_ref(),
                    layer_index,
                    CacheKind::MainScoreState,
                )?,
                indexer_kv_state: snapshot_optional_buffer(
                    &self._device,
                    layer.indexer_kv_state.as_ref(),
                    layer_index,
                    CacheKind::IndexerKvState,
                )?,
                indexer_score_state: snapshot_optional_buffer(
                    &self._device,
                    layer.indexer_score_state.as_ref(),
                    layer_index,
                    CacheKind::IndexerScoreState,
                )?,
            };
            let layer_bytes = std::iter::once(&snapshot.window_kv)
                .chain(snapshot.main_kv_state.iter())
                .chain(snapshot.main_score_state.iter())
                .chain(snapshot.indexer_kv_state.iter())
                .chain(snapshot.indexer_score_state.iter())
                .try_fold(0_u64, |total, buffer| {
                    u64::try_from(buffer.data_byte_len())
                        .ok()
                        .and_then(|bytes| total.checked_add(bytes))
                })
                .ok_or(CacheError::ByteOverflow {
                    layer: layer_index,
                    kind: CacheKind::WindowKv,
                })?;
            resident_bytes =
                resident_bytes
                    .checked_add(layer_bytes)
                    .ok_or(CacheError::ByteOverflow {
                        layer: layer_index,
                        kind: CacheKind::WindowKv,
                    })?;
            layers.push(snapshot);
        }
        Ok(Deepseek4CacheSnapshot {
            layers,
            plan: self.plan.clone(),
            next_position: self.next_position,
            resident_bytes,
        })
    }

    /// Restore a prior token-boundary snapshot into this fixed-capacity cache.
    /// The cache remains allocated; only its bytes and logical position move.
    pub fn restore(&mut self, snapshot: &Deepseek4CacheSnapshot) -> Result<(), CacheError> {
        if snapshot.plan != self.plan {
            return Err(CacheError::SnapshotPlanMismatch);
        }
        if snapshot.layers.len() != self.layers.len() {
            return Err(CacheError::SnapshotLayerCount {
                expected: self.layers.len(),
                actual: snapshot.layers.len(),
            });
        }
        for (layer_index, (source, destination)) in snapshot
            .layers
            .iter()
            .zip(self.layers.iter_mut())
            .enumerate()
        {
            restore_buffer(
                &source.window_kv,
                &mut destination.window_kv,
                layer_index,
                CacheKind::WindowKv,
            )?;
            restore_optional_buffer(
                source.main_kv_state.as_ref(),
                destination.main_kv_state.as_mut(),
                layer_index,
                CacheKind::MainKvState,
            )?;
            restore_optional_buffer(
                source.main_score_state.as_ref(),
                destination.main_score_state.as_mut(),
                layer_index,
                CacheKind::MainScoreState,
            )?;
            restore_optional_buffer(
                source.indexer_kv_state.as_ref(),
                destination.indexer_kv_state.as_mut(),
                layer_index,
                CacheKind::IndexerKvState,
            )?;
            restore_optional_buffer(
                source.indexer_score_state.as_ref(),
                destination.indexer_score_state.as_mut(),
                layer_index,
                CacheKind::IndexerScoreState,
            )?;
        }
        self.next_position = snapshot.next_position;
        self.poisoned = false;
        Ok(())
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

impl Deepseek4CacheSnapshot {
    pub fn position(&self) -> usize {
        self.next_position
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

fn snapshot_buffer(
    device: &MlxDevice,
    source: &MlxBuffer,
    layer: usize,
    kind: CacheKind,
) -> Result<MlxBuffer, CacheError> {
    let mut destination = device
        .alloc_buffer(
            source.data_byte_len(),
            source.dtype(),
            source.shape().to_vec(),
        )
        .map_err(|source| CacheError::SnapshotCopy {
            layer,
            kind,
            source,
        })?;
    copy_buffer(source, &mut destination, layer, kind)?;
    Ok(destination)
}

fn snapshot_optional_buffer(
    device: &MlxDevice,
    source: Option<&MlxBuffer>,
    layer: usize,
    kind: CacheKind,
) -> Result<Option<MlxBuffer>, CacheError> {
    source
        .map(|source| snapshot_buffer(device, source, layer, kind))
        .transpose()
}

fn restore_buffer(
    source: &MlxBuffer,
    destination: &mut MlxBuffer,
    layer: usize,
    kind: CacheKind,
) -> Result<(), CacheError> {
    if source.data_byte_len() != destination.data_byte_len()
        || source.dtype() != destination.dtype()
        || source.shape() != destination.shape()
    {
        return Err(CacheError::SnapshotBufferMismatch { layer, kind });
    }
    copy_buffer(source, destination, layer, kind)
}

fn restore_optional_buffer(
    source: Option<&MlxBuffer>,
    destination: Option<&mut MlxBuffer>,
    layer: usize,
    kind: CacheKind,
) -> Result<(), CacheError> {
    match (source, destination) {
        (Some(source), Some(destination)) => restore_buffer(source, destination, layer, kind),
        (None, None) => Ok(()),
        _ => Err(CacheError::SnapshotBufferMismatch { layer, kind }),
    }
}

fn copy_buffer(
    source: &MlxBuffer,
    destination: &mut MlxBuffer,
    layer: usize,
    kind: CacheKind,
) -> Result<(), CacheError> {
    let source = source
        .as_slice::<u8>()
        .map_err(|source| CacheError::SnapshotCopy {
            layer,
            kind,
            source,
        })?;
    let destination =
        destination
            .as_mut_slice::<u8>()
            .map_err(|source| CacheError::SnapshotCopy {
                layer,
                kind,
                source,
            })?;
    if source.len() != destination.len() {
        return Err(CacheError::SnapshotBufferMismatch { layer, kind });
    }
    destination.copy_from_slice(source);
    Ok(())
}

fn validate_growth_plans(
    source: &Deepseek4CachePlan,
    target: &Deepseek4CachePlan,
) -> Result<(), CacheError> {
    if target.context_length <= source.context_length {
        return Err(CacheError::MigrationContext {
            source_context: source.context_length,
            target_context: target.context_length,
        });
    }
    if source.layers.len() != target.layers.len() {
        return Err(CacheError::MigrationLayerCount {
            source_layers: source.layers.len(),
            target_layers: target.layers.len(),
        });
    }
    for (layer_index, (source, target)) in
        source.layers.iter().zip(target.layers.iter()).enumerate()
    {
        if source.layer_index != target.layer_index
            || source.compress_ratio != target.compress_ratio
            || !buffer_plan_is_prefix(&source.attention_kv, &target.attention_kv)
        {
            return Err(CacheError::MigrationPlanMismatch {
                layer: layer_index,
                kind: CacheKind::AttentionKv,
            });
        }
        if source.window_kv != target.window_kv {
            return Err(CacheError::MigrationPlanMismatch {
                layer: layer_index,
                kind: CacheKind::WindowKv,
            });
        }
        if !optional_buffer_plan_is_prefix(
            source.compressed_kv.as_ref(),
            target.compressed_kv.as_ref(),
        ) {
            return Err(CacheError::MigrationPlanMismatch {
                layer: layer_index,
                kind: CacheKind::CompressedKv,
            });
        }
        if !optional_buffer_plan_is_prefix(source.indexer_kv.as_ref(), target.indexer_kv.as_ref()) {
            return Err(CacheError::MigrationPlanMismatch {
                layer: layer_index,
                kind: CacheKind::IndexerKv,
            });
        }
        for (kind, source, target) in [
            (
                CacheKind::MainKvState,
                source.main_kv_state.as_ref(),
                target.main_kv_state.as_ref(),
            ),
            (
                CacheKind::MainScoreState,
                source.main_score_state.as_ref(),
                target.main_score_state.as_ref(),
            ),
            (
                CacheKind::IndexerKvState,
                source.indexer_kv_state.as_ref(),
                target.indexer_kv_state.as_ref(),
            ),
            (
                CacheKind::IndexerScoreState,
                source.indexer_score_state.as_ref(),
                target.indexer_score_state.as_ref(),
            ),
        ] {
            if source != target {
                return Err(CacheError::MigrationPlanMismatch {
                    layer: layer_index,
                    kind,
                });
            }
        }
    }
    Ok(())
}

fn buffer_plan_is_prefix(source: &CacheBufferPlan, target: &CacheBufferPlan) -> bool {
    source.dtype == target.dtype
        && !source.shape.is_empty()
        && source.shape.len() == target.shape.len()
        && source.shape[1..] == target.shape[1..]
        && source.shape[0] <= target.shape[0]
        && source.bytes <= target.bytes
}

fn optional_buffer_plan_is_prefix(
    source: Option<&CacheBufferPlan>,
    target: Option<&CacheBufferPlan>,
) -> bool {
    match (source, target) {
        (Some(source), Some(target)) => buffer_plan_is_prefix(source, target),
        (None, None) => true,
        _ => false,
    }
}

fn copy_buffer_prefix(
    source: &MlxBuffer,
    destination: &mut MlxBuffer,
    layer: usize,
    kind: CacheKind,
) -> Result<(), CacheError> {
    if source.dtype() != destination.dtype()
        || source.shape().is_empty()
        || source.shape().len() != destination.shape().len()
        || source.shape()[1..] != destination.shape()[1..]
        || source.shape()[0] > destination.shape()[0]
        || source.data_byte_len() > destination.data_byte_len()
    {
        return Err(CacheError::MigrationBufferMismatch { layer, kind });
    }
    let source = source
        .as_slice::<u8>()
        .map_err(|source| CacheError::MigrationCopy {
            layer,
            kind,
            source,
        })?;
    let destination =
        destination
            .as_mut_slice::<u8>()
            .map_err(|source| CacheError::MigrationCopy {
                layer,
                kind,
                source,
            })?;
    destination[..source.len()].copy_from_slice(source);
    Ok(())
}

fn copy_optional_buffer_prefix(
    source: Option<&MlxBuffer>,
    destination: Option<&mut MlxBuffer>,
    layer: usize,
    kind: CacheKind,
) -> Result<(), CacheError> {
    match (source, destination) {
        (Some(source), Some(destination)) => copy_buffer_prefix(source, destination, layer, kind),
        (None, None) => Ok(()),
        _ => Err(CacheError::MigrationBufferMismatch { layer, kind }),
    }
}
