//! Atomic start-zero prompt transactions for the DeepSeek-V4 cache.

use super::{CacheError, Deepseek4Cache};

/// Physical writes and final visibility for one start-zero prompt prefill.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerCacheSpan {
    pub layer_index: usize,
    pub window_write_start: usize,
    pub window_valid_after: usize,
    pub compressed_write_start: usize,
    pub compressed_count: usize,
    pub indexer_write_start: usize,
    pub indexer_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CacheSpan {
    pub start_position: usize,
    pub token_count: usize,
    pub layers: Vec<LayerCacheSpan>,
}

impl Deepseek4Cache {
    /// Plan one layer-major prompt without publishing logical visibility.
    pub fn plan_prefill_start0(&self, token_count: usize) -> Result<CacheSpan, CacheError> {
        if self.is_poisoned() {
            return Err(CacheError::Poisoned);
        }
        if token_count == 0 {
            return Err(CacheError::EmptyPrefill);
        }
        if self.next_position != 0 {
            return Err(CacheError::PrefillNotEmpty {
                position: self.next_position,
            });
        }
        if token_count > self.plan.context_length {
            return Err(CacheError::ContextBound {
                requested: token_count,
                maximum: self.plan.context_length,
            });
        }
        let maximum = self
            .plan
            .layers
            .iter()
            .map(|layer| layer.window_kv.shape[0])
            .min()
            .unwrap_or(0);
        if token_count > maximum {
            return Err(CacheError::PrefillWindow {
                requested: token_count,
                maximum,
            });
        }
        let layers = self
            .plan
            .layers
            .iter()
            .map(|layer| {
                let compressed_count = if layer.compress_ratio == 0 {
                    0
                } else {
                    token_count / layer.compress_ratio as usize
                };
                let indexer_count = usize::from(layer.compress_ratio == 4) * compressed_count;
                LayerCacheSpan {
                    layer_index: layer.layer_index,
                    window_write_start: 0,
                    window_valid_after: token_count,
                    compressed_write_start: 0,
                    compressed_count,
                    indexer_write_start: 0,
                    indexer_count,
                }
            })
            .collect();
        Ok(CacheSpan {
            start_position: 0,
            token_count,
            layers,
        })
    }

    /// Publish a fully drained start-zero prefill transaction atomically.
    pub fn commit_prefill(
        &mut self,
        start_position: usize,
        token_count: usize,
    ) -> Result<(), CacheError> {
        if self.is_poisoned() {
            return Err(CacheError::Poisoned);
        }
        if token_count == 0 {
            return Err(CacheError::EmptyPrefill);
        }
        if start_position != self.next_position {
            return Err(CacheError::StepOutOfOrder {
                expected: self.next_position,
                actual: start_position,
            });
        }
        if start_position != 0 {
            return Err(CacheError::PrefillNotEmpty {
                position: start_position,
            });
        }
        let end = start_position
            .checked_add(token_count)
            .ok_or(CacheError::ContextBound {
                requested: usize::MAX,
                maximum: self.plan.context_length,
            })?;
        if end > self.plan.context_length {
            return Err(CacheError::ContextBound {
                requested: end,
                maximum: self.plan.context_length,
            });
        }
        self.next_position = end;
        Ok(())
    }
}
