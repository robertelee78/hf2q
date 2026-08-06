//! Atomic bounded prompt transactions for the DeepSeek-V4 cache.

use super::{CacheError, Deepseek4Cache};

/// Physical writes and final visibility for one prompt transaction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerCacheSpan {
    pub layer_index: usize,
    pub window_write_start: usize,
    pub window_valid_after: usize,
    pub compressed_write_start: usize,
    pub compressed_count: usize,
    pub compressed_valid_after: usize,
    pub indexer_write_start: usize,
    pub indexer_count: usize,
    pub indexer_valid_after: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CacheSpan {
    pub start_position: usize,
    pub token_count: usize,
    pub layers: Vec<LayerCacheSpan>,
}

impl Deepseek4Cache {
    /// Plan one layer-major prompt transaction without publishing logical
    /// visibility. Raw attention reads the transaction's compact KV source,
    /// while the live circular cache retains only its final physical window.
    pub fn plan_prefill(&self, token_count: usize) -> Result<CacheSpan, CacheError> {
        if self.is_poisoned() {
            return Err(CacheError::Poisoned);
        }
        if token_count == 0 {
            return Err(CacheError::EmptyPrefill);
        }
        let end = self
            .next_position
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
        let layers = self
            .plan
            .layers
            .iter()
            .map(|layer| {
                let ratio = layer.compress_ratio as usize;
                let compressed_before = if ratio == 0 {
                    0
                } else {
                    self.next_position / ratio
                };
                let compressed_valid_after = if ratio == 0 { 0 } else { end / ratio };
                let compressed_count = compressed_valid_after - compressed_before;
                let indexer_count = usize::from(layer.compress_ratio == 4) * compressed_count;
                let indexer_valid_after =
                    usize::from(layer.compress_ratio == 4) * compressed_valid_after;
                LayerCacheSpan {
                    layer_index: layer.layer_index,
                    window_write_start: self.next_position % layer.window_kv.shape[0],
                    window_valid_after: end.min(layer.window_kv.shape[0]),
                    compressed_write_start: compressed_before,
                    compressed_count,
                    compressed_valid_after,
                    indexer_write_start: usize::from(layer.compress_ratio == 4) * compressed_before,
                    indexer_count,
                    indexer_valid_after,
                }
            })
            .collect();
        Ok(CacheSpan {
            start_position: self.next_position,
            token_count,
            layers,
        })
    }

    /// Backward-compatible strict start-zero planner used by callers that
    /// require an empty-cache transaction.
    pub fn plan_prefill_start0(&self, token_count: usize) -> Result<CacheSpan, CacheError> {
        if self.next_position != 0 {
            return Err(CacheError::PrefillNotEmpty {
                position: self.next_position,
            });
        }
        self.plan_prefill(token_count)
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
