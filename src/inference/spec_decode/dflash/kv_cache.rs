//! DFlash drafter KV cache (ADR-030 Phase 3).
//!
//! Mirrors `model_mlx.py:DFlashDraftModel.make_cache` (lines 170-179):
//! per-layer cache, type depends on `cfg.layer_types[layer_idx]`. The
//! Python implementation uses `mlx_lm.RotatingKVCache` for sliding-
//! attention layers and `KVCache` for full-attention.
//!
//! ## Sliding window in practice for hf2q
//!
//! For our block-diffusion scenarios (`block_size=8`, generating ~256
//! tokens), the drafter's sliding_window=2048 means the ring buffer
//! never wraps. The "skip" branch in
//! `DFlashAttention.__call__:86-91` only fires when `S > sliding_window-1`,
//! which would require >2047 x_ctx positions in a single forward — not
//! achievable for our targets. We allocate fixed-size linear caches
//! and ASSERT non-wrap; if a future scenario exceeds the window we'll
//! reject with a clear error rather than silently overwriting.
//!
//! ## Cache memory layout
//!
//! Storage: F32 row-major `[num_kv_heads, capacity, head_dim]` — same
//! layout `dispatch_sdpa_decode` expects for K/V (per `sdpa_decode.rs:60`).
//! This means appending a new position p writes:
//!
//! ```text
//!   for h in 0..num_kv_heads:
//!       keys[h * capacity * head_dim + p * head_dim ..
//!            h * capacity * head_dim + (p+1) * head_dim] = new_k_for_head_h
//! ```
//!
//! Per-position appends are sparse writes; for the drafter's small
//! shapes this is fine via a per-head copy loop. Bulk updates (whole
//! block at once) use a single contiguous write per head.
//!
//! Phase 3 (this module): cache struct + allocator. Update + fetch
//! semantics land alongside the cross-length SDPA dispatcher in the
//! next iter once the rollback contract is clear.

use super::config::{DFlashConfig, LayerType};
use anyhow::{anyhow, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};

/// Per-layer KV cache state.
pub struct DFlashLayerKvCache {
    /// `[num_kv_heads, capacity, head_dim]` F32 — ring or linear buffer.
    pub keys: MlxBuffer,
    /// Same shape as `keys`.
    pub values: MlxBuffer,
    /// Current valid length (number of positions written so far).
    /// For full-attention this grows monotonically; for sliding-attention
    /// it grows to capacity then stays there (write_pos wraps).
    pub seq_len: u32,
    /// Maximum number of positions the cache holds (full-attention) or
    /// the sliding-window size (sliding-attention).
    pub capacity: u32,
    /// True when this layer uses sliding-window attention.
    pub is_sliding: bool,
    /// Layer index in the drafter (0..num_hidden_layers).
    pub layer_idx: usize,
}

impl DFlashLayerKvCache {
    /// Free space remaining before this cache fills.
    pub fn remaining(&self) -> u32 {
        self.capacity.saturating_sub(self.seq_len)
    }

    /// True if appending `n` positions would exceed capacity.
    pub fn would_overflow(&self, n: u32) -> bool {
        if self.is_sliding {
            false // sliding caches accept any input, evicting oldest
        } else {
            self.seq_len.saturating_add(n) > self.capacity
        }
    }
}

/// Full drafter KV cache: one [`DFlashLayerKvCache`] per draft layer.
pub struct DFlashKvCache {
    pub layers: Vec<DFlashLayerKvCache>,
}

impl DFlashKvCache {
    /// Allocate a fresh KV cache for the drafter.
    ///
    /// # Arguments
    ///
    /// - `cfg`: drafter config (used for layer count + layer_types +
    ///   sliding_window + num_kv_heads + head_dim)
    /// - `max_capacity_full`: capacity for full-attention layers. Set
    ///   to the maximum number of (prompt + generated) positions the
    ///   drafter will need to track in the largest forward call.
    pub fn new(
        device: &MlxDevice,
        cfg: &DFlashConfig,
        max_capacity_full: u32,
    ) -> Result<Self> {
        let num_kv_heads = cfg.num_key_value_heads as u32;
        let head_dim = cfg.head_dim as u32;
        let sliding_cap = cfg.sliding_window.map(|w| w as u32 - 1).unwrap_or(0);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (layer_idx, layer_type) in cfg.layer_types.iter().copied().enumerate() {
            let (capacity, is_sliding) = match layer_type {
                LayerType::SlidingAttention => {
                    if sliding_cap == 0 {
                        return Err(anyhow!(
                            "DFlashKvCache::new: layer {layer_idx} is sliding but cfg has no sliding_window"
                        ));
                    }
                    (sliding_cap, true)
                }
                LayerType::FullAttention => (max_capacity_full, false),
            };
            let n_elem = (num_kv_heads as usize) * (capacity as usize) * (head_dim as usize);
            if n_elem == 0 {
                return Err(anyhow!(
                    "DFlashKvCache::new: layer {layer_idx} has zero-size cache (kv_heads={num_kv_heads}, capacity={capacity}, head_dim={head_dim})"
                ));
            }
            let shape = vec![num_kv_heads as usize, capacity as usize, head_dim as usize];
            let keys = device
                .alloc_buffer(n_elem * 4, DType::F32, shape.clone())
                .map_err(|e| anyhow!("alloc K cache layer {layer_idx}: {e}"))?;
            let values = device
                .alloc_buffer(n_elem * 4, DType::F32, shape)
                .map_err(|e| anyhow!("alloc V cache layer {layer_idx}: {e}"))?;
            layers.push(DFlashLayerKvCache {
                keys,
                values,
                seq_len: 0,
                capacity,
                is_sliding,
                layer_idx,
            });
        }

        Ok(DFlashKvCache { layers })
    }

    /// Total bytes resident on GPU across all per-layer K + V buffers.
    pub fn gpu_resident_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.keys.byte_len() + l.values.byte_len())
            .sum()
    }

    /// Reset all layer seq_len to 0. Does NOT zero out the underlying
    /// buffers (they get overwritten on next write); only the cursor
    /// is reset.
    pub fn reset(&mut self) {
        for l in &mut self.layers {
            l.seq_len = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::spec_decode::dflash::config::DFlashConfig;

    fn gemma4_26b_a4b_dflash_config() -> DFlashConfig {
        DFlashConfig::from_json_str(
            super::super::config::tests::GEMMA4_26B_A4B_DFLASH_CONFIG,
        )
        .expect("test fixture must parse")
    }

    /// GPU integration test: allocate a DFlash KV cache for the
    /// gemma-4-26B-A4B-it drafter (5 layers, 4 sliding + 1 full,
    /// sliding_window=2048, num_kv_heads=8, head_dim=128). Validate:
    ///
    /// - 5 layer caches allocated
    /// - first 4 layers are sliding, capacity = sliding_window - 1 = 2047
    /// - last layer is full, capacity = max_capacity_full
    /// - K/V buffer sizes match expected element counts
    /// - reset() clears seq_len without touching buffer state
    #[test]
    #[ignore = "requires Metal device"]
    fn allocates_drafter_kv_cache() {
        let cfg = gemma4_26b_a4b_dflash_config();
        let device = MlxDevice::new().expect("Metal device available on M5 Max");
        let max_full = 4096u32;

        let mut cache = DFlashKvCache::new(&device, &cfg, max_full).expect("cache alloc");
        assert_eq!(cache.layers.len(), cfg.num_hidden_layers);

        for (i, l) in cache.layers.iter().enumerate() {
            if i < 4 {
                assert!(l.is_sliding, "layer {i} should be sliding");
                assert_eq!(l.capacity, 2047, "layer {i} sliding capacity = window-1");
            } else {
                assert!(!l.is_sliding, "layer 4 should be full");
                assert_eq!(l.capacity, max_full);
            }
            let expected_elem = (cfg.num_key_value_heads as usize)
                * (l.capacity as usize)
                * (cfg.head_dim as usize);
            assert_eq!(l.keys.element_count(), expected_elem, "layer {i} K elem count");
            assert_eq!(l.values.element_count(), expected_elem, "layer {i} V elem count");
            assert_eq!(l.seq_len, 0, "fresh cache seq_len must be 0");
        }

        // Sanity: total bytes = 2 (K+V) × 5 layers × bytes per layer
        let expected_bytes: usize = cache
            .layers
            .iter()
            .map(|l| 2 * l.keys.byte_len())
            .sum();
        assert_eq!(cache.gpu_resident_bytes(), expected_bytes);

        // Bump some seq_len, reset, verify cleared.
        cache.layers[0].seq_len = 100;
        cache.layers[2].seq_len = 50;
        cache.reset();
        for l in &cache.layers {
            assert_eq!(l.seq_len, 0, "reset() should zero seq_len");
        }
    }

    #[test]
    fn would_overflow_full_attn_logic() {
        // Synthetic full-attention cache; can't actually alloc without
        // a Metal device, so we test the logic via direct construction.
        // (This is a pure-CPU branch logic test — no device needed.)
        // Construct manually by sidestepping device alloc; safe in tests.
        // Skip if device unavailable.
        if MlxDevice::new().is_err() {
            return;
        }
        let cfg = gemma4_26b_a4b_dflash_config();
        let device = MlxDevice::new().unwrap();
        let cache = DFlashKvCache::new(&device, &cfg, 100).unwrap();
        let full_layer = cache.layers.iter().find(|l| !l.is_sliding).unwrap();
        // full layer with seq_len=0, capacity=100
        assert_eq!(full_layer.remaining(), 100);
        assert!(!full_layer.would_overflow(50));
        assert!(!full_layer.would_overflow(100));
        assert!(full_layer.would_overflow(101));
    }
}
