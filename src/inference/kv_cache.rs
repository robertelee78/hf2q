//! KV cache with dual sliding ring buffer and global layout for Gemma 4.
//!
//! Gemma 4 has two types of attention layers:
//! - **Sliding** (25 of 30 layers): 8 KV heads, head_dim 256, window 1024.
//!   Uses a ring buffer that wraps at window size.
//! - **Global** (5 of 30 layers): 2 KV heads, head_dim 512, grows to max context.
//!   Pre-allocated to `max_seq_len` but tracks how many positions are filled.
//!
//! All cache buffers are pre-allocated at model load time as Metal buffers
//! (bf16, StorageModeShared) to avoid allocation during inference and to
//! eliminate bf16↔f32 conversions on the KV cache hot path.

use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};
use mlx_native::ops::kv_cache_copy;
use tracing::debug;

/// Errors from KV cache operations.
#[derive(Debug, thiserror::Error)]
pub enum KvCacheError {
    #[error("Metal buffer allocation failed for KV cache: {reason}")]
    AllocationFailed { reason: String },

    #[error("Layer index {index} out of bounds (num_layers={num_layers})")]
    LayerOutOfBounds { index: usize, num_layers: usize },

    #[error("KV cache buffer write error: {reason}")]
    BufferWriteError { reason: String },
}

/// Type of KV cache for a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheType {
    /// Sliding window attention: ring buffer of fixed size.
    Sliding {
        /// Window size in positions (e.g. 1024).
        window: usize,
    },
    /// Global (full) attention: growable up to max_seq_len.
    Global {
        /// Maximum sequence length this cache can hold.
        max_seq_len: usize,
    },
}

/// Per-layer KV cache entry.
#[derive(Debug)]
pub struct LayerCache {
    /// Key cache buffer: bf16, pre-allocated.
    /// Sliding: `[window, n_kv_heads * head_dim]`
    /// Global: `[max_seq_len, n_kv_heads * head_dim]`
    pub k_cache: MlxBuffer,

    /// Value cache buffer: bf16, pre-allocated.
    /// Same shape as k_cache.
    pub v_cache: MlxBuffer,

    /// Current write position (wraps for sliding, grows for global).
    position: usize,

    /// Total number of valid positions stored (for global: equals position;
    /// for sliding: min(total_written, window)).
    total_written: usize,

    /// Cache type and configuration.
    cache_type: CacheType,

    /// Number of KV heads for this layer.
    n_kv_heads: usize,

    /// Head dimension for this layer.
    head_dim: usize,
}

impl LayerCache {
    /// Create a new pre-allocated layer cache.
    fn new(
        device: &MlxDevice,
        cache_type: CacheType,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, KvCacheError> {
        let capacity = match cache_type {
            CacheType::Sliding { window } => window,
            CacheType::Global { max_seq_len } => max_seq_len,
        };

        let row_size = n_kv_heads * head_dim;
        let byte_len = capacity * row_size * std::mem::size_of::<u16>(); // bf16 = 2 bytes

        if byte_len == 0 {
            return Err(KvCacheError::AllocationFailed {
                reason: "Zero-size KV cache (check n_kv_heads, head_dim, capacity)".into(),
            });
        }

        let k_cache = device
            .alloc_buffer(byte_len, DType::BF16, vec![capacity, row_size])
            .map_err(|e| KvCacheError::AllocationFailed {
                reason: format!("K cache: {e}"),
            })?;

        let v_cache = device
            .alloc_buffer(byte_len, DType::BF16, vec![capacity, row_size])
            .map_err(|e| KvCacheError::AllocationFailed {
                reason: format!("V cache: {e}"),
            })?;

        Ok(Self {
            k_cache,
            v_cache,
            position: 0,
            total_written: 0,
            cache_type,
            n_kv_heads,
            head_dim,
        })
    }

    /// Append new K and V vectors (bf16 as u16) for one or more sequence positions.
    ///
    /// `k_new` and `v_new` are bf16 (u16) slices of shape `[n_new_positions, n_kv_heads * head_dim]`.
    /// This is the fast path: no conversion needed since the cache is bf16.
    ///
    /// For sliding layers, positions wrap around the ring buffer.
    /// For global layers, positions are appended sequentially.
    pub fn append_bf16(&mut self, k_new: &[u16], v_new: &[u16]) -> Result<(), KvCacheError> {
        let row_size = self.n_kv_heads * self.head_dim;
        if k_new.len() % row_size != 0 || v_new.len() % row_size != 0 {
            return Err(KvCacheError::BufferWriteError {
                reason: format!(
                    "K/V new data length must be a multiple of row_size={row_size}, \
                     got k={}, v={}",
                    k_new.len(),
                    v_new.len()
                ),
            });
        }

        let n_new = k_new.len() / row_size;
        if v_new.len() / row_size != n_new {
            return Err(KvCacheError::BufferWriteError {
                reason: "K and V must have the same number of new positions".into(),
            });
        }

        // Write each new position into the cache
        let k_slice: &mut [u16] = self
            .k_cache
            .as_mut_slice()
            .map_err(|e| KvCacheError::BufferWriteError {
                reason: format!("K cache write: {e}"),
            })?;
        let v_slice: &mut [u16] = self
            .v_cache
            .as_mut_slice()
            .map_err(|e| KvCacheError::BufferWriteError {
                reason: format!("V cache write: {e}"),
            })?;

        for i in 0..n_new {
            let write_pos = match self.cache_type {
                CacheType::Sliding { window } => (self.position + i) % window,
                CacheType::Global { max_seq_len } => {
                    let pos = self.position + i;
                    if pos >= max_seq_len {
                        return Err(KvCacheError::BufferWriteError {
                            reason: format!(
                                "Global cache overflow: position {pos} >= max_seq_len {max_seq_len}"
                            ),
                        });
                    }
                    pos
                }
            };

            let src_start = i * row_size;
            let dst_start = write_pos * row_size;
            k_slice[dst_start..dst_start + row_size]
                .copy_from_slice(&k_new[src_start..src_start + row_size]);
            v_slice[dst_start..dst_start + row_size]
                .copy_from_slice(&v_new[src_start..src_start + row_size]);
        }

        self.position += n_new;
        self.total_written += n_new;

        // For sliding, position wraps around capacity
        if let CacheType::Sliding { window } = self.cache_type {
            if self.position >= window {
                self.position %= window;
            }
        }

        Ok(())
    }

    /// Append new K and V tensors directly on the GPU, avoiding CPU round-trips.
    ///
    /// Both `k_src` and `v_src` must be bf16 `MlxBuffer`s with shape
    /// `[n_new, n_kv_heads * head_dim]`. The kernel copies data directly from
    /// the source buffers into the pre-allocated cache buffers at the correct
    /// write position, handling ring buffer wrapping for sliding window caches.
    ///
    /// The encoder is NOT committed here — the caller batches this with other
    /// GPU work and calls `commit_and_wait()` afterwards. Position tracking
    /// is updated on the CPU side immediately (safe because positions are
    /// metadata, not GPU data).
    ///
    /// # Arguments
    ///
    /// * `k_src`    - Source K buffer (bf16), shape `[n_new, n_kv_heads * head_dim]`.
    /// * `v_src`    - Source V buffer (bf16), shape `[n_new, n_kv_heads * head_dim]`.
    /// * `encoder`  - Command encoder to record GPU copy dispatches into.
    /// * `registry` - Kernel registry (must have `kv_cache_copy` registered).
    /// * `device`   - Metal device for pipeline compilation.
    pub fn append_gpu(
        &mut self,
        k_src: &MlxBuffer,
        v_src: &MlxBuffer,
        encoder: &mut CommandEncoder,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
    ) -> Result<(), KvCacheError> {
        let row_size = self.n_kv_heads * self.head_dim;
        if row_size == 0 {
            return Err(KvCacheError::BufferWriteError {
                reason: "row_size is zero".into(),
            });
        }

        let k_elements = k_src.element_count();
        let v_elements = v_src.element_count();
        if k_elements % row_size != 0 || v_elements % row_size != 0 {
            return Err(KvCacheError::BufferWriteError {
                reason: format!(
                    "GPU append: K/V element counts must be multiples of row_size={row_size}, \
                     got k={k_elements}, v={v_elements}"
                ),
            });
        }

        let n_new = k_elements / row_size;
        if v_elements / row_size != n_new {
            return Err(KvCacheError::BufferWriteError {
                reason: "GPU append: K and V must have the same number of new positions".into(),
            });
        }

        let (cache_cap, is_sliding) = match self.cache_type {
            CacheType::Sliding { window } => (window, true),
            CacheType::Global { max_seq_len } => {
                if self.position + n_new > max_seq_len {
                    return Err(KvCacheError::BufferWriteError {
                        reason: format!(
                            "Global cache overflow: position {} + n_new {} > max_seq_len {}",
                            self.position, n_new, max_seq_len
                        ),
                    });
                }
                (max_seq_len, false)
            }
        };

        // Dispatch GPU copy for K
        kv_cache_copy::dispatch_kv_cache_copy(
            encoder,
            registry,
            device,
            k_src,
            &self.k_cache,
            self.position as u32,
            row_size as u32,
            n_new as u32,
            cache_cap as u32,
            is_sliding,
        )
        .map_err(|e| KvCacheError::BufferWriteError {
            reason: format!("GPU K cache copy: {e}"),
        })?;

        // Dispatch GPU copy for V
        kv_cache_copy::dispatch_kv_cache_copy(
            encoder,
            registry,
            device,
            v_src,
            &self.v_cache,
            self.position as u32,
            row_size as u32,
            n_new as u32,
            cache_cap as u32,
            is_sliding,
        )
        .map_err(|e| KvCacheError::BufferWriteError {
            reason: format!("GPU V cache copy: {e}"),
        })?;

        // Update CPU-side position tracking (same logic as append_bf16)
        self.position += n_new;
        self.total_written += n_new;

        if let CacheType::Sliding { window } = self.cache_type {
            if self.position >= window {
                self.position %= window;
            }
        }

        Ok(())
    }

    /// Append new K and V vectors for one or more sequence positions.
    ///
    /// `k_new` and `v_new` are f32 buffers of shape `[n_new_positions, n_kv_heads * head_dim]`.
    /// Each f32 value is converted to bf16 before storing. Prefer `append_bf16` when
    /// the data is already in bf16 format.
    ///
    /// For sliding layers, positions wrap around the ring buffer.
    /// For global layers, positions are appended sequentially.
    pub fn append(&mut self, k_new: &[f32], v_new: &[f32]) -> Result<(), KvCacheError> {
        // Convert f32 -> bf16 (u16) and delegate to append_bf16
        let k_bf16: Vec<u16> = k_new
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_bits())
            .collect();
        let v_bf16: Vec<u16> = v_new
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_bits())
            .collect();
        self.append_bf16(&k_bf16, &v_bf16)
    }

    /// Get the current valid sequence length in the cache.
    pub fn seq_len(&self) -> usize {
        match self.cache_type {
            CacheType::Sliding { window } => std::cmp::min(self.total_written, window),
            CacheType::Global { .. } => self.position,
        }
    }

    /// Get references to the K and V cache buffers and the valid sequence length.
    ///
    /// For sliding caches, the returned buffers contain the full ring buffer;
    /// the caller must use `seq_len()` to determine how many positions are valid.
    /// For global caches, positions `0..seq_len()` are valid.
    pub fn keys_values(&self) -> (&MlxBuffer, &MlxBuffer, usize) {
        (&self.k_cache, &self.v_cache, self.seq_len())
    }

    /// Reset the cache to empty state.
    pub fn reset(&mut self) {
        self.position = 0;
        self.total_written = 0;
    }

    /// Restore the cache write position to a specific point.
    ///
    /// Used by prompt caching: after finding a common prefix of length `P`,
    /// the cache is restored to position `P` so that only the new tokens
    /// (from `P` onward) need to be encoded. The KV data in positions
    /// `0..P` is already valid from the previous forward pass.
    ///
    /// For sliding caches: if `total_written > window`, the `position`
    /// (ring buffer write head) wraps as `total_written % window`.
    /// For global caches: `position == total_written`.
    ///
    /// Returns an error if `new_total_written` exceeds the cache capacity
    /// (should never happen in practice since we are restoring to a
    /// previously valid state).
    pub fn restore_position(&mut self, new_total_written: usize) -> Result<(), KvCacheError> {
        match self.cache_type {
            CacheType::Sliding { window } => {
                // For sliding, the ring buffer write head wraps
                self.total_written = new_total_written;
                self.position = new_total_written % window;
            }
            CacheType::Global { max_seq_len } => {
                if new_total_written > max_seq_len {
                    return Err(KvCacheError::BufferWriteError {
                        reason: format!(
                            "Cannot restore global cache to position {new_total_written}: \
                             exceeds max_seq_len {max_seq_len}"
                        ),
                    });
                }
                self.total_written = new_total_written;
                self.position = new_total_written;
            }
        }
        Ok(())
    }

    /// Get the cache type.
    pub fn cache_type(&self) -> CacheType {
        self.cache_type
    }

    /// Get the number of KV heads.
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    /// Get the head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get the write position (useful for computing RoPE positions).
    pub fn write_position(&self) -> usize {
        match self.cache_type {
            CacheType::Sliding { .. } => self.total_written,
            CacheType::Global { .. } => self.position,
        }
    }
}

/// Full KV cache for all layers in the model.
pub struct KvCache {
    /// Per-layer cache entries, indexed by layer number.
    layers: Vec<LayerCache>,
}

impl KvCache {
    /// Pre-allocate the complete KV cache for Gemma 4.
    ///
    /// # Arguments
    ///
    /// * `device` - Metal device for buffer allocation
    /// * `layer_types` - Slice of `CacheType` for each layer
    /// * `layer_kv_configs` - Slice of `(n_kv_heads, head_dim)` per layer
    pub fn new(
        device: &MlxDevice,
        layer_types: &[CacheType],
        layer_kv_configs: &[(usize, usize)],
    ) -> Result<Self, KvCacheError> {
        if layer_types.len() != layer_kv_configs.len() {
            return Err(KvCacheError::AllocationFailed {
                reason: format!(
                    "layer_types length ({}) != layer_kv_configs length ({})",
                    layer_types.len(),
                    layer_kv_configs.len()
                ),
            });
        }

        let num_layers = layer_types.len();
        let mut layers = Vec::with_capacity(num_layers);

        for (i, (&cache_type, &(n_kv_heads, head_dim))) in
            layer_types.iter().zip(layer_kv_configs.iter()).enumerate()
        {
            debug!(
                layer = i,
                cache_type = ?cache_type,
                n_kv_heads = n_kv_heads,
                head_dim = head_dim,
                "Allocating KV cache for layer"
            );
            layers.push(LayerCache::new(device, cache_type, n_kv_heads, head_dim)?);
        }

        let total_bytes: usize = layers
            .iter()
            .map(|l| l.k_cache.byte_len() + l.v_cache.byte_len())
            .sum();

        debug!(
            num_layers = num_layers,
            total_mb = total_bytes / (1024 * 1024),
            "KV cache pre-allocated"
        );

        Ok(Self { layers })
    }

    /// Get a mutable reference to a specific layer's cache.
    pub fn layer_mut(&mut self, index: usize) -> Result<&mut LayerCache, KvCacheError> {
        let num_layers = self.layers.len();
        self.layers
            .get_mut(index)
            .ok_or(KvCacheError::LayerOutOfBounds {
                index,
                num_layers,
            })
    }

    /// Get an immutable reference to a specific layer's cache.
    pub fn layer(&self, index: usize) -> Result<&LayerCache, KvCacheError> {
        self.layers
            .get(index)
            .ok_or(KvCacheError::LayerOutOfBounds {
                index,
                num_layers: self.layers.len(),
            })
    }

    /// Reset all layer caches to empty state.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Number of layers in the cache.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Restore all layers' write positions to a given total-written count.
    ///
    /// Used by prompt caching to rewind the KV cache to the end of a
    /// previously-computed prefix. Each layer's `position` and
    /// `total_written` are set so that the next `append()` will write
    /// at position `new_total_written`.
    pub fn restore_all_positions(&mut self, new_total_written: usize) -> Result<(), KvCacheError> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.restore_position(new_total_written).map_err(|e| {
                KvCacheError::BufferWriteError {
                    reason: format!("Layer {i} restore failed: {e}"),
                }
            })?;
        }
        Ok(())
    }

    /// Total allocated byte size of all KV cache buffers across all layers.
    ///
    /// This reflects the pre-allocated capacity (not just the filled portion).
    pub fn total_allocated_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.k_cache.byte_len() + l.v_cache.byte_len())
            .sum()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn test_sliding_cache_basic() {
        let device = MlxDevice::new().expect("device");
        let mut cache =
            LayerCache::new(&device, CacheType::Sliding { window: 4 }, 2, 3).expect("cache");

        assert_eq!(cache.seq_len(), 0);

        // Append one position: 2 kv_heads * 3 head_dim = 6 floats
        let k = vec![1.0f32; 6];
        let v = vec![2.0f32; 6];
        cache.append(&k, &v).unwrap();
        assert_eq!(cache.seq_len(), 1);
        assert_eq!(cache.write_position(), 1);

        // Append 3 more positions to fill the window
        let k3 = vec![3.0f32; 18]; // 3 positions * 6 floats
        let v3 = vec![4.0f32; 18];
        cache.append(&k3, &v3).unwrap();
        assert_eq!(cache.seq_len(), 4); // window full
        assert_eq!(cache.write_position(), 4);
    }

    #[test]
    fn test_sliding_cache_wrap_around() {
        let device = MlxDevice::new().expect("device");
        let mut cache =
            LayerCache::new(&device, CacheType::Sliding { window: 3 }, 1, 2).expect("cache");

        // Fill 3 positions
        let k = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 positions * 2 floats
        let v = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        cache.append(&k, &v).unwrap();
        assert_eq!(cache.seq_len(), 3);

        // Append one more, should wrap to position 0
        let k_new = vec![7.0, 8.0];
        let v_new = vec![70.0, 80.0];
        cache.append(&k_new, &v_new).unwrap();
        assert_eq!(cache.seq_len(), 3); // still 3 (window size)
        assert_eq!(cache.write_position(), 4); // total written

        // Verify the ring buffer: position 0 should have been overwritten
        // Cache is bf16, so read as u16 and convert back to f32 for comparison
        let k_data: &[u16] = cache.k_cache.as_slice().expect("as_slice");
        let to_f32 = |bits: u16| half::bf16::from_bits(bits).to_f32();
        // Position 0 was overwritten with [7.0, 8.0]
        assert!((to_f32(k_data[0]) - 7.0).abs() < 0.1);
        assert!((to_f32(k_data[1]) - 8.0).abs() < 0.1);
        // Position 1 still has [3.0, 4.0]
        assert!((to_f32(k_data[2]) - 3.0).abs() < 0.1);
        assert!((to_f32(k_data[3]) - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_global_cache_basic() {
        let device = MlxDevice::new().expect("device");
        let mut cache =
            LayerCache::new(&device, CacheType::Global { max_seq_len: 100 }, 2, 4).expect("cache");

        assert_eq!(cache.seq_len(), 0);

        let k = vec![1.0f32; 8]; // 1 position * 2 kv_heads * 4 head_dim
        let v = vec![2.0f32; 8];
        cache.append(&k, &v).unwrap();
        assert_eq!(cache.seq_len(), 1);

        let k2 = vec![3.0f32; 16]; // 2 positions
        let v2 = vec![4.0f32; 16];
        cache.append(&k2, &v2).unwrap();
        assert_eq!(cache.seq_len(), 3);
    }

    #[test]
    fn test_global_cache_overflow() {
        let device = MlxDevice::new().expect("device");
        let mut cache =
            LayerCache::new(&device, CacheType::Global { max_seq_len: 2 }, 1, 2).expect("cache");

        let k = vec![1.0f32; 4]; // 2 positions
        let v = vec![2.0f32; 4];
        cache.append(&k, &v).unwrap();

        // This should fail: trying to write to position 2 when max is 2
        let k_extra = vec![3.0f32; 2];
        let v_extra = vec![4.0f32; 2];
        let result = cache.append(&k_extra, &v_extra);
        assert!(result.is_err());
    }

    #[test]
    fn test_kv_cache_multi_layer() {
        let device = MlxDevice::new().expect("device");

        let layer_types = vec![
            CacheType::Sliding { window: 1024 },
            CacheType::Sliding { window: 1024 },
            CacheType::Global { max_seq_len: 4096 },
        ];
        let kv_configs = vec![(8, 256), (8, 256), (2, 512)];

        let cache = KvCache::new(&device, &layer_types, &kv_configs).expect("cache");
        assert_eq!(cache.num_layers(), 3);

        let l0 = cache.layer(0).unwrap();
        assert_eq!(l0.cache_type(), CacheType::Sliding { window: 1024 });
        assert_eq!(l0.n_kv_heads(), 8);
        assert_eq!(l0.head_dim(), 256);

        let l2 = cache.layer(2).unwrap();
        assert_eq!(l2.cache_type(), CacheType::Global { max_seq_len: 4096 });
        assert_eq!(l2.n_kv_heads(), 2);
        assert_eq!(l2.head_dim(), 512);
    }

    #[test]
    fn test_cache_reset() {
        let device = MlxDevice::new().expect("device");
        let mut cache =
            LayerCache::new(&device, CacheType::Sliding { window: 10 }, 1, 2).expect("cache");

        let k = vec![1.0f32; 4]; // 2 positions
        let v = vec![2.0f32; 4];
        cache.append(&k, &v).unwrap();
        assert_eq!(cache.seq_len(), 2);

        cache.reset();
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.write_position(), 0);
    }

    #[test]
    fn test_keys_values() {
        let device = MlxDevice::new().expect("device");
        let mut cache =
            LayerCache::new(&device, CacheType::Global { max_seq_len: 10 }, 1, 2).expect("cache");

        let k = vec![1.0f32, 2.0, 3.0, 4.0]; // 2 positions * 2 floats
        let v = vec![5.0f32, 6.0, 7.0, 8.0];
        cache.append(&k, &v).unwrap();

        let (k_buf, v_buf, seq_len) = cache.keys_values();
        assert_eq!(seq_len, 2);
        assert_eq!(k_buf.dtype(), DType::BF16);
        assert_eq!(v_buf.dtype(), DType::BF16);
    }

    #[test]
    fn test_layer_out_of_bounds() {
        let device = MlxDevice::new().expect("device");
        let cache = KvCache::new(
            &device,
            &[CacheType::Sliding { window: 4 }],
            &[(1, 2)],
        )
        .expect("cache");

        let result = cache.layer(1);
        assert!(result.is_err());
        match result.unwrap_err() {
            KvCacheError::LayerOutOfBounds { index, num_layers } => {
                assert_eq!(index, 1);
                assert_eq!(num_layers, 1);
            }
            other => panic!("Expected LayerOutOfBounds, got: {other}"),
        }
    }

    #[test]
    fn test_mismatched_k_v_lengths() {
        let device = MlxDevice::new().expect("device");
        let mut cache =
            LayerCache::new(&device, CacheType::Global { max_seq_len: 10 }, 1, 2).expect("cache");

        let k = vec![1.0f32; 4]; // 2 positions
        let v = vec![1.0f32; 2]; // 1 position
        let result = cache.append(&k, &v);
        assert!(result.is_err());
    }
}
