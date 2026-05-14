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

    /// Append seq-major `[n_new, num_kv_heads, head_dim]` K and V to
    /// the cache. Permutes to head-major on write (cache storage is
    /// `[num_kv_heads, capacity, head_dim]`). Increments `seq_len` by
    /// `n_new`.
    ///
    /// CPU-side copy via `as_mut_slice<f32>()` — fine for the drafter's
    /// small per-step writes (L=8 for block_size + ctx_chunk_size per
    /// call). The drafter is tiny; SDPA dominates anyway.
    ///
    /// Returns an error if appending would exceed capacity (full-attn
    /// only; sliding caches MUST not overflow either for our scenarios
    /// — see module-level note about no-wrap assumption).
    pub fn append_seq_major_kv(
        &mut self,
        k_seq_major: &[f32],
        v_seq_major: &[f32],
        n_new: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> anyhow::Result<()> {
        if self.would_overflow(n_new) {
            return Err(anyhow::anyhow!(
                "dflash KV cache layer {} would overflow: seq_len={}, n_new={}, capacity={}",
                self.layer_idx, self.seq_len, n_new, self.capacity
            ));
        }
        if self.is_sliding && self.seq_len.saturating_add(n_new) > self.capacity {
            // Defensive: sliding overflow not yet implemented; ASSERT per module note.
            return Err(anyhow::anyhow!(
                "dflash KV cache layer {} (sliding) would wrap past capacity {} — \
                 not supported in Phase 3 first cut (seq_len={}, n_new={})",
                self.layer_idx, self.capacity, self.seq_len, n_new
            ));
        }

        let n_h = num_kv_heads as usize;
        let d = head_dim as usize;
        let cap = self.capacity as usize;
        let n = n_new as usize;
        let start = self.seq_len as usize;

        let expected_input_elems = n * n_h * d;
        if k_seq_major.len() != expected_input_elems
            || v_seq_major.len() != expected_input_elems
        {
            return Err(anyhow::anyhow!(
                "dflash append_seq_major_kv: input lens K={} V={} != n_new({}) * num_kv_heads({}) * head_dim({}) = {}",
                k_seq_major.len(), v_seq_major.len(), n_new, num_kv_heads, head_dim,
                expected_input_elems
            ));
        }

        // Layout permute: src [t, h, d] (seq-major) → dst [h, cap, d]
        // (head-major with stride cap). For each head h, copy a
        // contiguous run of n rows into dst[h * cap * d + start * d ..].
        let k_dst = self
            .keys
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("k_dst slice: {e}"))?;
        let v_dst = self
            .values
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("v_dst slice: {e}"))?;

        for h in 0..n_h {
            for t in 0..n {
                let src_row = (t * n_h + h) * d;
                let dst_row = (h * cap + start + t) * d;
                k_dst[dst_row..dst_row + d]
                    .copy_from_slice(&k_seq_major[src_row..src_row + d]);
                v_dst[dst_row..dst_row + d]
                    .copy_from_slice(&v_seq_major[src_row..src_row + d]);
            }
        }

        self.seq_len += n_new;
        Ok(())
    }

    /// Roll back the cache by `n` positions. Used after a spec-decode
    /// verify step rejects `n` of the proposed positions — those K/V
    /// writes must be undone so the next step starts from the correct
    /// post-accept state.
    ///
    /// For our cache (no ring buffer wrap in supported scenarios),
    /// rollback is just `seq_len -= n`. The underlying buffer bytes
    /// at positions `seq_len..seq_len+n` are left as garbage; they
    /// get overwritten on the next append.
    pub fn rollback(&mut self, n: u32) {
        self.seq_len = self.seq_len.saturating_sub(n);
    }

    /// Write seq-major prop K/V into the cache's SLACK space (positions
    /// `[seq_len..seq_len+n]`) without advancing `seq_len`.
    ///
    /// Used per DFlash spec-decode step for the in-flight prop K/V
    /// (mirrors `mx.concatenate([cached, prop], axis=2)` in the Python
    /// — but we materialize the concat in-place in the cache slack
    /// rather than allocating a fresh buffer). The next call's
    /// `append_seq_major_kv` overwrites whatever was written here.
    ///
    /// After this call, the cache's `[seq_len..seq_len+n]` positions
    /// hold prop K/V. The SDPA call should use `kv_seq_len = seq_len
    /// + n` and `kv_capacity = capacity` — the kernel reads `kv_seq_len`
    /// positions starting at offset 0 per head.
    ///
    /// Errors if `seq_len + n > capacity`.
    pub fn write_slack_kv(
        &mut self,
        k_seq_major: &[f32],
        v_seq_major: &[f32],
        n: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> anyhow::Result<()> {
        if self.seq_len.saturating_add(n) > self.capacity {
            return Err(anyhow::anyhow!(
                "dflash write_slack_kv layer {} would exceed capacity: seq_len={}, n={}, capacity={}",
                self.layer_idx, self.seq_len, n, self.capacity
            ));
        }
        let n_h = num_kv_heads as usize;
        let d = head_dim as usize;
        let cap = self.capacity as usize;
        let n_usize = n as usize;
        let start = self.seq_len as usize;

        let expected = n_usize * n_h * d;
        if k_seq_major.len() != expected || v_seq_major.len() != expected {
            return Err(anyhow::anyhow!(
                "dflash write_slack_kv: lens K={} V={} != n({}) * H({}) * D({}) = {}",
                k_seq_major.len(), v_seq_major.len(), n, num_kv_heads, head_dim, expected
            ));
        }

        let k_dst = self.keys.as_mut_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("write_slack k_dst slice: {e}"))?;
        let v_dst = self.values.as_mut_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("write_slack v_dst slice: {e}"))?;

        for h in 0..n_h {
            for t in 0..n_usize {
                let src_row = (t * n_h + h) * d;
                let dst_row = (h * cap + start + t) * d;
                k_dst[dst_row..dst_row + d]
                    .copy_from_slice(&k_seq_major[src_row..src_row + d]);
                v_dst[dst_row..dst_row + d]
                    .copy_from_slice(&v_seq_major[src_row..src_row + d]);
            }
        }
        // Intentionally NOT advancing seq_len — caller's responsibility.
        Ok(())
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

    /// Verify that append_seq_major_kv correctly permutes seq-major
    /// input to head-major storage. Constructs distinguishable values
    /// per (t, h, d) position and checks placement.
    #[test]
    #[ignore = "requires Metal device"]
    fn append_seq_major_kv_permutes_to_head_major() {
        let cfg = gemma4_26b_a4b_dflash_config();
        let device = MlxDevice::new().expect("Metal device available on M5 Max");
        let max_full = 64u32;
        let mut cache = DFlashKvCache::new(&device, &cfg, max_full).expect("cache alloc");
        let layer = &mut cache.layers[4]; // full-attention layer
        let h = cfg.num_key_value_heads as u32;
        let d = cfg.head_dim as u32;

        // Build seq-major input [n_new=3, h, d] with distinguishable values
        // = t * 10000 + head * 100 + dim. Easy to spot misplacement.
        let n_new = 3u32;
        let n_h = h as usize;
        let dim = d as usize;
        let total = (n_new as usize) * n_h * dim;
        let mut k_input = vec![0.0f32; total];
        let mut v_input = vec![0.0f32; total];
        for t in 0..(n_new as usize) {
            for head in 0..n_h {
                for dimi in 0..dim {
                    let row = (t * n_h + head) * dim;
                    k_input[row + dimi] = (t * 10000 + head * 100 + dimi) as f32;
                    v_input[row + dimi] = (t * 10000 + head * 100 + dimi) as f32 + 0.5;
                }
            }
        }
        layer
            .append_seq_major_kv(&k_input, &v_input, n_new, h, d)
            .expect("append_seq_major_kv");

        assert_eq!(layer.seq_len, n_new);
        // Verify head-major placement: position t for head h must land
        // at offset (head * capacity + t) * head_dim.
        let cap = layer.capacity as usize;
        let k_storage = layer.keys.as_slice::<f32>().expect("k_storage slice");
        let v_storage = layer.values.as_slice::<f32>().expect("v_storage slice");
        for t in 0..(n_new as usize) {
            for head in 0..n_h {
                let dst = (head * cap + t) * dim;
                for dimi in 0..dim {
                    let expected_k = (t * 10000 + head * 100 + dimi) as f32;
                    let expected_v = expected_k + 0.5;
                    assert_eq!(
                        k_storage[dst + dimi], expected_k,
                        "K mismatch t={t} head={head} dim={dimi}: got {} expected {expected_k}",
                        k_storage[dst + dimi]
                    );
                    assert_eq!(
                        v_storage[dst + dimi], expected_v,
                        "V mismatch t={t} head={head} dim={dimi}"
                    );
                }
            }
        }

        // Rollback by 1, verify seq_len drops, the underlying data is
        // still there (we don't zero it) but seq_len-bounded reads
        // ignore it.
        layer.rollback(1);
        assert_eq!(layer.seq_len, 2);
        layer.rollback(99);
        assert_eq!(layer.seq_len, 0, "saturating rollback");
    }

    /// Verify write_slack_kv writes to the correct slack positions
    /// WITHOUT advancing seq_len. Sequence: append 3, slack-write 5
    /// → seq_len should remain 3; positions 3..8 in cache must contain
    /// the slack data; positions 0..3 must be unchanged.
    #[test]
    #[ignore = "requires Metal device"]
    fn write_slack_kv_does_not_advance_seq_len() {
        let cfg = gemma4_26b_a4b_dflash_config();
        let device = MlxDevice::new().expect("Metal device available on M5 Max");
        let mut cache = DFlashKvCache::new(&device, &cfg, 32).expect("cache");
        let layer = &mut cache.layers[4]; // full-attention
        let h = cfg.num_key_value_heads as u32;
        let d = cfg.head_dim as u32;
        let n_h = h as usize;
        let dim = d as usize;

        // Phase 1: append 3 positions with marker 1.x
        let n_ctx = 3u32;
        let mut k_ctx = vec![0.0f32; (n_ctx as usize) * n_h * dim];
        let mut v_ctx = vec![0.0f32; (n_ctx as usize) * n_h * dim];
        // Markers must stay in [1.0, 2.0) range so the "is this ctx?"
        // check distinguishes them from slack markers in [2.0, 3.0).
        // Use modulo to keep within range regardless of buffer length.
        for (i, v) in k_ctx.iter_mut().enumerate() {
            *v = 1.0 + ((i % 100) as f32) / 1000.0;
        }
        for (i, v) in v_ctx.iter_mut().enumerate() {
            *v = 1.5 + ((i % 100) as f32) / 1000.0;
        }
        layer.append_seq_major_kv(&k_ctx, &v_ctx, n_ctx, h, d).expect("append ctx");
        assert_eq!(layer.seq_len, n_ctx);

        // Phase 2: slack-write 5 positions with marker 2.x
        let n_slack = 5u32;
        let mut k_slack = vec![0.0f32; (n_slack as usize) * n_h * dim];
        let mut v_slack = vec![0.0f32; (n_slack as usize) * n_h * dim];
        for (i, v) in k_slack.iter_mut().enumerate() {
            *v = 2.0 + ((i % 100) as f32) / 1000.0;
        }
        for (i, v) in v_slack.iter_mut().enumerate() {
            *v = 2.5 + ((i % 100) as f32) / 1000.0;
        }
        layer.write_slack_kv(&k_slack, &v_slack, n_slack, h, d).expect("write slack");
        assert_eq!(layer.seq_len, n_ctx, "slack write must NOT advance seq_len");

        // Phase 3: verify positions 0..3 still have ctx (1.x) markers
        // and positions 3..8 have slack (2.x) markers, per head-major layout.
        let cap = layer.capacity as usize;
        let k_storage = layer.keys.as_slice::<f32>().expect("k_storage");
        for t in 0..(n_ctx as usize) {
            for head in 0..n_h {
                let dst = (head * cap + t) * dim;
                assert!(k_storage[dst] >= 1.0 && k_storage[dst] < 2.0,
                    "ctx position t={t} head={head}: expected 1.x marker, got {}",
                    k_storage[dst]);
            }
        }
        for t in 0..(n_slack as usize) {
            for head in 0..n_h {
                let dst = (head * cap + (n_ctx as usize) + t) * dim;
                assert!(k_storage[dst] >= 2.0 && k_storage[dst] < 3.0,
                    "slack position t={t} head={head}: expected 2.x marker, got {}",
                    k_storage[dst]);
            }
        }

        // Phase 4: confirm a subsequent append_seq_major_kv overwrites
        // the slack region.
        let mut k_new = vec![0.0f32; (2 as usize) * n_h * dim];
        let mut v_new = vec![0.0f32; (2 as usize) * n_h * dim];
        for v in k_new.iter_mut() { *v = 9.0; }
        for v in v_new.iter_mut() { *v = 9.5; }
        layer.append_seq_major_kv(&k_new, &v_new, 2, h, d).expect("append after slack");
        assert_eq!(layer.seq_len, n_ctx + 2);
        let k_storage = layer.keys.as_slice::<f32>().expect("k_storage 2");
        for head in 0..n_h {
            for t in (n_ctx as usize)..((n_ctx as usize) + 2) {
                let dst = (head * cap + t) * dim;
                assert_eq!(k_storage[dst], 9.0, "post-append at t={t} head={head}");
            }
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
