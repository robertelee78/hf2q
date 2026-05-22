//! ADR-037 Phase E5b — Drafter KV cache + rollback.
//!
//! `DrafterKvCache` owns per-tree-node K and V buffers on the GPU
//! for the EAGLE-3 drafter. Each tree node `i` owns slot `i` in the
//! cache. Operations:
//!
//! - **append**: write K/V at slot `i` after computing them in the
//!   forward pass.
//! - **rollback_to_accepted**: after Phase E5a tree-walk produces an
//!   accepted-node-index list, compact the cache so the accepted
//!   slots occupy positions `[0, accepted.len())` in cache order.
//!   Non-accepted slots are dropped — their entries become free
//!   capacity for the next tree expansion.
//!
//! ## Layout
//!
//! Cache buffer shape: `[num_kv_heads, capacity, head_dim]` F32
//! row-major flat (matches `tree_attention` kernel's expected
//! K/V cache layout).
//!
//! ## Why pure F32 instead of BF16
//!
//! `tree_attention` accepts both F32 and F16 KV (per Phase E1 dispatch
//! wrapper). For simplicity + minimal precision loss across
//! append/rollback round-trips, this initial implementation uses
//! F32. Phase E7 may switch to F16 if memory becomes a bottleneck.
//!
//! ## Rollback algorithm
//!
//! Implemented as CPU download → reorder → upload. Simpler than a
//! Metal compact-permute kernel and correct. The cost is bounded:
//! `accepted.len() * num_kv_heads * head_dim * 4` bytes downloaded
//! + uploaded per rollback. At typical Qwen-3.6-27B shapes (head_dim
//! = 128, num_kv_heads = 8, accepted = ~10 tokens), ~40 KB per
//! K and V each — well within Apple Silicon unified memory
//! bandwidth. Future Phase E7 can replace with a GPU compact kernel
//! if profiling reveals it matters.

use anyhow::{anyhow, ensure, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};

/// Drafter K/V cache for tree-decoding spec-decode.
pub struct DrafterKvCache {
    /// `[num_kv_heads, capacity, head_dim]` F32. Lives on GPU.
    pub k_buf: MlxBuffer,
    /// Same shape + layout as `k_buf`.
    pub v_buf: MlxBuffer,
    /// Number of KV heads (cache stride dim 0).
    pub num_kv_heads: usize,
    /// Allocated capacity (positions axis). `len <= capacity`.
    pub capacity: usize,
    /// Per-head dim (innermost axis).
    pub head_dim: usize,
    /// Current valid length: positions `[0, len)` are populated;
    /// `[len, capacity)` are uninitialized (may contain stale data).
    len: usize,
}

impl DrafterKvCache {
    /// Allocate a fresh empty KV cache. `capacity` must be > 0.
    pub fn new(
        device: &MlxDevice,
        num_kv_heads: usize,
        capacity: usize,
        head_dim: usize,
    ) -> Result<Self> {
        ensure!(num_kv_heads > 0, "DrafterKvCache: num_kv_heads must be > 0");
        ensure!(capacity > 0, "DrafterKvCache: capacity must be > 0");
        ensure!(head_dim > 0, "DrafterKvCache: head_dim must be > 0");
        let total_elems = num_kv_heads
            .checked_mul(capacity)
            .and_then(|v| v.checked_mul(head_dim))
            .ok_or_else(|| {
                anyhow!(
                    "DrafterKvCache: num_kv_heads ({}) * capacity ({}) * head_dim ({}) overflows usize",
                    num_kv_heads,
                    capacity,
                    head_dim
                )
            })?;
        ensure!(
            total_elems <= (u32::MAX as usize),
            "DrafterKvCache: total elements ({}) exceeds u32::MAX",
            total_elems
        );
        let total_bytes = total_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow!("DrafterKvCache: byte size overflows usize"))?;
        let k_buf = device
            .alloc_buffer(
                total_bytes,
                DType::F32,
                vec![num_kv_heads, capacity, head_dim],
            )
            .map_err(|e| anyhow!("alloc K cache: {e}"))?;
        let v_buf = device
            .alloc_buffer(
                total_bytes,
                DType::F32,
                vec![num_kv_heads, capacity, head_dim],
            )
            .map_err(|e| anyhow!("alloc V cache: {e}"))?;
        Ok(Self {
            k_buf,
            v_buf,
            num_kv_heads,
            capacity,
            head_dim,
            len: 0,
        })
    }

    /// Current valid length (number of populated slots from 0).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Reset to empty (positions [0, capacity) remain allocated but
    /// `len = 0`). Data not zeroed — append overwrites.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Append a single K/V row at position `len`. The caller supplies
    /// row data in `[num_kv_heads, head_dim]` layout (one row per head).
    ///
    /// Per-head storage in the cache: row gets written at
    /// `cache[head, len, :] = row[head, :]` for each head.
    pub fn append(&mut self, k_row: &[f32], v_row: &[f32]) -> Result<()> {
        ensure!(
            self.len < self.capacity,
            "DrafterKvCache::append: cache full (len={}, capacity={})",
            self.len,
            self.capacity
        );
        let expected = self.num_kv_heads * self.head_dim;
        ensure!(
            k_row.len() == expected,
            "DrafterKvCache::append: k_row has {} elements, expected {} (num_kv_heads {} * head_dim {})",
            k_row.len(),
            expected,
            self.num_kv_heads,
            self.head_dim
        );
        ensure!(
            v_row.len() == expected,
            "DrafterKvCache::append: v_row has {} elements, expected {}",
            v_row.len(),
            expected
        );
        // Write per-head into [head, len, :] cells. Layout is
        // [num_kv_heads, capacity, head_dim] row-major, so the
        // (head, position, dim) offset is `head * capacity * head_dim
        // + position * head_dim + dim`.
        let k_slice = self
            .k_buf
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("k_buf slice: {e}"))?;
        let v_slice = self
            .v_buf
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("v_buf slice: {e}"))?;
        for h in 0..self.num_kv_heads {
            let cache_offset = h * self.capacity * self.head_dim + self.len * self.head_dim;
            let row_offset = h * self.head_dim;
            k_slice[cache_offset..cache_offset + self.head_dim]
                .copy_from_slice(&k_row[row_offset..row_offset + self.head_dim]);
            v_slice[cache_offset..cache_offset + self.head_dim]
                .copy_from_slice(&v_row[row_offset..row_offset + self.head_dim]);
        }
        self.len += 1;
        Ok(())
    }

    /// Rollback: keep only positions in `accepted` (in their accepted
    /// order). After this, `len = accepted.len()`. Positions
    /// `[len, capacity)` may contain stale data and must NOT be
    /// read by downstream consumers.
    ///
    /// Per Phase E5a `walk_tree_accept` contract:
    /// - `accepted[0]` is always root (index 0)
    /// - all indices in `accepted` are < pre-rollback `self.len`
    /// - indices may be out of order (the walk descends; ancestors
    ///   appear before descendants but they may not be index-ordered)
    ///
    /// Implementation: download K/V to CPU, build new sequences from
    /// accepted indices in order, upload to positions [0, accepted.len()).
    pub fn rollback_to_accepted(&mut self, accepted: &[usize]) -> Result<()> {
        ensure!(
            !accepted.is_empty(),
            "DrafterKvCache::rollback_to_accepted: accepted must be non-empty (root always included)"
        );
        ensure!(
            accepted.len() <= self.capacity,
            "DrafterKvCache::rollback_to_accepted: accepted len {} > capacity {}",
            accepted.len(),
            self.capacity
        );
        // Validate all indices fit current len.
        for (i, &idx) in accepted.iter().enumerate() {
            ensure!(
                idx < self.len,
                "DrafterKvCache::rollback_to_accepted: accepted[{}] = {} >= current len {}",
                i,
                idx,
                self.len
            );
        }
        // Detect duplicates (would corrupt the cache by writing
        // the same slot's data twice).
        let mut seen = std::collections::HashSet::with_capacity(accepted.len());
        for (i, &idx) in accepted.iter().enumerate() {
            ensure!(
                seen.insert(idx),
                "DrafterKvCache::rollback_to_accepted: duplicate index {} at position {}",
                idx,
                i
            );
        }
        // Download → reorder → upload.
        let k_data = self
            .k_buf
            .as_slice::<f32>()
            .map_err(|e| anyhow!("k_buf slice: {e}"))?
            .to_vec();
        let v_data = self
            .v_buf
            .as_slice::<f32>()
            .map_err(|e| anyhow!("v_buf slice: {e}"))?
            .to_vec();
        let stride_per_head = self.capacity * self.head_dim;
        let new_len = accepted.len();
        // Build the new K/V data per head, in accepted order.
        let mut new_k = vec![0.0f32; self.num_kv_heads * stride_per_head];
        let mut new_v = vec![0.0f32; self.num_kv_heads * stride_per_head];
        // First copy the unchanged tail (positions [new_len, capacity)
        // stay at whatever they were; only [0, new_len) gets rewritten).
        for h in 0..self.num_kv_heads {
            new_k[h * stride_per_head..(h + 1) * stride_per_head]
                .copy_from_slice(&k_data[h * stride_per_head..(h + 1) * stride_per_head]);
            new_v[h * stride_per_head..(h + 1) * stride_per_head]
                .copy_from_slice(&v_data[h * stride_per_head..(h + 1) * stride_per_head]);
        }
        // Now overwrite positions [0, new_len) with accepted source data.
        for (new_pos, &src_idx) in accepted.iter().enumerate() {
            for h in 0..self.num_kv_heads {
                let src_offset = h * stride_per_head + src_idx * self.head_dim;
                let dst_offset = h * stride_per_head + new_pos * self.head_dim;
                new_k[dst_offset..dst_offset + self.head_dim]
                    .copy_from_slice(&k_data[src_offset..src_offset + self.head_dim]);
                new_v[dst_offset..dst_offset + self.head_dim]
                    .copy_from_slice(&v_data[src_offset..src_offset + self.head_dim]);
            }
        }
        // Upload back.
        self.k_buf
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("k_buf mut slice: {e}"))?
            .copy_from_slice(&new_k);
        self.v_buf
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("v_buf mut slice: {e}"))?
            .copy_from_slice(&new_v);
        self.len = new_len;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn make_cache() -> Option<(MlxDevice, DrafterKvCache)> {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return None,
        };
        let cache = DrafterKvCache::new(&device, 2, 8, 4).expect("alloc");
        Some((device, cache))
    }

    /// Build a sentinel K/V row where the values encode (head, dim, tag).
    /// Per-head value: row[head][dim] = tag*1000 + head*100 + dim.
    fn sentinel_row(num_kv_heads: usize, head_dim: usize, tag: u32) -> Vec<f32> {
        let mut out = vec![0.0f32; num_kv_heads * head_dim];
        for h in 0..num_kv_heads {
            for d in 0..head_dim {
                out[h * head_dim + d] = (tag * 1000 + h as u32 * 100 + d as u32) as f32;
            }
        }
        out
    }

    #[test]
    fn adr_037_e5b_kv_cache_constructor_validates_dims_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        assert!(DrafterKvCache::new(&device, 0, 8, 4).is_err());
        assert!(DrafterKvCache::new(&device, 2, 0, 4).is_err());
        assert!(DrafterKvCache::new(&device, 2, 8, 0).is_err());
    }

    #[test]
    fn adr_037_e5b_kv_cache_initial_state_empty_2026_05_22() {
        let (_dev, cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.num_kv_heads, 2);
        assert_eq!(cache.capacity, 8);
        assert_eq!(cache.head_dim, 4);
    }

    #[test]
    fn adr_037_e5b_kv_cache_append_grows_len_2026_05_22() {
        let (_dev, mut cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        for tag in 0..5 {
            let k = sentinel_row(cache.num_kv_heads, cache.head_dim, tag);
            let v = sentinel_row(cache.num_kv_heads, cache.head_dim, tag + 100);
            cache.append(&k, &v).expect("append");
            assert_eq!(cache.len(), (tag + 1) as usize);
        }
    }

    #[test]
    fn adr_037_e5b_kv_cache_append_rejects_full_2026_05_22() {
        let (_dev, mut cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        let k = sentinel_row(2, 4, 0);
        let v = sentinel_row(2, 4, 0);
        for _ in 0..8 {
            cache.append(&k, &v).expect("append within capacity");
        }
        let err = cache.append(&k, &v).unwrap_err();
        assert!(err.to_string().contains("cache full"), "got: {err}");
    }

    #[test]
    fn adr_037_e5b_kv_cache_append_rejects_wrong_row_shape_2026_05_22() {
        let (_dev, mut cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        let bad_k = vec![0.0f32; 5]; // expected 2*4=8
        let v = sentinel_row(2, 4, 0);
        let err = cache.append(&bad_k, &v).unwrap_err();
        assert!(err.to_string().contains("k_row has 5"), "got: {err}");
    }

    #[test]
    fn adr_037_e5b_kv_cache_rollback_keeps_only_accepted_2026_05_22() {
        let (_dev, mut cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        // Append 5 sentinel-tagged rows. Tags 0..5.
        for tag in 0..5 {
            let k = sentinel_row(cache.num_kv_heads, cache.head_dim, tag);
            let v = sentinel_row(cache.num_kv_heads, cache.head_dim, tag + 100);
            cache.append(&k, &v).expect("append");
        }
        // Roll back to indices [0, 2, 4]. Expected new len = 3,
        // and positions [0, 1, 2] contain data from src 0, 2, 4.
        cache.rollback_to_accepted(&[0, 2, 4]).expect("rollback");
        assert_eq!(cache.len(), 3);
        // Verify content at positions [0, 1, 2].
        let k_data: Vec<f32> = cache.k_buf.as_slice::<f32>().unwrap().to_vec();
        let stride_per_head = cache.capacity * cache.head_dim;
        for (new_pos, &src_tag) in [0_u32, 2, 4].iter().enumerate() {
            for h in 0..cache.num_kv_heads {
                for d in 0..cache.head_dim {
                    let offset = h * stride_per_head + new_pos * cache.head_dim + d;
                    let expected =
                        (src_tag * 1000 + h as u32 * 100 + d as u32) as f32;
                    assert_eq!(
                        k_data[offset], expected,
                        "rollback pos {} head {} dim {} expected tag {} got {}",
                        new_pos, h, d, src_tag, k_data[offset]
                    );
                }
            }
        }
    }

    #[test]
    fn adr_037_e5b_kv_cache_rollback_rejects_empty_accepted_2026_05_22() {
        let (_dev, mut cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        let k = sentinel_row(2, 4, 0);
        let v = sentinel_row(2, 4, 0);
        cache.append(&k, &v).expect("append");
        let err = cache.rollback_to_accepted(&[]).unwrap_err();
        assert!(err.to_string().contains("must be non-empty"), "got: {err}");
    }

    #[test]
    fn adr_037_e5b_kv_cache_rollback_rejects_out_of_range_idx_2026_05_22() {
        let (_dev, mut cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        let k = sentinel_row(2, 4, 0);
        let v = sentinel_row(2, 4, 0);
        cache.append(&k, &v).expect("append"); // len = 1
        let err = cache.rollback_to_accepted(&[0, 5]).unwrap_err();
        assert!(
            err.to_string().contains(">= current len"),
            "got: {err}"
        );
    }

    #[test]
    fn adr_037_e5b_kv_cache_rollback_rejects_duplicate_idx_2026_05_22() {
        let (_dev, mut cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        let k = sentinel_row(2, 4, 0);
        let v = sentinel_row(2, 4, 0);
        for _ in 0..3 {
            cache.append(&k, &v).expect("append");
        }
        let err = cache.rollback_to_accepted(&[0, 1, 1]).unwrap_err();
        assert!(err.to_string().contains("duplicate index"), "got: {err}");
    }

    #[test]
    fn adr_037_e5b_kv_cache_rollback_to_root_only_2026_05_22() {
        let (_dev, mut cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        for tag in 0..4 {
            let k = sentinel_row(cache.num_kv_heads, cache.head_dim, tag);
            let v = sentinel_row(cache.num_kv_heads, cache.head_dim, tag + 100);
            cache.append(&k, &v).expect("append");
        }
        // Tree-walk-accept returns only [0] when verifier rejects
        // root's prediction. Rollback collapses cache to len=1.
        cache.rollback_to_accepted(&[0]).expect("rollback to root");
        assert_eq!(cache.len(), 1);
        // Verify position 0 has tag-0 data unchanged.
        let k_data = cache.k_buf.as_slice::<f32>().unwrap();
        let stride_per_head = cache.capacity * cache.head_dim;
        for h in 0..cache.num_kv_heads {
            for d in 0..cache.head_dim {
                let offset = h * stride_per_head + d;
                let expected = (0_u32 * 1000 + h as u32 * 100 + d as u32) as f32;
                assert_eq!(k_data[offset], expected);
            }
        }
    }

    #[test]
    fn adr_037_e5b_kv_cache_clear_resets_len_2026_05_22() {
        let (_dev, mut cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        let k = sentinel_row(2, 4, 0);
        let v = sentinel_row(2, 4, 0);
        cache.append(&k, &v).expect("append");
        cache.append(&k, &v).expect("append");
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn adr_037_e5b_kv_cache_integration_with_tree_walk_accept_2026_05_22() {
        // Build a 4-node tree + verifier argmax → walk → rollback.
        use crate::inference::spec_decode::eagle3::dynamic_tree::ExpandedTree;
        use crate::inference::spec_decode::eagle3::tree_walk::walk_tree_accept;

        let (_dev, mut cache) = match make_cache() {
            Some(t) => t,
            None => return,
        };
        // Append 4 sentinel rows for a 4-node tree (root + 3 descendants).
        for tag in 0..4 {
            let k = sentinel_row(cache.num_kv_heads, cache.head_dim, tag);
            let v = sentinel_row(cache.num_kv_heads, cache.head_dim, tag + 100);
            cache.append(&k, &v).expect("append");
        }
        // Tree: 0 → 1 → 2 (chain) + 0 → 3 (sibling of 1).
        // Verifier: root→token1 = 1 (match node 1), node1→token2 = 2 (match node 2).
        // Walk: [0, 1, 2].
        let tree = ExpandedTree {
            tokens: vec![100, 1, 2, 3],
            parents: vec![None, Some(0), Some(1), Some(0)],
            depths: vec![0, 1, 2, 1],
            cum_log_probs: vec![0.0, -0.1, -0.2, -0.5],
        };
        let argmax = vec![1_u32, 2, 0, 0];
        let accepted = walk_tree_accept(&tree, &argmax).expect("walk");
        assert_eq!(accepted, vec![0, 1, 2]);
        // Rollback cache to keep accepted positions.
        cache.rollback_to_accepted(&accepted).expect("rollback");
        assert_eq!(cache.len(), 3);
        // Cache positions [0, 1, 2] should contain tags 0, 1, 2 (root, node1, node2).
        let k_data = cache.k_buf.as_slice::<f32>().unwrap();
        let stride_per_head = cache.capacity * cache.head_dim;
        for (new_pos, expected_tag) in [0_u32, 1, 2].iter().enumerate() {
            for h in 0..cache.num_kv_heads {
                let offset = h * stride_per_head + new_pos * cache.head_dim;
                let expected =
                    (expected_tag * 1000 + h as u32 * 100) as f32;
                assert_eq!(k_data[offset], expected);
            }
        }
    }
}
