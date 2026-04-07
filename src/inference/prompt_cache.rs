//! Prompt caching with prefix matching for multi-turn conversations.
//!
//! Stores the token sequence and KV cache state from previous requests so that
//! follow-up messages that share a common prefix can skip re-encoding the
//! shared portion. This dramatically reduces time-to-first-token (TTFT) for
//! multi-turn chat, where each new turn contains the entire conversation
//! history plus the new user message.
//!
//! ## Design
//!
//! - **Single-entry default**: Since generation is serialized (one request at a
//!   time via the generation queue), a single cache entry suffices. The LRU
//!   design supports `max_entries > 1` for future multi-user scenarios.
//! - **Prefix matching**: Linear O(min(old, new)) scan comparing u32 token IDs.
//!   Even for 8K-token conversations this completes in microseconds.
//! - **KV cache reuse**: The KV cache buffers are pre-allocated and reused
//!   in-place. The prompt cache stores only the logical positions (not copies
//!   of the buffer data), and the engine restores write positions on cache hit.
//! - **Vision incompatibility**: Requests containing vision tokens are tagged
//!   and cannot reuse a text-only cache entry (or vice versa), since vision
//!   token injection changes the embedding sequence.

use std::time::Instant;

use tracing::{debug, info};

/// Configuration for the prompt cache.
#[derive(Debug, Clone)]
pub struct PromptCacheConfig {
    /// Maximum number of cache entries to retain.
    ///
    /// Default is 1 since generation is serialized. Set higher for future
    /// multi-user scenarios where different conversations might interleave.
    pub max_entries: usize,

    /// Maximum total memory (in bytes) that cached KV state may consume.
    ///
    /// When exceeded, the least-recently-used entries are evicted until
    /// usage falls below this threshold. Set to 0 to disable memory-based
    /// eviction (rely only on `max_entries`).
    pub max_memory_bytes: usize,

    /// Whether prompt caching is enabled at all.
    ///
    /// When false, `lookup` always returns `CacheLookup::Miss` and `store`
    /// is a no-op. Useful as a `--no-prompt-cache` escape hatch.
    pub enabled: bool,
}

impl Default for PromptCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1,
            max_memory_bytes: 0, // no memory-based eviction by default
            enabled: true,
        }
    }
}

/// A single cache entry storing the token sequence and KV cache state
/// from a completed generation request.
#[derive(Debug)]
struct CacheEntry {
    /// The full token sequence that was encoded (prompt + generated tokens
    /// are NOT included; only the prompt tokens that went through prefill).
    tokens: Vec<u32>,

    /// The number of positions written to the KV cache at the time this
    /// entry was stored. Used to restore the cache write positions on hit.
    ///
    /// This is the `total_written` value that was uniform across all layers
    /// at the end of the prefill + decode cycle. Note: for the prompt cache
    /// we store the state after *prefill* of all prompt tokens (before decode),
    /// so `kv_position == tokens.len()`.
    kv_position: usize,

    /// Whether the original request included vision/multimodal tokens.
    has_vision: bool,

    /// Timestamp of last access (for LRU eviction).
    last_used: Instant,

    /// Approximate byte cost of the KV cache data this entry represents.
    ///
    /// This is the total allocated KV cache size (all layers), stored once
    /// at entry creation time. Used for memory-based eviction decisions.
    kv_byte_size: usize,
}

/// Result of looking up a new token sequence against the prompt cache.
#[derive(Debug)]
pub enum CacheLookup {
    /// No usable prefix was found. The caller should encode all tokens
    /// from scratch (standard path).
    Miss,

    /// A prefix of length `prefix_len` matched the cached entry. The KV
    /// cache already holds valid data for positions `0..prefix_len`.
    ///
    /// The caller should:
    /// 1. Restore the KV cache positions to `prefix_len`
    /// 2. Encode only `new_tokens[prefix_len..]` through the forward pass
    Hit {
        /// Length of the matching prefix (in tokens).
        prefix_len: usize,

        /// The KV cache position to restore to (equals `prefix_len`).
        kv_position: usize,
    },
}

/// LRU prompt cache supporting prefix matching and KV cache state tracking.
pub struct PromptCache {
    /// Cache entries ordered by insertion (newest last for simple LRU).
    entries: Vec<CacheEntry>,

    /// Configuration.
    config: PromptCacheConfig,

    /// Running count of cache hits (for stats/logging).
    hit_count: u64,

    /// Running count of cache misses.
    miss_count: u64,
}

impl PromptCache {
    /// Create a new prompt cache with the given configuration.
    pub fn new(config: PromptCacheConfig) -> Self {
        info!(
            enabled = config.enabled,
            max_entries = config.max_entries,
            max_memory_bytes = config.max_memory_bytes,
            "Prompt cache initialized"
        );
        Self {
            entries: Vec::with_capacity(config.max_entries),
            config,
            hit_count: 0,
            miss_count: 0,
        }
    }

    /// Look up a new token sequence against the cache.
    ///
    /// Returns `CacheLookup::Hit` if a prefix match is found with length > 0,
    /// or `CacheLookup::Miss` otherwise.
    ///
    /// `has_vision` indicates whether the new request contains vision tokens.
    /// If the vision state differs from the cached entry, the cache is
    /// bypassed (returns Miss).
    pub fn lookup(&mut self, new_tokens: &[u32], has_vision: bool) -> CacheLookup {
        if !self.config.enabled || self.entries.is_empty() || new_tokens.is_empty() {
            self.miss_count += 1;
            return CacheLookup::Miss;
        }

        // Search entries for the best (longest) prefix match.
        // With max_entries=1 this is trivially the single entry.
        let mut best_idx: Option<usize> = None;
        let mut best_prefix_len: usize = 0;

        for (idx, entry) in self.entries.iter().enumerate() {
            // Vision incompatibility: skip if vision state differs
            if entry.has_vision != has_vision {
                debug!(
                    cached_vision = entry.has_vision,
                    request_vision = has_vision,
                    "Skipping cache entry: vision state mismatch"
                );
                continue;
            }

            let prefix_len = find_prefix_match(&entry.tokens, new_tokens);
            if prefix_len > best_prefix_len {
                best_prefix_len = prefix_len;
                best_idx = Some(idx);
            }
        }

        if let Some(idx) = best_idx {
            if best_prefix_len > 0 {
                // Update last_used for LRU
                self.entries[idx].last_used = Instant::now();
                self.hit_count += 1;

                // The kv_position to restore is the prefix length, but capped
                // at the entry's kv_position (can't restore beyond what was
                // actually computed). In practice prefix_len <= entry.kv_position
                // because the prefix match can't exceed the stored token count.
                let kv_position = best_prefix_len.min(self.entries[idx].kv_position);

                debug!(
                    prefix_len = best_prefix_len,
                    kv_position = kv_position,
                    new_tokens_len = new_tokens.len(),
                    tokens_to_encode = new_tokens.len() - best_prefix_len,
                    "Prompt cache hit"
                );

                return CacheLookup::Hit {
                    prefix_len: best_prefix_len,
                    kv_position,
                };
            }
        }

        self.miss_count += 1;
        debug!(
            new_tokens_len = new_tokens.len(),
            num_entries = self.entries.len(),
            "Prompt cache miss"
        );
        CacheLookup::Miss
    }

    /// Store a completed request's token sequence and KV cache state.
    ///
    /// If the cache is full (`max_entries` reached), the least-recently-used
    /// entry is evicted first. If `max_memory_bytes > 0`, entries are also
    /// evicted to stay under the memory budget.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The full prompt token sequence that was prefilled.
    /// * `kv_position` - The KV cache position after prefill (== tokens.len()).
    /// * `has_vision` - Whether the request included vision tokens.
    /// * `kv_byte_size` - Approximate byte cost of the KV cache data.
    pub fn store(
        &mut self,
        tokens: Vec<u32>,
        kv_position: usize,
        has_vision: bool,
        kv_byte_size: usize,
    ) {
        if !self.config.enabled {
            return;
        }

        // Evict entries if at capacity
        while self.entries.len() >= self.config.max_entries {
            self.evict_lru();
        }

        // Memory-based eviction
        if self.config.max_memory_bytes > 0 {
            let current_memory: usize = self.entries.iter().map(|e| e.kv_byte_size).sum();
            let new_total = current_memory + kv_byte_size;
            if new_total > self.config.max_memory_bytes {
                // Evict LRU entries until we have room
                while !self.entries.is_empty() {
                    let remaining: usize = self.entries.iter().map(|e| e.kv_byte_size).sum();
                    if remaining + kv_byte_size <= self.config.max_memory_bytes {
                        break;
                    }
                    self.evict_lru();
                }
            }
        }

        let entry = CacheEntry {
            tokens,
            kv_position,
            has_vision,
            last_used: Instant::now(),
            kv_byte_size,
        };

        debug!(
            token_count = entry.tokens.len(),
            kv_position = entry.kv_position,
            has_vision = entry.has_vision,
            kv_bytes = entry.kv_byte_size,
            "Prompt cache: stored entry"
        );

        self.entries.push(entry);
    }

    /// Clear all cache entries.
    pub fn clear(&mut self) {
        let count = self.entries.len();
        self.entries.clear();
        if count > 0 {
            debug!(evicted = count, "Prompt cache cleared");
        }
    }

    /// Whether the cache is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Number of cache hits since creation.
    pub fn hit_count(&self) -> u64 {
        self.hit_count
    }

    /// Number of cache misses since creation.
    pub fn miss_count(&self) -> u64 {
        self.miss_count
    }

    /// Number of entries currently stored.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Evict the least-recently-used entry.
    fn evict_lru(&mut self) {
        if self.entries.is_empty() {
            return;
        }

        // Find the entry with the oldest last_used timestamp
        let lru_idx = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(idx, _)| idx)
            .unwrap(); // Safe: entries is non-empty

        let evicted = self.entries.remove(lru_idx);
        debug!(
            token_count = evicted.tokens.len(),
            kv_bytes = evicted.kv_byte_size,
            "Prompt cache: evicted LRU entry"
        );
    }
}

/// Find the length of the longest common prefix between two token sequences.
///
/// Performs a simple linear scan from position 0 comparing u32 token IDs.
/// Returns 0 if no tokens match (cache miss).
///
/// This is O(min(a.len(), b.len())) but operates on u32 integers, so even
/// for 8K-token conversations it completes in microseconds.
pub fn find_prefix_match(cached: &[u32], new: &[u32]) -> usize {
    let limit = cached.len().min(new.len());
    let mut i = 0;
    while i < limit && cached[i] == new[i] {
        i += 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // find_prefix_match unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefix_match_identical() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 4, 5];
        assert_eq!(find_prefix_match(&a, &b), 5);
    }

    #[test]
    fn test_prefix_match_partial_overlap() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 6, 7];
        assert_eq!(find_prefix_match(&a, &b), 3);
    }

    #[test]
    fn test_prefix_match_no_overlap() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        assert_eq!(find_prefix_match(&a, &b), 0);
    }

    #[test]
    fn test_prefix_match_empty_cached() {
        let a: Vec<u32> = vec![];
        let b = vec![1, 2, 3];
        assert_eq!(find_prefix_match(&a, &b), 0);
    }

    #[test]
    fn test_prefix_match_empty_new() {
        let a = vec![1, 2, 3];
        let b: Vec<u32> = vec![];
        assert_eq!(find_prefix_match(&a, &b), 0);
    }

    #[test]
    fn test_prefix_match_both_empty() {
        let a: Vec<u32> = vec![];
        let b: Vec<u32> = vec![];
        assert_eq!(find_prefix_match(&a, &b), 0);
    }

    #[test]
    fn test_prefix_match_new_is_strict_prefix_of_cached() {
        // New sequence is a prefix of cached -- full match up to new's length
        let cached = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let new = vec![1, 2, 3, 4, 5];
        assert_eq!(find_prefix_match(&cached, &new), 5);
    }

    #[test]
    fn test_prefix_match_cached_is_strict_prefix_of_new() {
        // Cached is a prefix of new (common in multi-turn chat)
        let cached = vec![1, 2, 3, 4, 5];
        let new = vec![1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(find_prefix_match(&cached, &new), 5);
    }

    #[test]
    fn test_prefix_match_single_token_match() {
        let a = vec![42, 100, 200];
        let b = vec![42, 999, 888];
        assert_eq!(find_prefix_match(&a, &b), 1);
    }

    #[test]
    fn test_prefix_match_divergence_at_position_zero() {
        let a = vec![1, 2, 3];
        let b = vec![99, 2, 3];
        assert_eq!(find_prefix_match(&a, &b), 0);
    }

    // -----------------------------------------------------------------------
    // PromptCache unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_disabled() {
        let config = PromptCacheConfig {
            enabled: false,
            ..Default::default()
        };
        let mut cache = PromptCache::new(config);

        // Store should be a no-op
        cache.store(vec![1, 2, 3], 3, false, 1000);
        assert_eq!(cache.entry_count(), 0);

        // Lookup should always miss
        let result = cache.lookup(&[1, 2, 3], false);
        assert!(matches!(result, CacheLookup::Miss));
    }

    #[test]
    fn test_cache_store_and_hit() {
        let mut cache = PromptCache::new(PromptCacheConfig::default());

        // Store a 5-token entry
        cache.store(vec![10, 20, 30, 40, 50], 5, false, 1000);
        assert_eq!(cache.entry_count(), 1);

        // Lookup with the same prefix + new tokens
        let result = cache.lookup(&[10, 20, 30, 40, 50, 60, 70], false);
        match result {
            CacheLookup::Hit {
                prefix_len,
                kv_position,
            } => {
                assert_eq!(prefix_len, 5);
                assert_eq!(kv_position, 5);
            }
            CacheLookup::Miss => panic!("Expected cache hit"),
        }
        assert_eq!(cache.hit_count(), 1);
        assert_eq!(cache.miss_count(), 0);
    }

    #[test]
    fn test_cache_miss_no_common_prefix() {
        let mut cache = PromptCache::new(PromptCacheConfig::default());

        cache.store(vec![1, 2, 3, 4, 5], 5, false, 1000);

        let result = cache.lookup(&[99, 98, 97], false);
        assert!(matches!(result, CacheLookup::Miss));
        assert_eq!(cache.miss_count(), 1);
    }

    #[test]
    fn test_cache_divergence_handling() {
        let mut cache = PromptCache::new(PromptCacheConfig::default());

        cache.store(vec![1, 2, 3, 4, 5], 5, false, 1000);

        // Diverges at position 2
        let result = cache.lookup(&[1, 2, 99, 100, 101], false);
        match result {
            CacheLookup::Hit {
                prefix_len,
                kv_position,
            } => {
                assert_eq!(prefix_len, 2);
                assert_eq!(kv_position, 2);
            }
            CacheLookup::Miss => panic!("Expected partial cache hit at position 2"),
        }
    }

    #[test]
    fn test_cache_lru_eviction_max_entries() {
        let config = PromptCacheConfig {
            max_entries: 1,
            max_memory_bytes: 0,
            enabled: true,
        };
        let mut cache = PromptCache::new(config);

        // Store first entry
        cache.store(vec![1, 2, 3], 3, false, 1000);
        assert_eq!(cache.entry_count(), 1);

        // Store second entry -- should evict the first
        cache.store(vec![10, 20, 30], 3, false, 1000);
        assert_eq!(cache.entry_count(), 1);

        // The old entry should be gone
        let result = cache.lookup(&[1, 2, 3, 4], false);
        assert!(matches!(result, CacheLookup::Miss));

        // The new entry should be present
        let result = cache.lookup(&[10, 20, 30, 40], false);
        assert!(matches!(result, CacheLookup::Hit { prefix_len: 3, .. }));
    }

    #[test]
    fn test_cache_memory_eviction() {
        let config = PromptCacheConfig {
            max_entries: 10,
            max_memory_bytes: 2000,
            enabled: true,
        };
        let mut cache = PromptCache::new(config);

        // Store an entry consuming 1500 bytes
        cache.store(vec![1, 2, 3], 3, false, 1500);
        assert_eq!(cache.entry_count(), 1);

        // Store another consuming 1500 bytes -- should evict the first to
        // stay under 2000 byte budget
        cache.store(vec![4, 5, 6], 3, false, 1500);
        assert_eq!(cache.entry_count(), 1);

        // Only the second entry should remain
        let result = cache.lookup(&[1, 2, 3, 4], false);
        assert!(matches!(result, CacheLookup::Miss));

        let result = cache.lookup(&[4, 5, 6, 7], false);
        assert!(matches!(result, CacheLookup::Hit { prefix_len: 3, .. }));
    }

    #[test]
    fn test_cache_vision_incompatibility() {
        let mut cache = PromptCache::new(PromptCacheConfig::default());

        // Store a text-only entry
        cache.store(vec![1, 2, 3, 4, 5], 5, false, 1000);

        // Lookup with vision tokens -- should miss despite matching prefix
        let result = cache.lookup(&[1, 2, 3, 4, 5, 6], true);
        assert!(matches!(result, CacheLookup::Miss));

        // Lookup without vision -- should hit
        let result = cache.lookup(&[1, 2, 3, 4, 5, 6], false);
        assert!(matches!(result, CacheLookup::Hit { prefix_len: 5, .. }));
    }

    #[test]
    fn test_cache_vision_compatible() {
        let mut cache = PromptCache::new(PromptCacheConfig::default());

        // Store a vision entry
        cache.store(vec![1, 2, 3, 4, 5], 5, true, 1000);

        // Lookup with vision -- should hit
        let result = cache.lookup(&[1, 2, 3, 4, 5, 6], true);
        assert!(matches!(result, CacheLookup::Hit { prefix_len: 5, .. }));

        // Lookup without vision -- should miss
        let result = cache.lookup(&[1, 2, 3, 4, 5, 6], false);
        assert!(matches!(result, CacheLookup::Miss));
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = PromptCache::new(PromptCacheConfig::default());

        cache.store(vec![1, 2, 3], 3, false, 1000);
        assert_eq!(cache.entry_count(), 1);

        cache.clear();
        assert_eq!(cache.entry_count(), 0);

        let result = cache.lookup(&[1, 2, 3, 4], false);
        assert!(matches!(result, CacheLookup::Miss));
    }

    #[test]
    fn test_cache_lookup_empty_tokens() {
        let mut cache = PromptCache::new(PromptCacheConfig::default());
        cache.store(vec![1, 2, 3], 3, false, 1000);

        // Empty lookup should miss
        let result = cache.lookup(&[], false);
        assert!(matches!(result, CacheLookup::Miss));
    }

    #[test]
    fn test_cache_identical_request() {
        let mut cache = PromptCache::new(PromptCacheConfig::default());

        cache.store(vec![1, 2, 3, 4, 5], 5, false, 1000);

        // Identical request -- full prefix match
        let result = cache.lookup(&[1, 2, 3, 4, 5], false);
        match result {
            CacheLookup::Hit {
                prefix_len,
                kv_position,
            } => {
                assert_eq!(prefix_len, 5);
                assert_eq!(kv_position, 5);
            }
            CacheLookup::Miss => panic!("Expected full cache hit"),
        }
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = PromptCache::new(PromptCacheConfig::default());

        assert_eq!(cache.hit_count(), 0);
        assert_eq!(cache.miss_count(), 0);

        cache.store(vec![1, 2, 3], 3, false, 1000);

        // Miss
        let _ = cache.lookup(&[99, 98], false);
        assert_eq!(cache.miss_count(), 1);

        // Hit
        let _ = cache.lookup(&[1, 2, 3, 4], false);
        assert_eq!(cache.hit_count(), 1);

        // Another hit
        let _ = cache.lookup(&[1, 2, 3, 4, 5], false);
        assert_eq!(cache.hit_count(), 2);
    }

    #[test]
    fn test_cache_lru_updates_on_hit() {
        let config = PromptCacheConfig {
            max_entries: 2,
            max_memory_bytes: 0,
            enabled: true,
        };
        let mut cache = PromptCache::new(config);

        // Store two entries
        cache.store(vec![1, 2, 3], 3, false, 1000);
        // Small delay to ensure different timestamps
        std::thread::sleep(std::time::Duration::from_millis(1));
        cache.store(vec![10, 20, 30], 3, false, 1000);

        // Access the first entry (updates its last_used)
        let _ = cache.lookup(&[1, 2, 3, 4], false);

        // Now store a third entry -- should evict the second (LRU)
        cache.store(vec![100, 200, 300], 3, false, 1000);
        assert_eq!(cache.entry_count(), 2);

        // First entry should still be present (was accessed more recently)
        let result = cache.lookup(&[1, 2, 3, 4], false);
        assert!(matches!(result, CacheLookup::Hit { prefix_len: 3, .. }));

        // Second entry should have been evicted
        let result = cache.lookup(&[10, 20, 30, 40], false);
        assert!(matches!(result, CacheLookup::Miss));
    }
}
