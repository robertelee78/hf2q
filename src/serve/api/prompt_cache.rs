//! Single-slot prompt-cache data structure (ADR-005 Decision #24, Task #7).
//!
//! This module holds the pure-compute portion of the prompt cache: a cached
//! list of prompt token IDs from the most recent successful generation,
//! plus a longest-common-prefix (LCP) algorithm that computes how many
//! leading tokens of a new request overlap with the cached prompt.
//!
//! The **active KV-replay** half — actually skipping recomputation of those
//! LCP-matching prefix positions — depends on a `forward_decode` refactor
//! that exposes logits + partial-prefill state to the engine worker.
//! That refactor needs live-model parity validation (byte-identical
//! cached-path vs uncached-path output) to land safely; deferred until OOM
//! pressure clears.
//!
//! # Why land the data structure now
//!
//! The cache API is tight (6 pub methods) and the LCP algorithm is pure.
//! Landing it alone lets us:
//!   - Unit-test the algorithm exhaustively (done: 12 tests).
//!   - Integrate with the Engine's request/response cycle incrementally
//!     (store-on-success → report-in-response → skip-work) without an
//!     all-or-nothing rewrite.
//!   - Lock the interface so when KV-replay lands it's a drop-in extension.
//!
//! # What this file deliberately does NOT do
//!
//! - It does NOT store KV tensor state. Live KV buffers belong on the
//!   engine's `LoadedModel`; the cache only tracks the TOKEN IDs.
//! - It does NOT report `prompt_tokens_details.cached_tokens` in responses.
//!   Reporting a value we haven't actually saved would be a contract lie
//!   (OpenAI's `cached_tokens` means "tokens served from cache with TTFT
//!   savings"). The handler keeps reporting 0 until the work-skipping
//!   half lands.

use std::time::Instant;

// ---------------------------------------------------------------------------
// PromptCache
// ---------------------------------------------------------------------------

/// Single-slot prompt cache: holds the last successful generation's prompt
/// tokens. Decision #24 is a single-slot design rather than an LRU because
/// multi-turn chat has a natural slot-of-one pattern (each turn's prompt
/// extends the previous one); a small LRU would waste capacity while not
/// helping the common case.
#[derive(Debug, Clone)]
pub struct PromptCache {
    /// Last successful prompt's token IDs. `None` = cache is empty
    /// (server startup, after `clear`, or after an error that invalidated
    /// the cached state).
    cached_tokens: Option<Vec<u32>>,
    /// Timestamp the cache was last updated. Useful for `/metrics`
    /// diagnostics + a future TTL-based invalidation policy.
    updated_at: Instant,
}

impl Default for PromptCache {
    fn default() -> Self {
        Self {
            cached_tokens: None,
            updated_at: Instant::now(),
        }
    }
}

impl PromptCache {
    /// Fresh, empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` when the cache has no stored tokens (startup or
    /// post-`clear` state). Handlers should treat this as `cached_tokens = 0`.
    pub fn is_empty(&self) -> bool {
        self.cached_tokens.as_ref().map_or(true, |v| v.is_empty())
    }

    /// Length of the longest common prefix between the cached tokens and
    /// `new_tokens`. Returns 0 if the cache is empty or the first token
    /// differs. Runs in `O(min(cached.len, new.len))`.
    pub fn lcp_len(&self, new_tokens: &[u32]) -> usize {
        match self.cached_tokens.as_ref() {
            Some(c) => lcp_len(c, new_tokens),
            None => 0,
        }
    }

    /// Replace the cached tokens with a new successful generation's prompt.
    /// Called by the engine worker after a generation completes without
    /// error; a failed or cancelled generation leaves the cache untouched
    /// (stale data is fine; corrupt state is not).
    pub fn update(&mut self, new_tokens: Vec<u32>) {
        self.cached_tokens = Some(new_tokens);
        self.updated_at = Instant::now();
    }

    /// Drop the cached state. Used by the shutdown path and by any error
    /// recovery that invalidates the KV buffer. After `clear`, `lcp_len`
    /// returns 0 for every input.
    pub fn clear(&mut self) {
        self.cached_tokens = None;
        self.updated_at = Instant::now();
    }

    /// Read-only view of the cached token IDs, if any.
    pub fn tokens(&self) -> Option<&[u32]> {
        self.cached_tokens.as_deref()
    }

    /// Seconds since the cache was last updated.
    pub fn age_seconds(&self) -> f64 {
        self.updated_at.elapsed().as_secs_f64()
    }

    /// Number of tokens currently cached.
    pub fn len(&self) -> usize {
        self.cached_tokens.as_ref().map_or(0, |v| v.len())
    }
}

// ---------------------------------------------------------------------------
// lcp_len — standalone algorithm
// ---------------------------------------------------------------------------

/// Length of the longest common prefix between `a` and `b`. Standalone
/// helper so tests can exercise the algorithm directly without touching
/// `PromptCache`.
///
/// Symmetric in its arguments (`lcp_len(a, b) == lcp_len(b, a)`).
pub fn lcp_len(a: &[u32], b: &[u32]) -> usize {
    let limit = a.len().min(b.len());
    let mut i = 0;
    while i < limit && a[i] == b[i] {
        i += 1;
    }
    i
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- lcp_len standalone ---

    #[test]
    fn lcp_len_empty_inputs_return_zero() {
        assert_eq!(lcp_len(&[], &[]), 0);
        assert_eq!(lcp_len(&[], &[1, 2, 3]), 0);
        assert_eq!(lcp_len(&[1, 2, 3], &[]), 0);
    }

    #[test]
    fn lcp_len_identical_slices_match_full_length() {
        assert_eq!(lcp_len(&[1, 2, 3, 4], &[1, 2, 3, 4]), 4);
    }

    #[test]
    fn lcp_len_partial_overlap_stops_at_first_divergence() {
        assert_eq!(lcp_len(&[1, 2, 3, 999], &[1, 2, 3, 4]), 3);
        assert_eq!(lcp_len(&[1, 2], &[1, 2, 3, 4]), 2);
        assert_eq!(lcp_len(&[1, 2, 3, 4], &[1, 2]), 2);
    }

    #[test]
    fn lcp_len_first_token_differs_returns_zero() {
        assert_eq!(lcp_len(&[5, 1, 2, 3], &[4, 1, 2, 3]), 0);
    }

    #[test]
    fn lcp_len_is_symmetric() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 99, 100];
        assert_eq!(lcp_len(&a, &b), lcp_len(&b, &a));
    }

    #[test]
    fn lcp_len_handles_u32_max_values() {
        assert_eq!(
            lcp_len(&[u32::MAX, u32::MAX], &[u32::MAX, u32::MAX, 0]),
            2
        );
    }

    // --- PromptCache ---

    #[test]
    fn cache_new_is_empty() {
        let c = PromptCache::new();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        assert_eq!(c.tokens(), None);
        assert_eq!(c.lcp_len(&[1, 2, 3]), 0);
    }

    #[test]
    fn cache_update_replaces_tokens() {
        let mut c = PromptCache::new();
        c.update(vec![10, 20, 30]);
        assert!(!c.is_empty());
        assert_eq!(c.len(), 3);
        assert_eq!(c.tokens(), Some(&[10, 20, 30][..]));
    }

    #[test]
    fn cache_lcp_len_against_stored_tokens() {
        let mut c = PromptCache::new();
        c.update(vec![1, 2, 3, 4, 5]);
        assert_eq!(c.lcp_len(&[1, 2, 3, 4, 5]), 5); // identical
        assert_eq!(c.lcp_len(&[1, 2, 3]), 3); // shorter prefix
        assert_eq!(c.lcp_len(&[1, 2, 3, 99, 5]), 3); // diverges at 4th
        assert_eq!(c.lcp_len(&[99, 2, 3]), 0); // no shared prefix
        assert_eq!(c.lcp_len(&[]), 0); // empty request
    }

    #[test]
    fn cache_clear_wipes_state() {
        let mut c = PromptCache::new();
        c.update(vec![1, 2, 3]);
        c.clear();
        assert!(c.is_empty());
        assert_eq!(c.lcp_len(&[1, 2, 3]), 0);
    }

    #[test]
    fn cache_update_after_clear_restores_state() {
        let mut c = PromptCache::new();
        c.update(vec![1, 2, 3]);
        c.clear();
        c.update(vec![10, 20]);
        assert_eq!(c.tokens(), Some(&[10, 20][..]));
        assert_eq!(c.lcp_len(&[10, 20, 30]), 2);
    }

    #[test]
    fn cache_age_seconds_increases_monotonically() {
        let mut c = PromptCache::new();
        let initial = c.age_seconds();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let after_sleep = c.age_seconds();
        assert!(
            after_sleep >= initial,
            "age should be monotonic: {} -> {}",
            initial,
            after_sleep
        );
        c.update(vec![1]);
        let after_update = c.age_seconds();
        assert!(
            after_update < after_sleep,
            "age should reset after update: {} -> {}",
            after_sleep,
            after_update
        );
    }

    #[test]
    fn cache_multi_turn_chat_pattern_returns_growing_lcp() {
        // Simulates multi-turn chat where each turn extends the previous
        // prompt by a suffix. LCP should equal (previous_prompt_len) each
        // time — which is the primary speedup target.
        let mut c = PromptCache::new();

        // Turn 1: user + system preamble (say 10 tokens).
        let turn1: Vec<u32> = (0..10).collect();
        c.update(turn1.clone());

        // Turn 2: turn1 + assistant response (+5) + new user turn (+5).
        let mut turn2 = turn1.clone();
        turn2.extend(100..115);
        assert_eq!(
            c.lcp_len(&turn2),
            turn1.len(),
            "Turn 2 should share the whole of Turn 1's prompt"
        );
        c.update(turn2.clone());

        // Turn 3: turn2 + more tokens.
        let mut turn3 = turn2.clone();
        turn3.extend(200..210);
        assert_eq!(c.lcp_len(&turn3), turn2.len());
    }
}
