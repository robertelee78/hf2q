//! N-gram speculative decode proposer (ADR-029 Phase 1).
//!
//! Pure CPU KMP-style longest-prefix-suffix matching. Proposes K
//! draft tokens by:
//! 1. Reversing the token sequence (suffix-match becomes prefix-match).
//! 2. KMP scan to find the longest prefix-of-tokens that is also a
//!    suffix at some earlier position in the reversed view.
//! 3. The K tokens immediately following that earlier occurrence in
//!    the original sequence are the draft.
//!
//! Bit-faithful Rust port of
//! `/opt/vllm/vllm/v1/spec_decode/ngram_proposer.py`
//! `_find_longest_matched_ngram_and_propose_tokens` (lines 198-285,
//! commit-pinned 2026-05-09).
//!
//! No proposal is committed without target verification. The legacy KMP
//! proposer remains available to CLI speculative decode; the OpenAI Qwen
//! server uses the request-owned [`HistoryLookupIndex`] under
//! `HF2Q_QWEN_SPECULATION=auto`, with an independent measured cost gate.

use std::collections::HashMap;

/// Configuration for the n-gram proposer.
#[derive(Debug, Clone, Copy)]
pub struct NgramConfig {
    /// Minimum n-gram length to match (inclusive).
    pub min_ngram: usize,
    /// Maximum n-gram length to match (inclusive). Caps the LPS table size.
    pub max_ngram: usize,
    /// Number of draft tokens to propose after the matched n-gram.
    pub k: usize,
    /// Maximum model context length (drafts truncated so we never propose
    /// past `max_model_len`).
    pub max_model_len: usize,
}

/// Request-history lookup configuration for long-context assistant traffic.
///
/// This is deliberately separate from [`NgramConfig`].  The legacy proposer
/// above reproduces vLLM's short n-gram algorithm, including its historical
/// tie-breaking.  Lookup drafting has a different contract: match a longer
/// suffix and copy the continuation from its *most recent* prior occurrence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoryLookupConfig {
    /// Shortest suffix considered useful enough to draft from.
    pub min_match: usize,
    /// Longest suffix examined. Longer matches win.
    pub max_match: usize,
    /// Maximum continuation length returned to the verifier.
    pub max_draft_tokens: usize,
    /// Maximum model context length.
    pub max_model_len: usize,
}

impl HistoryLookupConfig {
    /// Conservative Apple-Silicon starting point.
    ///
    /// The 6..=12 match range follows the useful long-context lookup regime.
    /// Three is the qualified fixed MTP depth and starting verifier-width
    /// ceiling. Any larger width requires a matched local correctness and
    /// throughput benchmark.
    pub fn default_for_decode(max_model_len: usize) -> Self {
        Self {
            min_match: 6,
            max_match: 12,
            max_draft_tokens: 3,
            max_model_len,
        }
    }
}

impl NgramConfig {
    /// Default per ADR-028 iter-99 vLLM/dflash literature (K=3 optimal,
    /// n-grams 1..3 covering most natural-language repetitions).
    pub fn default_for_decode(max_model_len: usize) -> Self {
        Self {
            min_ngram: 1,
            max_ngram: 3,
            k: 3,
            max_model_len,
        }
    }
}

/// Propose up to K draft tokens by finding the longest n-gram in
/// `[min_ngram, max_ngram]` that matches the suffix of `tokens` and
/// returning the K tokens that followed an earlier occurrence.
///
/// Returns an empty `Vec` when no valid n-gram exists or when the
/// sequence is at the model-length limit.
///
/// # Algorithm
///
/// Reverses `tokens` so suffix-matching becomes prefix-matching.
/// Then runs KMP's failure-function build to compute, for each
/// position `i`, the longest prefix of `reversed[..max_ngram]` that
/// is also a suffix of `reversed[..=i]`. The match with the largest
/// `prev_lps ≥ min_ngram` at the latest position in the reversed
/// view (== earliest position in the original view) wins.
///
/// # Complexity
///
/// `O(n)` time, `O(max_ngram)` memory for the LPS table.
pub fn propose(tokens: &[u32], cfg: &NgramConfig) -> Vec<u32> {
    let total = tokens.len();
    if total < cfg.min_ngram {
        return Vec::new();
    }

    // Cap K so we never propose past max_model_len.
    let k_room = cfg.max_model_len.saturating_sub(total);
    let k_capped = cfg.k.min(k_room);
    if k_capped == 0 {
        return Vec::new();
    }

    if cfg.max_ngram == 0 || cfg.min_ngram > cfg.max_ngram {
        return Vec::new();
    }

    // Work on the reversed sequence — suffix match becomes prefix match.
    // We don't materialize a reversed Vec; index from the right via
    // `rev_idx(i) = total - 1 - i`.
    let rev = |i: usize| -> u32 { tokens[total - 1 - i] };

    // LPS table: lps[i] = length of the longest proper prefix of
    // reversed[..max_ngram] that is also a suffix of reversed[..=i].
    // Capped at max_ngram entries — we only need the prefix tracking
    // up to max_ngram length.
    let lps_len = cfg.max_ngram;
    let mut lps = vec![0u32; lps_len];

    let mut longest_ngram: usize = 0;
    let mut position: usize = 0;
    let mut prev_lps: usize = 0;

    // lps[0] is always 0; iterate from i = 1.
    let mut i: usize = 1;
    while i < total {
        if rev(prev_lps) == rev(i) {
            // Token match: extend the current match.
            prev_lps += 1;
            // Update best-match record. `>=` (not `>`) so we keep the
            // EARLIEST occurrence in the original sequence (== latest
            // position in the reversed view), matching vLLM line 253.
            if prev_lps >= longest_ngram {
                longest_ngram = prev_lps;
                position = i;
            }
            if i < lps_len {
                lps[i] = prev_lps as u32;
            }
            if prev_lps == cfg.max_ngram {
                // Cap at max_ngram by jumping back via lps[max_ngram - 1].
                prev_lps = lps[cfg.max_ngram - 1] as usize;
            }
            i += 1;
        } else if prev_lps != 0 {
            // Mismatch: try second-longest prefix-suffix.
            prev_lps = lps[prev_lps - 1] as usize;
        } else {
            // No prefix matches — advance.
            i += 1;
        }
    }

    if longest_ngram < cfg.min_ngram {
        return Vec::new();
    }

    // Map back from the reversed view to the original sequence:
    //   the matched n-gram in original_tokens spans
    //   [total - 1 - position, total - 1 - position + longest_ngram)
    //   so drafts start at total - 1 - position + longest_ngram.
    let start = total - 1 - position + longest_ngram;
    let drafts_room = total.saturating_sub(start);
    let n = k_capped.min(drafts_room);
    if n == 0 {
        return Vec::new();
    }
    tokens[start..start + n].to_vec()
}

/// Copy a continuation from the most recent prior occurrence of the longest
/// matching request-history suffix.
///
/// The proposal is only a candidate.  The target model must verify every
/// returned token before it is committed to output or KV/recurrent state.
/// Consequently this helper cannot change model quality by itself.
///
/// Search order is part of the contract:
///
/// 1. longest suffix first (`max_match` down to `min_match`);
/// 2. for equal-length matches, newest prior occurrence first;
/// 3. copy at most `max_draft_tokens`, bounded by available history and model
///    context room.
///
/// This stateless implementation is allocation-free on a miss and performs
/// exact token comparison, so hash collisions cannot corrupt proposals.  A
/// persistent index may replace the scan after profiling, provided it
/// preserves the same ordering and exact-comparison contract.
pub fn propose_recent(tokens: &[u32], cfg: &HistoryLookupConfig) -> Vec<u32> {
    let total = tokens.len();
    if cfg.min_match == 0
        || cfg.max_match < cfg.min_match
        || cfg.max_draft_tokens == 0
        || total < cfg.min_match
    {
        return Vec::new();
    }

    let draft_room = cfg
        .max_draft_tokens
        .min(cfg.max_model_len.saturating_sub(total));
    if draft_room == 0 {
        return Vec::new();
    }

    let max_match = cfg.max_match.min(total);
    for match_len in (cfg.min_match..=max_match).rev() {
        let suffix_start = total - match_len;
        if suffix_start == 0 {
            continue;
        }
        let suffix = &tokens[suffix_start..];

        // Starts are visited newest-first.  `start < suffix_start` excludes
        // the suffix itself while still allowing overlapping repetitions.
        for start in (0..suffix_start).rev() {
            let end = start + match_len;
            if end > total || &tokens[start..end] != suffix {
                continue;
            }
            let available = total.saturating_sub(end);
            let n = draft_room.min(available);
            if n > 0 {
                return tokens[end..end + n].to_vec();
            }
        }
    }

    Vec::new()
}

/// Incremental request-owned index for [`propose_recent`] semantics.
///
/// The index records every position for each token ID.  A lookup starts from
/// occurrences of the suffix's final token, newest first, and then performs
/// an exact slice comparison.  That reduces the common miss from a complete
/// history scan to a handful of candidate positions without introducing a
/// collision-sensitive hash as a correctness dependency.
///
/// Only target-verified tokens may be appended.  Draft tokens must remain
/// outside this state until the verifier commits them; resetting at request
/// admission or cache invalidation makes the conversation boundary explicit.
#[derive(Debug, Clone)]
pub struct HistoryLookupIndex {
    cfg: HistoryLookupConfig,
    tokens: Vec<u32>,
    positions: HashMap<u32, Vec<usize>>,
}

impl HistoryLookupIndex {
    pub fn new(cfg: HistoryLookupConfig) -> Self {
        Self {
            cfg,
            tokens: Vec::new(),
            positions: HashMap::new(),
        }
    }

    /// Replace the complete verified request history.
    pub fn reset(&mut self, tokens: &[u32]) {
        self.tokens.clear();
        self.positions.clear();
        self.tokens.reserve(tokens.len());
        for &token in tokens {
            self.push_verified(token);
        }
    }

    /// Append target-verified tokens after a successful commit.
    pub fn extend_verified(&mut self, tokens: &[u32]) {
        self.tokens.reserve(tokens.len());
        for &token in tokens {
            self.push_verified(token);
        }
    }

    pub fn verified_len(&self) -> usize {
        self.tokens.len()
    }

    /// Return the same proposal ordering as [`propose_recent`] without a full
    /// request-history scan on ordinary misses.
    pub fn propose(&self) -> Vec<u32> {
        let total = self.tokens.len();
        if self.cfg.min_match == 0
            || self.cfg.max_match < self.cfg.min_match
            || self.cfg.max_draft_tokens == 0
            || total < self.cfg.min_match
        {
            return Vec::new();
        }

        let draft_room = self
            .cfg
            .max_draft_tokens
            .min(self.cfg.max_model_len.saturating_sub(total));
        if draft_room == 0 {
            return Vec::new();
        }

        let Some(candidate_ends) = self.positions.get(&self.tokens[total - 1]) else {
            return Vec::new();
        };
        let max_match = self.cfg.max_match.min(total);
        for match_len in (self.cfg.min_match..=max_match).rev() {
            let suffix_start = total - match_len;
            if suffix_start == 0 {
                continue;
            }
            let suffix = &self.tokens[suffix_start..];
            for &end in candidate_ends.iter().rev() {
                // Skip the suffix's own final token and candidates too short
                // to contain this match length.
                if end >= total - 1 || end + 1 < match_len {
                    continue;
                }
                let start = end + 1 - match_len;
                if start >= suffix_start || &self.tokens[start..=end] != suffix {
                    continue;
                }
                let continuation = end + 1;
                let n = draft_room.min(total - continuation);
                if n > 0 {
                    return self.tokens[continuation..continuation + n].to_vec();
                }
            }
        }
        Vec::new()
    }

    fn push_verified(&mut self, token: u32) {
        let position = self.tokens.len();
        self.tokens.push(token);
        self.positions.entry(token).or_default().push(position);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(min_n: usize, max_n: usize, k: usize) -> NgramConfig {
        NgramConfig {
            min_ngram: min_n,
            max_ngram: max_n,
            k,
            max_model_len: 4096,
        }
    }

    #[test]
    fn propose_empty_when_below_min_ngram() {
        // total < min_ngram → no proposal.
        assert!(propose(&[], &cfg(1, 3, 3)).is_empty());
        assert!(propose(&[7], &cfg(2, 3, 3)).is_empty());
    }

    #[test]
    fn propose_empty_when_no_match() {
        // [1,2,3,4,5,6] has no repeated n-gram → no proposal.
        let drafts = propose(&[1, 2, 3, 4, 5, 6], &cfg(2, 3, 3));
        assert!(drafts.is_empty(), "expected no drafts, got {:?}", drafts);
    }

    #[test]
    fn propose_basic_repetition() {
        // Tokens: [a, b, c, X, Y, a, b, c]
        // Suffix [a,b,c] (length 3) matches the prefix [a,b,c]; tokens
        // following the earlier occurrence are [X, Y, ...] — but the
        // earlier match is at indices 0..3, followed by [X, Y, a, b, c].
        // So drafts = [X, Y, a] truncated to k.
        let tokens = vec![10u32, 20, 30, 99, 88, 10, 20, 30];
        let drafts = propose(&tokens, &cfg(1, 3, 3));
        assert_eq!(drafts, vec![99, 88, 10]);
    }

    #[test]
    fn propose_respects_k_truncation() {
        // Same fixture; K=2 should truncate.
        let tokens = vec![10u32, 20, 30, 99, 88, 10, 20, 30];
        let drafts = propose(&tokens, &cfg(1, 3, 2));
        assert_eq!(drafts, vec![99, 88]);
    }

    #[test]
    fn propose_respects_max_ngram_cap() {
        // Suffix has a 5-long match, but max_ngram=2 caps it.
        // [a,b,c,d,e, X, a,b,c,d,e]: suffix [a..e] length 5 matches.
        // With max_ngram=2, the match cap is 2 (suffix [d,e] matching
        // the [d,e] in the middle).
        let tokens = vec![1u32, 2, 3, 4, 5, 99, 1, 2, 3, 4, 5];
        // n_gram=2 match: suffix [4,5], earlier occurrence at indices
        // 3..5 followed by [99, 1, 2, ...]. drafts = [99, 1, 2].
        let drafts = propose(&tokens, &cfg(2, 2, 3));
        assert_eq!(drafts, vec![99, 1, 2]);
    }

    #[test]
    fn propose_picks_earliest_occurrence_on_tie() {
        // Two equal-length matches; vLLM picks earliest in original
        // (= latest in reversed). Per line 253 of vLLM proposer using
        // `>=` not `>` for the position update.
        // Tokens: [a, b, X1, a, b, X2, a, b]
        // Suffix [a,b] matches the [a,b] at indices 0..2 (followed by
        // [X1, a, b, X2, a, b]) AND at indices 3..5 (followed by
        // [X2, a, b]). Earliest in original = 0..2 → drafts = [X1, a, b].
        let tokens = vec![10u32, 20, 100, 10, 20, 200, 10, 20];
        let drafts = propose(&tokens, &cfg(1, 3, 3));
        assert_eq!(drafts, vec![100, 10, 20]);
    }

    #[test]
    fn propose_caps_k_at_max_model_len() {
        let cfg = NgramConfig {
            min_ngram: 1,
            max_ngram: 3,
            k: 5,
            max_model_len: 10, // tokens.len() = 8 → k_room = 2
        };
        let tokens = vec![10u32, 20, 30, 99, 88, 10, 20, 30];
        let drafts = propose(&tokens, &cfg);
        assert_eq!(drafts.len(), 2, "expected k clamped to max_model_len - len");
        assert_eq!(drafts, vec![99, 88]);
    }

    #[test]
    fn propose_handles_longest_match_at_seq_end() {
        // Suffix exactly == start of tokens. No tokens after → 0 drafts.
        // [a, b, c, a, b, c]: suffix [a,b,c] matches prefix [a,b,c] at
        // 0..3. earlier_match end + drafts_start = 3. Tokens at 3.. = [a,b,c],
        // so drafts = [a,b,c].
        let tokens = vec![1u32, 2, 3, 1, 2, 3];
        let drafts = propose(&tokens, &cfg(1, 3, 3));
        assert_eq!(drafts, vec![1, 2, 3]);
    }

    #[test]
    fn propose_zero_max_ngram_returns_empty() {
        let bad_cfg = NgramConfig {
            min_ngram: 0,
            max_ngram: 0,
            k: 3,
            max_model_len: 4096,
        };
        assert!(propose(&[1, 2, 3], &bad_cfg).is_empty());
    }

    #[test]
    fn propose_k_zero_returns_empty() {
        let bad_cfg = NgramConfig {
            min_ngram: 1,
            max_ngram: 3,
            k: 0,
            max_model_len: 4096,
        };
        assert!(propose(&[1, 2, 3], &bad_cfg).is_empty());
    }

    #[test]
    fn default_config_is_reasonable() {
        let cfg = NgramConfig::default_for_decode(4096);
        assert_eq!(cfg.k, 3);
        assert_eq!(cfg.min_ngram, 1);
        assert_eq!(cfg.max_ngram, 3);
        assert_eq!(cfg.max_model_len, 4096);
    }

    /// Pseudo-random token generator for the bench fixtures.
    fn rand_tokens(seed: u64, n: usize, vocab: u32) -> Vec<u32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as u32) % vocab
            })
            .collect()
    }

    /// Microbench: confirm proposer CPU cost is sub-µs at realistic
    /// decode-state lengths so spec-decode overhead doesn't eat into
    /// the speedup. Per ADR-029 Phase 4 scope: every decode token
    /// runs one propose() call. At hf2q's 16 µs/dispatch GPU floor,
    /// proposer cost ≥ 100 µs would erase any spec-decode benefit at
    /// low acceptance rates.
    ///
    /// Run:
    ///   cargo test --release --bin hf2q --no-default-features \
    ///     bench_ngram_proposer -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_ngram_proposer_at_realistic_decode_lengths() {
        use std::time::Instant;

        // max_model_len comfortably above the longest tested length so
        // KMP actually runs (vs early-returning when tokens.len() ==
        // max_model_len triggers k_room == 0).
        let cfg = NgramConfig {
            min_ngram: 1,
            max_ngram: 3,
            k: 3,
            max_model_len: 16_384,
        };
        let lengths = [128usize, 512, 1024, 2048, 4096, 8192];

        for &n in &lengths {
            let tokens = rand_tokens(0xCAFE_BEEF, n, 256);
            // Warmup — primes branch predictor + cache.
            for _ in 0..100 {
                let _ = propose(&tokens, &cfg);
            }

            // Time 1000 iterations to get stable nanosecond p50.
            let mut samples: Vec<u128> = Vec::with_capacity(1000);
            for _ in 0..1000 {
                let t0 = Instant::now();
                let _ = propose(&tokens, &cfg);
                samples.push(t0.elapsed().as_nanos());
            }
            samples.sort();
            let p50 = samples[500];
            let p99 = samples[990];

            eprintln!(
                "[BENCH iter-115] propose len={:5} p50={:6} ns p99={:6} ns",
                n, p50, p99
            );

            // Falsifier: at any reasonable length, propose must be
            // <100 µs (= 100,000 ns). At 16 µs/dispatch, even one
            // propose call < 6% of one decode dispatch.
            assert!(
                (p50 as usize) < 100_000,
                "propose at len={n} took {p50} ns p50 — too slow for hot path (target <100 µs)"
            );
        }
    }

    #[test]
    fn history_lookup_prefers_longest_suffix() {
        // A two-token suffix occurs near the end, but the six-token suffix at
        // the beginning is the stronger lookup key and must win.
        let tokens = vec![1, 2, 3, 4, 5, 6, 70, 71, 9, 5, 6, 80, 1, 2, 3, 4, 5, 6];
        let cfg = HistoryLookupConfig {
            min_match: 2,
            max_match: 6,
            max_draft_tokens: 2,
            max_model_len: 4096,
        };
        assert_eq!(propose_recent(&tokens, &cfg), vec![70, 71]);
    }

    #[test]
    fn history_lookup_prefers_most_recent_occurrence_on_tie() {
        let key = [10, 11, 12, 13, 14, 15];
        let mut tokens = Vec::new();
        tokens.extend_from_slice(&key);
        tokens.extend_from_slice(&[100, 101]);
        tokens.extend_from_slice(&key);
        tokens.extend_from_slice(&[200, 201]);
        tokens.extend_from_slice(&key);
        let cfg = HistoryLookupConfig {
            max_draft_tokens: 2,
            ..HistoryLookupConfig::default_for_decode(4096)
        };
        assert_eq!(propose_recent(&tokens, &cfg), vec![200, 201]);
    }

    #[test]
    fn history_lookup_caps_draft_at_context_room() {
        let tokens = vec![1, 2, 3, 4, 5, 6, 90, 91, 92, 1, 2, 3, 4, 5, 6];
        let cfg = HistoryLookupConfig {
            min_match: 6,
            max_match: 12,
            max_draft_tokens: 5,
            max_model_len: tokens.len() + 2,
        };
        assert_eq!(propose_recent(&tokens, &cfg), vec![90, 91]);
    }

    #[test]
    fn history_lookup_rejects_invalid_or_unmatched_config() {
        let tokens = [1, 2, 3, 4, 5, 6];
        assert!(propose_recent(
            &tokens,
            &HistoryLookupConfig {
                min_match: 0,
                max_match: 12,
                max_draft_tokens: 5,
                max_model_len: 4096,
            }
        )
        .is_empty());
        assert!(propose_recent(&tokens, &HistoryLookupConfig::default_for_decode(4096)).is_empty());
    }

    #[test]
    fn history_lookup_index_matches_scan_and_updates_only_on_commit() {
        let cfg = HistoryLookupConfig::default_for_decode(4096);
        let initial = [1, 2, 3, 4, 5, 6, 90, 91, 1, 2, 3, 4, 5, 6];
        let mut index = HistoryLookupIndex::new(cfg);
        index.reset(&initial);
        assert_eq!(index.propose(), propose_recent(&initial, &cfg));
        assert_eq!(index.propose(), vec![90, 91, 1]);

        // A draft has no effect until the target verifier commits it.
        let unverified = [90, 91];
        assert_eq!(index.verified_len(), initial.len());
        assert_eq!(index.propose(), vec![90, 91, 1]);
        index.extend_verified(&unverified[..1]);
        let mut committed = initial.to_vec();
        committed.push(90);
        assert_eq!(index.propose(), propose_recent(&committed, &cfg));
        assert_eq!(index.verified_len(), committed.len());
    }

    #[test]
    fn history_lookup_index_matches_scan_across_random_prefixes() {
        let cfg = HistoryLookupConfig {
            min_match: 2,
            max_match: 8,
            max_draft_tokens: 5,
            max_model_len: 4096,
        };
        let mut tokens = rand_tokens(0xA11C_E5ED, 300, 32);
        // Plant repeated regions so the property covers hits as well as misses.
        let repeated = tokens[40..70].to_vec();
        tokens.extend_from_slice(&repeated);
        let mut index = HistoryLookupIndex::new(cfg);
        index.reset(&tokens[..16]);
        for &token in &tokens[16..] {
            assert_eq!(index.propose(), propose_recent(&index.tokens, &cfg));
            index.extend_verified(&[token]);
        }
        assert_eq!(index.propose(), propose_recent(&tokens, &cfg));
    }

    /// Developer microbenchmark for the stateless lookup scan.  Random input
    /// exercises the full miss cost; copied-tail input exercises the expected
    /// long-context assistant hit path.  This is evidence, not a hosted gate:
    /// run it in `--release` and record the exact host/commit with any claim.
    #[test]
    #[ignore]
    fn bench_history_lookup_random_miss_and_recent_hit() {
        use std::time::Instant;

        for &n in &[8_192usize, 100_000] {
            let mut random = rand_tokens(0x1385_9420, n, 248_064);
            let cfg = HistoryLookupConfig::default_for_decode(n + 256);
            let hit_tail = random[n - 64..n].to_vec();

            let mut miss_ns = Vec::with_capacity(200);
            for _ in 0..200 {
                let start = Instant::now();
                std::hint::black_box(propose_recent(std::hint::black_box(&random), &cfg));
                miss_ns.push(start.elapsed().as_nanos());
            }

            random.extend_from_slice(&hit_tail[..12]);
            let mut hit_ns = Vec::with_capacity(200);
            for _ in 0..200 {
                let start = Instant::now();
                std::hint::black_box(propose_recent(std::hint::black_box(&random), &cfg));
                hit_ns.push(start.elapsed().as_nanos());
            }

            miss_ns.sort_unstable();
            hit_ns.sort_unstable();
            eprintln!(
                "history_lookup len={n} miss_p50={}ns miss_p99={}ns hit_p50={}ns hit_p99={}ns",
                miss_ns[100], miss_ns[198], hit_ns[100], hit_ns[198]
            );
        }
    }

    #[test]
    #[ignore]
    fn bench_history_lookup_index_random_miss_and_recent_hit() {
        use std::time::Instant;

        for &n in &[8_192usize, 100_000] {
            let random = rand_tokens(0x1385_9420, n, 248_064);
            let cfg = HistoryLookupConfig::default_for_decode(n + 256);
            let mut index = HistoryLookupIndex::new(cfg);
            index.reset(&random);

            let mut miss_ns = Vec::with_capacity(1_000);
            for _ in 0..1_000 {
                let start = Instant::now();
                std::hint::black_box(index.propose());
                miss_ns.push(start.elapsed().as_nanos());
            }

            index.extend_verified(&random[n - 64..n - 52]);
            let mut hit_ns = Vec::with_capacity(1_000);
            for _ in 0..1_000 {
                let start = Instant::now();
                std::hint::black_box(index.propose());
                hit_ns.push(start.elapsed().as_nanos());
            }

            miss_ns.sort_unstable();
            hit_ns.sort_unstable();
            eprintln!(
                "history_lookup_index len={n} miss_p50={}ns miss_p99={}ns hit_p50={}ns hit_p99={}ns",
                miss_ns[500], miss_ns[990], hit_ns[500], hit_ns[990]
            );
        }
    }
}
