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
//! No model, no GPU, no quality loss. Default OFF in production
//! behind `HF2Q_SPEC_DECODE` env gate; iter-113 lands the algorithm
//! only — generation-loop integration arrives in ADR-029 Phase 3
//! after Phase 2's `forward_decode_verify` clears its byte-identity
//! gate.

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

impl NgramConfig {
    /// Default per ADR-028 iter-99 vLLM/dflash literature (K=3 optimal,
    /// n-grams 1..3 covering most natural-language repetitions).
    pub fn default_for_decode(max_model_len: usize) -> Self {
        Self { min_ngram: 1, max_ngram: 3, k: 3, max_model_len }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(min_n: usize, max_n: usize, k: usize) -> NgramConfig {
        NgramConfig { min_ngram: min_n, max_ngram: max_n, k, max_model_len: 4096 }
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
            min_ngram: 0, max_ngram: 0, k: 3, max_model_len: 4096,
        };
        assert!(propose(&[1, 2, 3], &bad_cfg).is_empty());
    }

    #[test]
    fn propose_k_zero_returns_empty() {
        let bad_cfg = NgramConfig {
            min_ngram: 1, max_ngram: 3, k: 0, max_model_len: 4096,
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
        (0..n).map(|_| {
            state = state.wrapping_mul(6364136223846793005)
                          .wrapping_add(1442695040888963407);
            ((state >> 33) as u32) % vocab
        }).collect()
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
            min_ngram: 1, max_ngram: 3, k: 3, max_model_len: 16_384,
        };
        let lengths = [128usize, 512, 1024, 2048, 4096, 8192];

        for &n in &lengths {
            let tokens = rand_tokens(0xCAFE_BEEF, n, 256);
            // Warmup — primes branch predictor + cache.
            for _ in 0..100 { let _ = propose(&tokens, &cfg); }

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

            eprintln!("[BENCH iter-115] propose len={:5} p50={:6} ns p99={:6} ns",
                      n, p50, p99);

            // Falsifier: at any reasonable length, propose must be
            // <100 µs (= 100,000 ns). At 16 µs/dispatch, even one
            // propose call < 6% of one decode dispatch.
            assert!(
                (p50 as usize) < 100_000,
                "propose at len={n} took {p50} ns p50 — too slow for hot path (target <100 µs)"
            );
        }
    }
}
