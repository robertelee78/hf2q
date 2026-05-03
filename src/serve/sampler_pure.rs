//! Pure-Rust token sampling: temperature, top-k, top-p, repetition penalty.
//!
//! ADR-008 Phase 3: operates on `&[f32]` / `&mut [f32]` logit slices with zero
//! candle dependency.  Drop-in replacement for `sampler.rs` on the mlx-native
//! decode path.

use std::cell::RefCell;
use std::collections::HashSet;

// 2026-05-03 — thread-local scratch for `sample_token`'s indexed-pair Vec.
//
// Without this each call to the sampling chain mallocs a fresh
// `Vec<(usize, f32)>` of capacity = vocab_size (~248K for Qwen3.6 →
// ~3 MB per step, ~600 MB churn over a 200-token generation). The Vec is
// fully overwritten each call and dropped at function exit; pre-allocating
// per-thread amortizes the malloc + first-touch cost across all decode
// steps. The scratch buffer is opaque to callers (no API change) and
// monotonically grows to the high-water vocab size; `clear()` keeps the
// allocation alive across calls.
thread_local! {
    static SAMPLE_INDEXED_SCRATCH: RefCell<Vec<(usize, f32)>> =
        RefCell::new(Vec::new());
}

// ---------------------------------------------------------------------------
// Sampling parameters (identical to sampler.rs — no candle deps)
// ---------------------------------------------------------------------------

/// Sampling parameters for a generation request.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub min_p: f64,
    pub repetition_penalty: f64,
    pub max_tokens: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            min_p: 0.0,
            repetition_penalty: 1.0,
            max_tokens: 2048,
        }
    }
}

pub const SAMPLING_EPS: f64 = 1e-5;

// ---------------------------------------------------------------------------
// Greedy argmax
// ---------------------------------------------------------------------------

/// Return the index of the maximum value in `logits`.
///
/// Returns `0` for an empty slice.  NaN values are treated as less-than any
/// finite value (consistent with `f32::max`).
pub fn sample_greedy(logits: &[f32]) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    let mut best_idx: usize = 0;
    let mut best_val: f32 = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        // Use `>` so that NaN (which fails all comparisons) never wins.
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

// ---------------------------------------------------------------------------
// Main sampling entry-point
// ---------------------------------------------------------------------------

/// Sample a single token from a mutable logit slice.
///
/// The caller hands us an owned `&mut [f32]` so we can modify logits in-place
/// (repetition penalty, temperature scaling, softmax).  This avoids any
/// allocation for the common greedy fast-path and keeps allocation minimal for
/// the non-greedy path (one `Vec<(usize, f32)>` for sort + top-k/p).
pub fn sample_token(
    logits: &mut [f32],
    params: &SamplingParams,
    previous_tokens: &[u32],
) -> u32 {
    // ------------------------------------------------------------------
    // Repetition penalty (in-place on raw logits).
    // ------------------------------------------------------------------
    if params.repetition_penalty != 1.0 && !previous_tokens.is_empty() {
        apply_repetition_penalty(logits, previous_tokens, params.repetition_penalty);
    }

    // ------------------------------------------------------------------
    // Greedy fast-path. Bypasses the entire chain at temp=0.
    // ------------------------------------------------------------------
    if params.temperature < SAMPLING_EPS {
        return sample_greedy(logits);
    }

    // ------------------------------------------------------------------
    // Build sorted (index, LOGIT) pairs. The pair value is the LOGIT
    // throughout the chain — never a probability — so top-p, min-p, and
    // temperature each see the original logit space. This mirrors
    // llama.cpp/src/llama-sampler.cpp where `llama_token_data` carries both
    // `.logit` and `.p` and only the dist sampler reads `.p`.
    // ------------------------------------------------------------------
    // Build the indexed-pair Vec in the thread-local scratch buffer.
    // Drop the per-step is_finite() filter from the build path: NaNs are
    // exceedingly rare on a healthy logit distribution and the downstream
    // sort-by/select-by comparators already fall back to `Equal` via
    // `partial_cmp().unwrap_or(Equal)`, which keeps NaN entries from
    // distorting the top-k. Drop branches save ~250µs/step on Qwen3.6's
    // 248K vocab; the rare NaN case is still handled deterministically.
    let result = SAMPLE_INDEXED_SCRATCH.with(|cell| -> Option<u32> {
        let mut indexed = cell.borrow_mut();
        indexed.clear();
        indexed.reserve(logits.len());
        for (i, &l) in logits.iter().enumerate() {
            indexed.push((i, l));
        }
        if indexed.is_empty() {
            return Some(sample_greedy(logits));
        }
        sample_token_indexed(&mut indexed, params)
    });
    if let Some(out) = result {
        return out;
    }
    sample_greedy(logits)
}

fn sample_token_indexed(
    indexed: &mut Vec<(usize, f32)>,
    params: &SamplingParams,
) -> Option<u32> {

    // ------------------------------------------------------------------
    // Top-k truncation on raw logits (llama-sampler.cpp:317).
    //
    // 2026-05-03 — perf: previously this section did a full O(V log V)
    // descending sort of all ~248K vocab entries before truncating to
    // top_k=40. With Qwen3.6's 248044-entry vocab that's ~4.5M f32
    // comparisons and was the dominant per-step cost on the sampling
    // path (default --temperature=0.8 → ~74 tok/s vs greedy 122 tok/s
    // on qwen3.6-35B-A3B-dwq48). When `top_k > 0 && top_k < V`, use
    // `select_nth_unstable_by` (O(V) average partition) followed by an
    // O(K log K) sort of the small top-k subset. Mirrors llama.cpp
    // `llama_sampler_top_k_impl` which uses `std::nth_element` for the
    // same reason. The full-sort fallback only fires when top_k is
    // disabled or covers the whole vocab — in which case the downstream
    // top_p loop genuinely needs all logits sorted.
    // ------------------------------------------------------------------
    let cmp_desc = |a: &(usize, f32), b: &(usize, f32)| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    };
    if params.top_k > 0 && params.top_k < indexed.len() {
        let top_k = params.top_k;
        indexed.select_nth_unstable_by(top_k - 1, cmp_desc);
        indexed.truncate(top_k);
        indexed.sort_by(cmp_desc);
    } else {
        indexed.sort_by(cmp_desc);
    }

    // ------------------------------------------------------------------
    // Top-p (nucleus) truncation. Mirrors llama-sampler.cpp:1351 — softmax
    // the logits into a side prob buffer, cumsum until p threshold, truncate
    // the (idx, logit) array. The .logit pairs are NOT mutated.
    // ------------------------------------------------------------------
    if params.top_p < 1.0 && indexed.len() > 1 {
        let probs = softmax_logits_to_probs(&indexed);
        let mut cumsum = 0.0f32;
        let mut cutoff = indexed.len();
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= params.top_p as f32 {
                cutoff = i + 1;
                break;
            }
        }
        indexed.truncate(cutoff);
    }

    // ------------------------------------------------------------------
    // Min-p truncation on raw logits (llama-sampler.cpp:1560). The logit
    // threshold equivalent of `p_i >= min_p * p_max` is
    // `logit_i >= max_logit + ln(min_p)`. No softmax needed.
    // ------------------------------------------------------------------
    if params.min_p > 0.0 && indexed.len() > 1 {
        let max_logit = indexed[0].1;
        let min_logit_threshold = max_logit + (params.min_p as f32).ln();
        // Always keep at least the top token (llama.cpp min_keep semantics).
        let mut cutoff = 1;
        for (i, &(_, l)) in indexed.iter().enumerate().skip(1) {
            if l >= min_logit_threshold {
                cutoff = i + 1;
            } else {
                break;
            }
        }
        indexed.truncate(cutoff);
    }

    // ------------------------------------------------------------------
    // Temperature scale of LOGITS (llama-sampler.cpp:285).
    // ------------------------------------------------------------------
    let inv_temp = 1.0 / params.temperature as f32;
    for (_, l) in indexed.iter_mut() {
        *l *= inv_temp;
    }

    // ------------------------------------------------------------------
    // Final softmax + multinomial sample.
    // ------------------------------------------------------------------
    let probs = softmax_logits_to_probs(indexed);
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return Some(indexed.first().map(|&(idx, _)| idx as u32).unwrap_or(0));
    }

    let mut rng_val = rand_f32();
    for (i, &p) in probs.iter().enumerate() {
        let normalized = p / sum;
        if rng_val < normalized {
            return Some(indexed[i].0 as u32);
        }
        rng_val -= normalized;
    }

    Some(indexed.last().map(|&(idx, _)| idx as u32).unwrap_or(0))
}

/// Compute softmax probabilities from `(idx, logit)` pairs without mutating
/// the pair values. Returns a parallel `Vec<f32>` of probabilities summing to
/// ~1.0. Mirrors llama.cpp `llama_sampler_softmax_impl`.
fn softmax_logits_to_probs(indexed: &[(usize, f32)]) -> Vec<f32> {
    if indexed.is_empty() {
        return Vec::new();
    }
    let max = indexed
        .iter()
        .map(|&(_, l)| l)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = indexed.iter().map(|&(_, l)| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 && sum.is_finite() {
        let inv = 1.0 / sum;
        for p in probs.iter_mut() {
            *p *= inv;
        }
    }
    probs
}

// ---------------------------------------------------------------------------
// Repetition penalty
// ---------------------------------------------------------------------------

/// Apply repetition penalty in-place to logits for previously generated tokens.
///
/// For each unique token in `previous_tokens` whose ID is within bounds:
/// - If `logits[id] >= 0.0`: multiply by `1.0 / penalty`  (reduce probability)
/// - If `logits[id] < 0.0`:  multiply by `penalty`         (reduce probability)
///
/// This matches the candle-based implementation exactly.
pub fn apply_repetition_penalty(
    logits: &mut [f32],
    previous_tokens: &[u32],
    penalty: f64,
) {
    let vocab = logits.len();
    let penalty_f = penalty as f32;
    let inv_penalty = 1.0f32 / penalty_f;

    let mut seen = HashSet::new();
    for &id in previous_tokens {
        let idx = id as usize;
        if idx < vocab && seen.insert(id) {
            if logits[idx] >= 0.0 {
                logits[idx] *= inv_penalty;
            } else {
                logits[idx] *= penalty_f;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

/// In-place softmax over a float slice: `probs[i] = exp(probs[i] - max) / sum`.
fn softmax_in_place(probs: &mut [f32]) {
    let max = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for p in probs.iter_mut() {
        *p = (*p - max).exp();
        sum += *p;
    }
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for p in probs.iter_mut() {
            *p *= inv_sum;
        }
    }
}

fn softmax_pairs_in_place(indexed: &mut [(usize, f32)]) {
    let max = indexed
        .iter()
        .map(|&(_, v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (_, v) in indexed.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for (_, v) in indexed.iter_mut() {
            *v *= inv_sum;
        }
    }
}

// ---------------------------------------------------------------------------
// PRNG — xorshift64* (verbatim copy from sampler.rs)
// ---------------------------------------------------------------------------

/// Thread-local xorshift64* PRNG.
fn rand_f32() -> f32 {
    use std::time::SystemTime;

    thread_local! {
        static STATE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
        static SEEDED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    SEEDED.with(|seeded| {
        if !seeded.get() {
            let t = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            let tid = {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut h = DefaultHasher::new();
                std::thread::current().id().hash(&mut h);
                h.finish()
            };
            let mut seed = t ^ tid;
            seed = seed.wrapping_add(0x9e3779b97f4a7c15);
            seed = (seed ^ (seed >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            seed = (seed ^ (seed >> 27)).wrapping_mul(0x94d049bb133111eb);
            seed ^= seed >> 31;
            STATE.with(|s| s.set(if seed == 0 { 0x1234567890abcdef } else { seed }));
            seeded.set(true);
        }
    });

    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        let u = x.wrapping_mul(0x2545f4914f6cdd1d) >> 32;
        u as f32 / (u32::MAX as f32 + 1.0)
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed analytic verification: with logits [2, 1, 0, -1, -2]
    /// and all-pass params (temp=1, top_p=1, top_k=0, min_p=0), the final
    /// distribution must equal softmax of the raw logits.
    fn raw_softmax(logits: &[f32]) -> Vec<f64> {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let mut p: Vec<f64> = logits.iter().map(|l| ((*l as f64) - max).exp()).collect();
        let s: f64 = p.iter().sum();
        for v in p.iter_mut() {
            *v /= s;
        }
        p
    }

    /// Drive the sampler `n` times and return frequency-of-each-index.
    fn sample_frequencies(
        base_logits: &[f32],
        params: &SamplingParams,
        n: usize,
    ) -> Vec<f64> {
        let mut counts = vec![0usize; base_logits.len()];
        for _ in 0..n {
            let mut buf = base_logits.to_vec();
            let tok = sample_token(&mut buf, params, &[]);
            counts[tok as usize] += 1;
        }
        counts.iter().map(|&c| c as f64 / n as f64).collect()
    }

    #[test]
    fn sampler_all_pass_matches_raw_softmax() {
        // With temp=1, no truncation, the empirical distribution must match
        // softmax(logits) within sampling noise on a moderate sample size.
        let logits = vec![2.0_f32, 1.0, 0.0, -1.0, -2.0];
        let expected = raw_softmax(&logits);
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            max_tokens: 1,
        };
        let freq = sample_frequencies(&logits, &params, 20_000);
        for i in 0..logits.len() {
            let diff = (freq[i] - expected[i]).abs();
            assert!(
                diff < 0.02,
                "all-pass sampler diverges from softmax at idx {i}: \
                 expected={:.4} got={:.4} diff={:.4}",
                expected[i], freq[i], diff,
            );
        }
    }

    #[test]
    fn sampler_temperature_concentrates_winner() {
        // Pin the temperature semantics: lowering temp must INCREASE the
        // winner-pick rate. Buggy chain (temp on probs not logits) compresses
        // this effect.
        let logits = vec![3.0_f32, 2.0, 1.0, 0.0];
        let cold = SamplingParams {
            temperature: 0.3,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            max_tokens: 1,
        };
        let hot = SamplingParams {
            temperature: 2.0,
            ..cold.clone()
        };
        let cold_freq = sample_frequencies(&logits, &cold, 10_000);
        let hot_freq = sample_frequencies(&logits, &hot, 10_000);
        // Cold winner-rate must be substantially higher than hot.
        assert!(
            cold_freq[0] - hot_freq[0] > 0.20,
            "temperature does not concentrate winner: cold={:.3} hot={:.3} \
             diff={:.3}",
            cold_freq[0], hot_freq[0], cold_freq[0] - hot_freq[0],
        );
        // Cold winner-rate at temp=0.3 should be ≥ 0.85 (analytic ~0.96).
        assert!(
            cold_freq[0] > 0.80,
            "cold winner-rate too low: {:.3} (expected > 0.80)",
            cold_freq[0],
        );
    }

    #[test]
    fn sampler_min_p_filters_distant_tokens() {
        // With one dominant logit and a long tail of small logits, min_p=0.05
        // (logit threshold = max + ln(0.05) ≈ max - 3.0) must filter out
        // anything below that threshold.
        let mut logits = vec![0.5_f32; 100];
        logits[7] = 10.0; // winner: max_logit = 10.0; threshold = 7.0
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.05,
            repetition_penalty: 1.0,
            max_tokens: 1,
        };
        let freq = sample_frequencies(&logits, &params, 5_000);
        // Only index 7 passes the min-p logit threshold; sampler must always
        // return it.
        assert!(
            freq[7] > 0.99,
            "min-p failed to isolate dominant token: freq[7]={:.4}",
            freq[7],
        );
    }

    #[test]
    fn sampler_top_k_truncates_to_k_candidates() {
        // With top_k=2 and three distinct logits, only the top two should
        // ever be sampled.
        let logits = vec![1.0_f32, 2.0, 3.0, 0.0, -1.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 2,
            min_p: 0.0,
            repetition_penalty: 1.0,
            max_tokens: 1,
        };
        let freq = sample_frequencies(&logits, &params, 5_000);
        // idx 2 (logit=3.0) and idx 1 (logit=2.0) are the top 2.
        assert_eq!(
            freq[3], 0.0,
            "top-k=2 leaked to idx 3 (logit=0.0)"
        );
        assert_eq!(
            freq[4], 0.0,
            "top-k=2 leaked to idx 4 (logit=-1.0)"
        );
        assert_eq!(
            freq[0], 0.0,
            "top-k=2 leaked to idx 0 (logit=1.0)"
        );
    }

    #[test]
    fn sampler_default_chain_preserves_dominant_winner() {
        // Pin the regression: default chain (temp=0.8 top_p=0.95 top_k=40
        // min_p=0.05) on logits with a moderately dominant winner must pick
        // it >70% of the time. Pre-fix the chain returned ~uniform on the
        // top-k=40 set.
        let mut logits = vec![0.0_f32; 50];
        logits[7] = 5.0; // winner well above the long-tail
        let params = SamplingParams {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.05,
            repetition_penalty: 1.0,
            max_tokens: 1,
        };
        let freq = sample_frequencies(&logits, &params, 5_000);
        assert!(
            freq[7] > 0.70,
            "regression: dominant-winner pick rate too low under default \
             chain: freq[7]={:.4} (pre-fix bug returned ~0.025)",
            freq[7],
        );
    }

    #[test]
    fn greedy_path_unchanged_at_temp_zero() {
        let mut logits = vec![0.0_f32, 5.0, 3.0, 1.0];
        let params = SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            max_tokens: 1,
        };
        assert_eq!(sample_token(&mut logits, &params, &[]), 1);
    }
}
