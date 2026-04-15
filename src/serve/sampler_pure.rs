//! Pure-Rust token sampling: temperature, top-k, top-p, repetition penalty.
//!
//! ADR-008 Phase 3: operates on `&[f32]` / `&mut [f32]` logit slices with zero
//! candle dependency.  Drop-in replacement for `sampler.rs` on the mlx-native
//! decode path.

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Sampling parameters (identical to sampler.rs — no candle deps)
// ---------------------------------------------------------------------------

/// Sampling parameters for a generation request.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repetition_penalty: f64,
    pub max_tokens: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
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
    // Greedy fast-path: no temperature, no rep-penalty effect
    // ------------------------------------------------------------------
    if params.temperature < SAMPLING_EPS
        && (params.repetition_penalty == 1.0 || previous_tokens.is_empty())
    {
        return sample_greedy(logits);
    }

    // ------------------------------------------------------------------
    // Repetition penalty (in-place)
    // ------------------------------------------------------------------
    if params.repetition_penalty != 1.0 && !previous_tokens.is_empty() {
        apply_repetition_penalty(logits, previous_tokens, params.repetition_penalty);
    }

    // ------------------------------------------------------------------
    // Temperature scaling (in-place)
    // ------------------------------------------------------------------
    let inv_temp = 1.0 / params.temperature as f32;
    for v in logits.iter_mut() {
        *v *= inv_temp;
    }

    // ------------------------------------------------------------------
    // Softmax → probabilities (in-place)
    // ------------------------------------------------------------------
    softmax_in_place(logits);

    // ------------------------------------------------------------------
    // Build sorted (index, probability) pairs
    // ------------------------------------------------------------------
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // ------------------------------------------------------------------
    // Top-k truncation
    // ------------------------------------------------------------------
    if params.top_k > 0 && params.top_k < indexed.len() {
        indexed.truncate(params.top_k);
    }

    // ------------------------------------------------------------------
    // Top-p (nucleus) truncation
    // ------------------------------------------------------------------
    if params.top_p < 1.0 {
        let mut cumsum = 0.0f32;
        let mut cutoff = indexed.len();
        for (i, &(_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum >= params.top_p as f32 {
                cutoff = i + 1;
                break;
            }
        }
        indexed.truncate(cutoff);
    }

    // ------------------------------------------------------------------
    // Renormalize and multinomial sample
    // ------------------------------------------------------------------
    let sum: f32 = indexed.iter().map(|&(_, p)| p).sum();
    if sum <= 0.0 {
        return indexed.first().map(|&(idx, _)| idx as u32).unwrap_or(0);
    }

    let mut rng_val = rand_f32();
    for &(idx, prob) in &indexed {
        let normalized = prob / sum;
        if rng_val < normalized {
            return idx as u32;
        }
        rng_val -= normalized;
    }

    indexed.last().map(|&(idx, _)| idx as u32).unwrap_or(0)
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
