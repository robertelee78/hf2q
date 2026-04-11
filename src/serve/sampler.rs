//! Token sampling: temperature, top-k, top-p, repetition penalty.

use anyhow::Result;
use candle_core::{DType, Tensor};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::gemma4::DispatchCounters;

/// Sampling parameters for a generation request.
#[derive(Debug, Clone)]
#[allow(dead_code)]
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

/// ADR-005 1bNEW.3 fast-path: run `argmax` on greedy-decode logits WITHOUT
/// calling `to_scalar`, so the result stays a lazy `[1, 1]` u32 tensor.
///
/// The returned tensor is shaped to be handed straight to the next
/// `model.forward` as `input_ids`: candle's `Embedding::forward` calls
/// `index_select` on the indices tensor, which is a GPU op that takes any
/// contiguous u32 tensor — no host roundtrip required. Callers chain
/// `forward(N+1)` on the lazy argmax of `forward(N)`, accumulate a window
/// of argmax tensors, and drain them all at once via `cat + to_vec1`:
/// ONE forced GPU→CPU sync for the whole window instead of one per token.
/// See `mod.rs::run_decode_greedy_batched`.
///
/// This function does NOT itself force a GPU→CPU sync. The
/// `counters.sampler_sync_count` increment is deferred to the caller's
/// drain (one per drained window, not one per token).
///
/// Op chain is `squeeze(0) → squeeze(0) → argmax(0) → reshape((1,1))` —
/// bitwise-identical to the non-batched `sample_token` greedy fast path
/// (which does `squeeze → squeeze → argmax → to_scalar`). Only the
/// terminal `to_scalar` is skipped; the three view rewrites and the
/// argmax kernel dispatch are the same.
///
/// Gated on greedy T=0 with `repetition_penalty == 1.0`; any other
/// sampling configuration must route through the per-token `sample_token`
/// path, which needs the previous u32 before sampling the next one
/// (for RNG draws and the rep-penalty gather).
pub fn greedy_argmax_lazy(
    logits: &Tensor,
    counters: &Arc<DispatchCounters>,
) -> Result<Tensor> {
    let logits = logits.squeeze(0)?;
    let logits = if logits.dims().len() > 1 {
        logits.squeeze(0)?
    } else {
        logits
    };
    let arg_scalar = logits.argmax(0)?; // shape: [], dtype: u32
    debug_assert_eq!(arg_scalar.dtype(), DType::U32, "candle argmax must return U32");
    let next_input = arg_scalar.reshape((1, 1))?;
    // argmax is the only real dispatch; squeeze and reshape are view
    // rewrites on contiguous storage. Counted as 1 real dispatch to keep
    // per-token averages meaningful.
    counters.dispatches_per_token.fetch_add(1, Ordering::Relaxed);
    Ok(next_input)
}

/// Sample a token from logits.
///
/// `counters` is ADR-005 1bNEW.0 instrumentation — the greedy fast-path issues
/// exactly one forced GPU→CPU sync (`argmax(...).to_scalar()`), which this
/// function counts into `counters.sampler_sync_count`. Non-greedy paths pull
/// the full probs vector back to CPU via `to_vec1()`, which is also a sync
/// and is counted.
pub fn sample_token(
    logits: &Tensor,
    params: &SamplingParams,
    previous_tokens: &[u32],
    counters: &Arc<DispatchCounters>,
) -> Result<u32> {
    // Flatten to 1D
    let logits = logits.squeeze(0)?;
    let logits = if logits.dims().len() > 1 {
        logits.squeeze(0)?
    } else {
        logits
    };

    // Greedy fast-path
    if params.temperature < SAMPLING_EPS
        && (params.repetition_penalty == 1.0 || previous_tokens.is_empty())
    {
        // argmax + to_scalar forces one waitUntilCompleted. This is the
        // baseline `sampler_sync_count == 1` per token referenced in
        // ADR-005 1bNEW.0.
        let token_id = logits.argmax(0)?.to_scalar::<u32>()?;
        counters.sampler_sync_count.fetch_add(1, Ordering::Relaxed);
        counters.dispatches_per_token.fetch_add(2, Ordering::Relaxed);
        return Ok(token_id);
    }

    let logits = logits.to_dtype(DType::F32)?;

    // Repetition penalty
    let logits = if params.repetition_penalty != 1.0 && !previous_tokens.is_empty() {
        apply_repetition_penalty(&logits, previous_tokens, params.repetition_penalty)?
    } else {
        logits
    };

    // Temperature
    let logits = (&logits / params.temperature)?;

    // Softmax → probabilities
    let probs = candle_nn::ops::softmax_last_dim(&logits)?;
    // Full probs pull is a forced sync on the non-greedy path.
    let probs_vec: Vec<f32> = probs.to_vec1()?;
    counters.sampler_sync_count.fetch_add(1, Ordering::Relaxed);
    counters.dispatches_per_token.fetch_add(2, Ordering::Relaxed);

    let mut indexed: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Top-k
    if params.top_k > 0 && params.top_k < indexed.len() {
        indexed.truncate(params.top_k);
    }

    // Top-p (nucleus)
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

    // Renormalize and sample
    let sum: f32 = indexed.iter().map(|&(_, p)| p).sum();
    if sum <= 0.0 {
        return Ok(indexed.first().map(|&(idx, _)| idx as u32).unwrap_or(0));
    }

    let mut rng_val = rand_f32();
    for &(idx, prob) in &indexed {
        let normalized = prob / sum;
        if rng_val < normalized {
            return Ok(idx as u32);
        }
        rng_val -= normalized;
    }

    Ok(indexed.last().map(|&(idx, _)| idx as u32).unwrap_or(0))
}

fn apply_repetition_penalty(
    logits: &Tensor,
    previous_tokens: &[u32],
    penalty: f64,
) -> Result<Tensor> {
    let vocab = logits.dim(0)?;
    let device = logits.device();
    let penalty_f = penalty as f32;

    let mut seen = std::collections::HashSet::new();
    let unique_ids: Vec<u32> = previous_tokens
        .iter()
        .copied()
        .filter(|&id| (id as usize) < vocab && seen.insert(id))
        .collect();

    if unique_ids.is_empty() {
        return Ok(logits.clone());
    }

    let indices = Tensor::new(unique_ids.as_slice(), device)?;
    let gathered = logits.gather(&indices, 0)?;
    let gathered_vec: Vec<f32> = gathered.to_vec1()?;

    let factors: Vec<f32> = gathered_vec
        .iter()
        .map(|&s| if s >= 0.0 { 1.0 / penalty_f } else { penalty_f })
        .collect();

    let mut multiplier = vec![1.0f32; vocab];
    for (&id, &f) in unique_ids.iter().zip(factors.iter()) {
        multiplier[id as usize] = f;
    }
    let mult = Tensor::from_vec(multiplier, vocab, device)?;
    (logits * mult).map_err(Into::into)
}

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
