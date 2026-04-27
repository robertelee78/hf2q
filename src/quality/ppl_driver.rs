//! ADR-014 P10 iter-2b — hf2q-side PPL driver for Qwen3.5-dense GGUFs.
//!
//! Wraps the EXISTING public API of [`Qwen35Model`] (`load_from_gguf`
//! plus `forward_cpu`) and routes the produced logits through
//! [`crate::quality::perplexity::compute_perplexity`]. The driver lives
//! in `src/quality/` (not `src/inference/models/qwen35/`) so the
//! parallel ADR-013 session that owns the engine can keep its file
//! fence intact: ppl_driver only READS the engine's public API; never
//! modifies it.
//!
//! # Public surface
//!
//! - [`measure_ppl_qwen35_dense`] — the entry point.
//! - [`PplDriverError`] — the typed error enum.
//! - [`chunk_count`] — small pure helper exposed for unit tests in
//!   [`crate::quality::ppl_driver`] and in the integration crate at
//!   `tests/ppl_driver.rs`. Same arithmetic that drives the chunking
//!   loop, surfaced so the integration tests can assert it without
//!   loading a model.
//!
//! # Algorithm
//!
//! 1. `GgufFile::open(model)` → typed `Gguf` error if the path is
//!    missing, the magic is wrong, or the file is otherwise unreadable.
//! 2. `Qwen35Model::load_from_gguf(&gguf)` → typed `Load` error.
//! 3. Compute `seq_len = caller_override.unwrap_or(model.cfg.max_position_embeddings)`,
//!    clamped to `tokens.len()` so the largest possible single chunk
//!    is the whole corpus when the user doesn't pass an explicit
//!    window.
//! 4. Walk `tokens` in non-overlapping windows of `seq_len`. For each
//!    window:
//!    - Build text-convention `[i, i, i, i]` positions.
//!    - Call `forward_cpu` (typed `Forward { chunk, source }` error
//!      with the chunk index so a regression at, say, the 5th window
//!      surfaces with the right index in the message).
//!    - Reshape the returned `[L × vocab]` flat `Vec<f32>` into a
//!      `Vec<Vec<f32>>` of `L` rows.
//! 5. Drop the last row of every window's logits AND the first token
//!    of every window's targets. Standard "predict next token"
//!    alignment for PPL: `logits[i]` predicts `tokens[i+1]`. This also
//!    matches `compute_perplexity`'s contract that
//!    `logits_sequence.len() == targets.len()`.
//! 6. Concatenate all per-window `(logits, targets)` pairs into one
//!    `(Vec<Vec<f32>>, Vec<u32>)` and call `compute_perplexity`.
//! 7. Cast the resulting `f64` PPL to `f32` (the
//!    `tests/peer_parity_gates.rs` `CellResult::ppl_hf2q` field is
//!    `Option<f32>`; the f64 → f32 boundary lives here so the
//!    public driver type matches the consumer).
//!
//! # Determinism
//!
//! `forward_cpu` is deterministic for the same model + same tokens
//! (the engine's own `forward_cpu_deterministic` test pins this).
//! `compute_perplexity` is a pure reduction over the logits. Therefore
//! `measure_ppl_qwen35_dense` is deterministic for the same model
//! file + same token slice + same `seq_len`.

use std::path::{Path, PathBuf};

use thiserror::Error;

use mlx_native::gguf::GgufFile;
use mlx_native::MlxError;

use crate::inference::models::qwen35::forward_cpu::text_positions;
use crate::inference::models::qwen35::model::Qwen35Model;
use crate::inference::models::qwen35::Qwen35Variant;
use crate::quality::perplexity::{compute_perplexity, PerplexityError};

/// Errors surfaced by [`measure_ppl_qwen35_dense`]. Every failure mode
/// — IO, GGUF parse, model load, forward pass, perplexity compute,
/// invalid input — lands in a discriminated variant so callers
/// (`tests/peer_parity_gates.rs::run_cell` for the dense cells) can
/// route on the cause without inspecting strings.
#[derive(Debug, Error)]
pub enum PplDriverError {
    /// `GgufFile::open` failed (file missing, bad magic, truncated
    /// header, unsupported version, …). The `path` field carries the
    /// caller-supplied path verbatim so the markdown table / log line
    /// surfaces *which* GGUF triggered the failure.
    #[error("failed to read GGUF at {path}: {source}")]
    Gguf {
        path: PathBuf,
        #[source]
        source: MlxError,
    },

    /// `Qwen35Model::load_from_gguf` returned a non-Gguf error
    /// (config parse, unexpected dtype, weight-loader complaint).
    /// We carry the formatted message rather than the `anyhow::Error`
    /// so `PplDriverError` stays `Send + Sync + 'static` without
    /// transitively requiring the same of every internal source.
    #[error("model load failed: {0}")]
    Load(String),

    /// `forward_cpu` failed at chunk index `chunk` (0-indexed). The
    /// chunk index lets a regression at the Nth window surface in the
    /// error rather than blending into the first window's failure.
    /// `cause` carries the formatted source message — we render the
    /// `anyhow::Error` chain into a `String` at the catch site so
    /// `PplDriverError` stays `Send + Sync + 'static` without
    /// transitively requiring the same of the qwen35 forward path.
    #[error("forward pass failed at chunk {chunk}: {cause}")]
    Forward {
        chunk: usize,
        cause: String,
    },

    /// `compute_perplexity` rejected the `(logits, targets)` pair
    /// (count mismatch, NaN logits, OOB target, empty input). Should
    /// be unreachable in production because the driver constructs the
    /// inputs itself; surfaced explicitly so a future regression
    /// can't be mistaken for a `Forward` or `Invalid` error.
    #[error("compute_perplexity failed: {0}")]
    Perplexity(#[from] PerplexityError),

    /// Caller-supplied input violates the driver's preconditions:
    /// fewer than 2 tokens (no prediction possible), an explicit
    /// `seq_len = Some(0)`, or a non-Dense GGUF (the driver is
    /// dense-only this iter — MoE lands at iter-2c when the public
    /// API for a quantized-MoE forward pass is wired).
    #[error("invalid input: {0}")]
    Invalid(String),
}

/// Number of non-overlapping `seq_len`-sized windows required to
/// cover `n_tokens`. Pure arithmetic — no model load. Used both
/// internally (the chunking loop driver) and by the integration test
/// crate at `tests/ppl_driver.rs` to lock the chunking math without
/// needing a model.
///
/// Returns 0 if either input is 0; otherwise rounds up
/// (`(n_tokens + seq_len - 1) / seq_len`).
pub fn chunk_count(n_tokens: usize, seq_len: usize) -> usize {
    if n_tokens == 0 || seq_len == 0 {
        return 0;
    }
    n_tokens.div_ceil(seq_len)
}

/// Measure the perplexity of a Qwen3.5-dense GGUF model against a
/// `u32` token corpus.
///
/// # Arguments
///
/// * `model` — path to a Qwen3.5-dense GGUF (architecture
///   `qwen35`). MoE variants are rejected with
///   [`PplDriverError::Invalid`] until iter-2c lands a MoE driver.
///   Quantized-FFN dense GGUFs (`DenseQ`) are loadable but
///   `forward_cpu` errors on them; that error flows through
///   [`PplDriverError::Forward`] and is surfaced honestly to the
///   caller (the driver does NOT silently fall back to a different
///   path).
/// * `tokens` — corpus tokens. Must contain at least 2 tokens
///   (one prediction = one logits row + one target).
/// * `seq_len` — optional override for the per-chunk window size.
///   `None` uses `cfg.max_position_embeddings` clamped to
///   `tokens.len()` (so a 512-token corpus + a 32k-context model
///   runs as a single chunk). Pass `Some(N)` to force `N`-token
///   windows (useful for tests with synthetic corpora < the model's
///   training context).
///
/// # Returns
///
/// The corpus perplexity as `f32`. The internal `compute_perplexity`
/// returns `f64`; we narrow at the boundary because
/// `tests/peer_parity_gates.rs::CellResult::ppl_hf2q` is
/// `Option<f32>`.
///
/// # Errors
///
/// See [`PplDriverError`]. Each variant identifies a distinct
/// failure surface so the caller can route without string-matching.
///
/// # Determinism
///
/// Same `model` + same `tokens` + same `seq_len` ⇒ identical f32
/// PPL bit pattern. Inherits determinism from `forward_cpu`
/// (validated by the engine's `forward_cpu_deterministic` test) and
/// the pure-reduction nature of `compute_perplexity`.
pub fn measure_ppl_qwen35_dense(
    model: &Path,
    tokens: &[u32],
    seq_len: Option<usize>,
) -> Result<f32, PplDriverError> {
    // --- 1. Input validation (cheap, no IO) -----------------------
    if tokens.len() < 2 {
        return Err(PplDriverError::Invalid(format!(
            "tokens.len() = {} < 2; need at least one prediction (one logits row + one target) to compute PPL",
            tokens.len()
        )));
    }
    if let Some(0) = seq_len {
        return Err(PplDriverError::Invalid(
            "seq_len override cannot be 0; pass None for the model default or a positive value".to_string(),
        ));
    }

    // --- 2. Open the GGUF -----------------------------------------
    let gguf = GgufFile::open(model).map_err(|source| PplDriverError::Gguf {
        path: model.to_path_buf(),
        source,
    })?;

    // --- 3. Load the model ----------------------------------------
    let qwen = Qwen35Model::load_from_gguf(&gguf)
        .map_err(|e| PplDriverError::Load(format!("{e:#}")))?;

    // Variant gate: dense-only this iter. MoE driver lands at iter-2c.
    if qwen.cfg.variant != Qwen35Variant::Dense {
        return Err(PplDriverError::Invalid(format!(
            "model variant is {:?}; measure_ppl_qwen35_dense only supports Dense \
             (MoE driver lands at iter-2c)",
            qwen.cfg.variant
        )));
    }

    let vocab_size = qwen.cfg.vocab_size as usize;
    if vocab_size == 0 {
        return Err(PplDriverError::Invalid(
            "model.cfg.vocab_size is 0; cannot reshape logits".to_string(),
        ));
    }

    // --- 4. Resolve effective seq_len ------------------------------
    // Default: model's training context, clamped down to the corpus
    // length so a small corpus runs as a single chunk. Caller
    // override wins outright (subject to the > 0 check above).
    let n_tokens = tokens.len();
    let effective_seq_len = match seq_len {
        Some(n) => n,
        None => {
            let ctx = qwen.cfg.max_position_embeddings as usize;
            // Defensive: if a GGUF reports max_position_embeddings = 0
            // (malformed metadata), fall back to the corpus length so
            // we still produce a single chunk — `forward_cpu` will
            // surface any actual capacity issue.
            if ctx == 0 {
                n_tokens
            } else {
                ctx.min(n_tokens)
            }
        }
    };
    if effective_seq_len == 0 {
        // Unreachable given the validations above (n_tokens >= 2 and
        // seq_len > 0 if Some), but the explicit check keeps the
        // chunking loop's invariant local.
        return Err(PplDriverError::Invalid(
            "resolved effective seq_len is 0".to_string(),
        ));
    }

    // --- 5. Chunked forward pass + logits accumulation -------------
    //
    // Standard non-overlapping PPL evaluation:
    //
    //   chunk i covers tokens[s..s+L] (s = i * effective_seq_len, L = chunk len)
    //   forward_cpu produces logits[0..L] of shape [L, vocab]
    //   prediction alignment: logits[j] predicts tokens[s + j + 1]
    //   ⇒ contributions = (logits[0..L-1], tokens[s+1..s+L])
    //
    // The very last token of the very last window has no target
    // (nothing comes after it), so we drop one row per window
    // including the last — this aligns with the standard convention
    // and matches `compute_perplexity`'s `logits.len() == targets.len()`
    // contract.
    let total_chunks = chunk_count(n_tokens, effective_seq_len);
    debug_assert!(total_chunks >= 1, "n_tokens >= 2 and seq_len >= 1 ⇒ at least one chunk");

    // Pre-allocate the accumulators once. Capacity = number of
    // (logits, target) PAIRS = n_tokens - total_chunks (we drop the
    // last row of every window's logits + the first token of every
    // window's targets).
    let pairs_capacity = n_tokens.saturating_sub(total_chunks);
    let mut all_logits: Vec<Vec<f32>> = Vec::with_capacity(pairs_capacity);
    let mut all_targets: Vec<u32> = Vec::with_capacity(pairs_capacity);

    for chunk_idx in 0..total_chunks {
        let start = chunk_idx * effective_seq_len;
        // Last chunk may be shorter than effective_seq_len when the
        // corpus length isn't a multiple of the window. We honor
        // forward_cpu's contract that tokens is non-empty by skipping
        // any chunk of length 0 (which can't happen with chunk_count
        // > 0 anyway, but is a defensive belt).
        let end = (start + effective_seq_len).min(n_tokens);
        if start >= end {
            continue;
        }
        let window = &tokens[start..end];
        let window_len = window.len();
        if window_len < 2 {
            // A chunk with only 1 token contributes 0 prediction pairs.
            // `forward_cpu` accepts it (it requires non-empty input)
            // but the alignment step below would discard everything.
            // Skip the call entirely to avoid a spurious GPU dispatch
            // and to keep the loop invariant simple.
            continue;
        }
        let positions = text_positions(window_len as u32);

        let chunk_logits = qwen
            .forward_cpu(window, &positions)
            .map_err(|e| PplDriverError::Forward {
                chunk: chunk_idx,
                cause: format!("{e:#}"),
            })?;

        // Shape sanity. `forward_cpu` returns `[seq_len * vocab_size]`.
        let expected_logits_len = window_len * vocab_size;
        if chunk_logits.len() != expected_logits_len {
            return Err(PplDriverError::Forward {
                chunk: chunk_idx,
                cause: format!(
                    "forward_cpu returned {} logits; expected {} ({} tokens × {} vocab)",
                    chunk_logits.len(),
                    expected_logits_len,
                    window_len,
                    vocab_size,
                ),
            });
        }

        // Reshape into per-position rows. Drop the last row (no
        // target after it within this window) and pair with
        // tokens[start+1..end] as targets.
        for row_idx in 0..window_len - 1 {
            let row_start = row_idx * vocab_size;
            let row_end = row_start + vocab_size;
            all_logits.push(chunk_logits[row_start..row_end].to_vec());
            all_targets.push(tokens[start + row_idx + 1]);
        }
        // Drop the chunk's logits buffer once we've copied the rows
        // we needed; for a 27B-dense model + 8k vocab this is a
        // sizable allocation and we want it freed before the next
        // forward pass.
        drop(chunk_logits);
    }

    if all_logits.is_empty() {
        // Defensive: should be unreachable given tokens.len() >= 2
        // and effective_seq_len >= 1, but explicitly surfacing this
        // is better than handing compute_perplexity an empty input
        // and getting a less-informative `EmptySequence` back.
        return Err(PplDriverError::Invalid(
            "no (logits, target) pairs produced from corpus + windowing; \
             tokens.len() may be smaller than expected for the chosen seq_len"
                .to_string(),
        ));
    }
    debug_assert_eq!(
        all_logits.len(),
        all_targets.len(),
        "internal invariant: logits and targets must be the same length"
    );

    // --- 6. Compute perplexity ------------------------------------
    let ppl_f64 = compute_perplexity(&all_logits, &all_targets)?;

    // --- 7. Narrow to f32 at the type boundary --------------------
    Ok(ppl_f64 as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pure-arithmetic chunking tests. No model load, no IO. The
    // integration crate at `tests/ppl_driver.rs` re-exercises these
    // via the public `chunk_count` API to lock the math from outside
    // the crate; the in-crate tests below also cover edge cases that
    // don't merit a full integration test.

    #[test]
    fn chunk_count_zero_n_tokens_is_zero() {
        assert_eq!(chunk_count(0, 4), 0);
    }

    #[test]
    fn chunk_count_zero_seq_len_is_zero() {
        assert_eq!(chunk_count(100, 0), 0);
    }

    #[test]
    fn chunk_count_exact_multiple() {
        // 512 / 128 = 4 chunks exactly.
        assert_eq!(chunk_count(512, 128), 4);
    }

    #[test]
    fn chunk_count_rounds_up_partial_window() {
        // 513 / 128 = 4.x ⇒ 5 chunks (the 5th has 1 token).
        assert_eq!(chunk_count(513, 128), 5);
    }

    #[test]
    fn chunk_count_corpus_smaller_than_window() {
        // 100 tokens with 1024-window ⇒ 1 chunk of length 100.
        assert_eq!(chunk_count(100, 1024), 1);
    }

    #[test]
    fn chunk_count_window_one() {
        // Degenerate but well-defined.
        assert_eq!(chunk_count(7, 1), 7);
    }

    #[test]
    fn measure_ppl_returns_invalid_on_short_input() {
        // < 2 tokens ⇒ Invalid (no prediction possible).
        let result = measure_ppl_qwen35_dense(
            std::path::Path::new("/nonexistent/model.gguf"),
            &[],
            None,
        );
        assert!(matches!(result, Err(PplDriverError::Invalid(_))));

        let result = measure_ppl_qwen35_dense(
            std::path::Path::new("/nonexistent/model.gguf"),
            &[42u32],
            None,
        );
        assert!(matches!(result, Err(PplDriverError::Invalid(_))));
    }

    #[test]
    fn measure_ppl_returns_invalid_on_zero_seq_len_override() {
        let result = measure_ppl_qwen35_dense(
            std::path::Path::new("/nonexistent/model.gguf"),
            &[1u32, 2, 3, 4],
            Some(0),
        );
        match result {
            Err(PplDriverError::Invalid(msg)) => assert!(msg.contains("seq_len")),
            other => panic!("expected Invalid(seq_len ...), got {other:?}"),
        }
    }

    #[test]
    fn measure_ppl_returns_gguf_on_missing_path() {
        let missing =
            std::path::Path::new("/nonexistent/path/that/cannot/exist/qwen35-dense.gguf");
        let result = measure_ppl_qwen35_dense(missing, &[1u32, 2, 3, 4], Some(2));
        match result {
            Err(PplDriverError::Gguf { path, source: _ }) => {
                assert_eq!(path, missing.to_path_buf());
            }
            other => panic!("expected Gguf {{ ... }}, got {other:?}"),
        }
    }
}
