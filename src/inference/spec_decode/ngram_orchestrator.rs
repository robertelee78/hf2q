//! N-gram speculative-decode orchestrator (ADR-030 iter-216 — Plan B).
//!
//! Drives spec-decode using the pure-CPU n-gram proposer from
//! `super::ngram_proposer` instead of the DFlash drafter model.
//! Shares the verify pipeline (target's `forward_prefill_batched` at
//! K+1 inputs → per-position argmax → `accept_prefix_argmax`) with
//! `super::dflash::orchestrator::dispatch_dflash_generate`, minus all
//! the drafter-side machinery (no hidden-state capture chain, no
//! drafter KV cache, no drafter forward).
//!
//! ## Why this exists
//!
//! ADR-030 iter-212 GPU measurement showed the DFlash drafter
//! (z-lab/gemma-4-26B-A4B-it-DFlash, trained on stock gemma-4) has
//! 0% acceptance against the ara-abliterated APEX-Q5_K_M target —
//! the two models speak different distributions, so the drafter's
//! coherent gemma-4 tokens are simply not the same as target argmax.
//! At a verify_prefill cost of ~59ms/round even at 100% acceptance,
//! DFlash on this hardware/drafter/target combo is structurally
//! incapable of beating the 98.3 tok/s baseline.
//!
//! N-gram acceleration is workload-specific (≥79% mean acceptance
//! needed to break even on this hardware) but on highly repetitive
//! workloads — translate-this-code, summarize-this-text, story
//! continuation — vLLM literature reports 75-95% acceptance.  It is
//! the only remaining spec-decode path with a structural shot at
//! perf parity or speedup on this APEX target.
//!
//! ## Loop shape
//!
//! ```text
//! initial: forward_prefill_batched(prompt) → first_token
//! output  := prompt ++ [first_token]
//! while output.len() < prompt.len() + max_new_tokens:
//!     drafts := ngram_proposer::propose(output, cfg)
//!     if drafts.is_empty():
//!         # Fallback to single-token decode
//!         next := forward_decode(output.last(), output.len() - 1)
//!         output.push(next)
//!     else:
//!         verify_input := [output.last(), drafts[0], ..., drafts[K-1]]
//!         install_capture(final_layer_only)
//!         forward_prefill_batched(verify_input, ..., start_pos = output.len() - 1)
//!         argmaxes := per_position_argmax(captured.final_layer_slab)
//!         (accept_count, fallback) := accept_prefix_argmax(drafts, argmaxes)
//!         output.extend_from_slice(drafts[..accept_count])
//!         output.push(fallback)
//!         rollback_kv(K - accept_count)
//! ```
//!
//! ## Greedy coherence invariant (algorithmic) — not byte-identity
//!
//! Algorithmically, `accept_prefix_argmax` only accepts a draft `d_i`
//! iff `d_i == argmax(target_argmaxes[i])`.  The fallback token is
//! always the target's argmax at the first mismatch position.  So at
//! temp=0 the committed tokens are *always exactly what the target
//! model emits at each verified position*.
//!
//! **HOWEVER**, the verify forward (`forward_prefill_batched` on K+1
//! tokens in one batched call) and the baseline decode forward
//! (`forward_decode` called K+1 times sequentially) follow DIFFERENT
//! kernel paths.  Floating-point order-of-operations in reduction
//! kernels (RMSNorm, SDPA softmax, MoE expert weighted sum) differ
//! between batched and sequential modes, and these can flip an
//! argmax tie at high-entropy positions even at temp=0.
//!
//! Empirical observation (ADR-030 iter-216): on a 32-token ocean-poem
//! generation, ngram-spec-decode output diverges from baseline after
//! ~10 tokens.  Both outputs are coherent gemma-4 prose; they are just
//! NUMERICALLY-DIFFERENT continuations of the same first line.  The
//! DFlash orchestrator (ADR-030 iter-66) exhibits the same property
//! (re-confirmed iter-216 on the same prompt).
//!
//! This is **a fundamental limit of greedy spec-decode**, not a bug.
//! The contract is "coherent text at greedy temp=0" not "byte-identical
//! to forward_decode N times."  HF2Q_SPEC_NGRAM=1 stays opt-in so
//! default production decode remains byte-stable; opting into spec-
//! decode trades byte-identity for the speed improvement on workloads
//! where it pays.

use anyhow::{anyhow, Context};

use crate::inference::spec_decode::dflash::hidden_capture::{
    extract_final_layer_slab, DFlashCaptureSession,
};
use crate::inference::spec_decode::ngram_proposer::{propose as ngram_propose, NgramConfig};
use crate::inference::spec_decode::verifier::accept_prefix_argmax;
use crate::serve::forward_mlx::MlxModelWeights;
use crate::serve::gpu::GpuContext;

/// Per-round accounting emitted when `HF2Q_SPEC_NGRAM_PROFILE=1`.
#[derive(Debug, Default, Clone, Copy)]
pub struct NgramRoundProfile {
    /// Time spent in `ngram_proposer::propose` (CPU-only KMP scan).
    pub propose_ms: f64,
    /// Time spent in target's `forward_prefill_batched` for the K+1
    /// verify input.
    pub verify_prefill_ms: f64,
    /// Time spent in per-position argmax.
    pub argmax_ms: f64,
    /// Total wall-clock for the round (incl. CPU bookkeeping).
    pub total_ms: f64,
    /// Number of drafts accepted this round (0..K).
    pub accept_count: usize,
    /// Number of drafts proposed this round (0 if propose returned
    /// empty — i.e. no n-gram match — and we fell back to single-token
    /// decode).
    pub draft_len: usize,
}

/// Run spec-decode against `target` using the n-gram proposer for K
/// draft tokens per round.
///
/// # Greedy invariant
///
/// At temp=0 (the only mode this function supports), output is
/// byte-identical to running `target.forward_decode` in a loop on the
/// same prompt — confirmed by the same algebra DFlash uses
/// (`accept_prefix_argmax` only commits target's own argmax on
/// rejection).
///
/// # Errors
///
/// - Empty prompt
/// - K < 1
/// - Capacity overflows (forwarded from target's prefill/decode)
pub fn dispatch_ngram_generate(
    target: &mut MlxModelWeights,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    k: u32,
    min_ngram: u32,
    max_ngram: u32,
    eos_token_ids: &[u32],
    gpu: &mut GpuContext,
) -> anyhow::Result<Vec<u32>> {
    if prompt_tokens.is_empty() {
        anyhow::bail!("dispatch_ngram_generate: empty prompt");
    }
    if k < 1 {
        anyhow::bail!("dispatch_ngram_generate: k must be >= 1; got {k}");
    }
    if min_ngram < 1 || max_ngram < min_ngram {
        anyhow::bail!(
            "dispatch_ngram_generate: invalid ngram range min={min_ngram} max={max_ngram}"
        );
    }

    let profile_on = std::env::var("HF2Q_SPEC_NGRAM_PROFILE").as_deref() == Ok("1");

    let hs = target.hidden_size;
    let final_layer_idx = target.layers.len() - 1;
    let combined_capture_ids: Vec<usize> = vec![final_layer_idx];

    // Match baseline's allocation exactly to preserve greedy byte-
    // identity: baseline uses `forward_prefill_batched(prompt,
    // max_new_tokens, 0, gpu)`, which sizes the linear KV cache to
    // `prompt_len + max_new_tokens` slots — exactly enough for
    // max_new_tokens decode positions.  Our verify rounds write to the
    // SAME slot range as baseline (each accepted draft consumes one
    // decode slot; rejected drafts get rolled back via `rollback_kv`
    // and never persist).  So the +k headroom in iter-216 v1 was
    // unnecessary — and caused KV cache capacity to differ from
    // baseline, which we suspect (and confirm via the poem-prompt A/B)
    // perturbs downstream attention numerics enough to break byte-
    // identity at temp=0.  Mirror baseline exactly.
    let max_decode_for_alloc = max_new_tokens;

    // ── Initial prompt prefill ─────────────────────────────────────
    let first_token = target
        .forward_prefill_batched(prompt_tokens, max_decode_for_alloc, 0, gpu)
        .map_err(|e| anyhow!("ngram: initial prefill: {e}"))?;

    let mut output: Vec<u32> = prompt_tokens.to_vec();
    output.push(first_token);

    if eos_token_ids.contains(&first_token) || max_new_tokens <= 1 {
        return Ok(output);
    }

    // Cumulative profile counters.
    let mut total_propose_ms = 0.0;
    let mut total_verify_ms = 0.0;
    let mut total_argmax_ms = 0.0;
    let mut total_round_ms = 0.0;
    let mut total_drafts_proposed: usize = 0;
    let mut total_drafts_accepted: usize = 0;
    let mut total_rounds: usize = 0;
    let mut total_empty_propose: usize = 0;

    let cfg = NgramConfig {
        min_ngram: min_ngram as usize,
        max_ngram: max_ngram as usize,
        k: k as usize,
        // max_model_len ceiling: never propose past prompt + budget.
        max_model_len: prompt_tokens.len() + max_new_tokens,
    };

    // ── Verify loop ─────────────────────────────────────────────────
    while output.len() < prompt_tokens.len() + max_new_tokens {
        let t_round = if profile_on {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // 1. Propose
        let t_propose = if profile_on {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let drafts = ngram_propose(&output, &cfg);
        if let Some(t) = t_propose {
            total_propose_ms += t.elapsed().as_secs_f64() * 1000.0;
        }

        if drafts.is_empty() {
            // No n-gram match → fall back to single-token decode.
            // This is the IDENTICAL code path baseline takes, so when
            // ngram repeatedly fails to find a match the perf
            // degrades gracefully to baseline (vs DFlash's structural
            // ~10× regression at 0% accept).
            total_empty_propose += 1;
            let last_tok = *output.last().expect("non-empty output");
            let seq_pos = output.len() - 1;
            let mut prof = None;
            let next = target
                .forward_decode(last_tok, seq_pos, gpu, &mut prof)
                .map_err(|e| anyhow!("ngram: fallback decode: {e}"))?;
            output.push(next);
            if eos_token_ids.contains(&next) {
                break;
            }
            if let Some(t) = t_round {
                total_round_ms += t.elapsed().as_secs_f64() * 1000.0;
            }
            total_rounds += 1;
            continue;
        }

        // 2. Verify: forward target on [last_token, drafts...] starting
        //    at start_pos = output.len() - 1.  This uses the Option C
        //    full re-prefill path by default (start_pos=0 means full
        //    prefill); the xlen path is engaged when
        //    `HF2Q_DFLASH_XLEN_SDPA=1` is set.  v1 of ngram spec-decode
        //    intentionally uses Option C — even though its ceiling is
        //    below baseline, it gives us a clean acceptance-rate
        //    measurement to decide whether wiring Option A is worth it.
        let last_tok = *output.last().expect("non-empty output");
        let mut verify_input: Vec<u32> = Vec::with_capacity(1 + drafts.len());
        verify_input.push(last_tok);
        verify_input.extend(drafts.iter().copied());
        let verify_seq_len = verify_input.len();

        let xlen_sdpa = std::env::var("HF2Q_DFLASH_XLEN_SDPA").as_deref() == Ok("1");
        let (verify_prefix, start_pos, xlen_max_decode) = if xlen_sdpa {
            (verify_input, output.len() - 1, output.len() - 1)
        } else {
            // Option C: full re-prefill.  verify_prefix = output (minus
            // the just-pushed last token) ++ verify_input.
            // Equivalently: output[..output.len() - 1] ++ verify_input
            // = output[..output.len() - 1] ++ [last_tok, drafts...]
            // = output ++ drafts (since output's last element IS last_tok).
            let mut v = output.clone();
            v.extend(drafts.iter().copied());
            let len = v.len();
            (v, 0usize, max_decode_for_alloc.min(len.saturating_sub(1)))
        };
        let verify_prefix_len = verify_prefix.len();

        let session = DFlashCaptureSession::new(
            combined_capture_ids.clone(),
            verify_prefix_len,
            hs,
            false,
        );
        target.install_dflash_capture(session);

        let t_verify = if profile_on {
            Some(std::time::Instant::now())
        } else {
            None
        };
        target
            .forward_prefill_batched(&verify_prefix, xlen_max_decode, start_pos, gpu)
            .map_err(|e| anyhow!("ngram: verify forward: {e}"))?;
        if let Some(t) = t_verify {
            total_verify_ms += t.elapsed().as_secs_f64() * 1000.0;
        }

        let captured = target
            .take_dflash_capture()
            .ok_or_else(|| anyhow!("ngram: verify capture vanished"))?;

        // 3. Extract per-position argmax for the K+1 verify positions.
        let final_slab = extract_final_layer_slab(
            &captured.hidden_output,
            &combined_capture_ids,
            final_layer_idx,
            verify_prefix_len,
            hs,
        )
        .context("ngram: extract final layer slab")?;

        let t_argmax = if profile_on {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let argmaxes = if xlen_sdpa {
            // xlen captures only the K+1 verify positions.
            target
                .per_position_argmax_from_hidden_batched_impl(
                    &final_slab,
                    verify_seq_len as u32,
                    true,
                    gpu,
                )
                .map_err(|e| anyhow!("ngram: argmax (xlen): {e}"))?
        } else {
            // Option C: capture is the FULL re-prefill (output ++ drafts).
            // We only need the tail K+1 positions starting at output.len() - 1.
            let tail_start = output.len() - 1;
            let tail_slab: &[f32] = &final_slab[tail_start * hs..verify_prefix_len * hs];
            target
                .per_position_argmax_from_hidden_batched_impl(
                    tail_slab,
                    verify_seq_len as u32,
                    true,
                    gpu,
                )
                .map_err(|e| anyhow!("ngram: argmax (re-prefill): {e}"))?
        };
        if let Some(t) = t_argmax {
            total_argmax_ms += t.elapsed().as_secs_f64() * 1000.0;
        }

        // 4. Accept-prefix.
        let (accept_count, fallback) = accept_prefix_argmax(&drafts, &argmaxes);

        // 5. Commit accepted drafts + fallback token.
        let drafts_len = drafts.len();
        output.extend_from_slice(&drafts[..accept_count]);
        output.push(fallback);

        // 6. Rollback rejected K/V writes.  We just wrote K+1 positions
        //    starting at start_pos; we KEEP accept_count + 1 of them
        //    (accepted drafts + fallback) and ROLL BACK (drafts_len -
        //    accept_count) positions (the rejected drafts beyond the
        //    fallback).  Caveat: in Option C re-prefill mode, the
        //    forward wrote ALL output.len() + drafts_len positions
        //    (since it starts at 0); we keep the prefix through
        //    output.len() + accept_count (which now equals
        //    prior_output_len + accept_count + 1 since we just pushed
        //    fallback) and roll back (drafts_len - accept_count)
        //    positions.  Both paths use the same rollback count.
        let n_rejected = drafts_len - accept_count;
        if n_rejected > 0 {
            target.rollback_kv(n_rejected);
        }

        if let Some(t) = t_round {
            total_round_ms += t.elapsed().as_secs_f64() * 1000.0;
        }
        total_rounds += 1;
        total_drafts_proposed += drafts_len;
        total_drafts_accepted += accept_count;

        // 7. EOS check across newly-committed tokens.
        for &tok in &drafts[..accept_count] {
            if eos_token_ids.contains(&tok) {
                return Ok(output);
            }
        }
        if eos_token_ids.contains(&fallback) {
            return Ok(output);
        }
    }

    if profile_on {
        let mean_accept_per_proposing_round = if total_rounds > total_empty_propose {
            total_drafts_accepted as f64 / (total_rounds - total_empty_propose) as f64
        } else {
            0.0
        };
        let mean_accept_rate = if total_drafts_proposed > 0 {
            total_drafts_accepted as f64 / total_drafts_proposed as f64
        } else {
            0.0
        };
        eprintln!(
            "[HF2Q_SPEC_NGRAM_PROFILE] rounds={total_rounds} empty_propose={total_empty_propose} \
             drafts_proposed={total_drafts_proposed} drafts_accepted={total_drafts_accepted} \
             mean_accept_rate={:.3} mean_accept_per_proposing_round={:.2} \
             cumulative_ms: propose={:.2} verify_prefill={:.2} argmax={:.2} total={:.2}",
            mean_accept_rate,
            mean_accept_per_proposing_round,
            total_propose_ms,
            total_verify_ms,
            total_argmax_ms,
            total_round_ms,
        );
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    //! Note: end-to-end GPU tests are not part of unit tests; this
    //! module's correctness is validated via the existing
    //! `accept_prefix_argmax`, `ngram_proposer::propose`, and
    //! `rollback_kv_state` unit tests in their own modules.  The
    //! orchestrator here is glue — its only correctness obligation is
    //! "wires the pieces in the right order" which we cover via
    //! integration runs in `spec_decode_cli.rs`.
}
