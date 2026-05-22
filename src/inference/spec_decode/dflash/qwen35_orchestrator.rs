//! ADR-034 task #78 Step 3c.B (2026-05-21) — Qwen35 DFlash orchestrator.
//!
//! Parallel to [`super::orchestrator::dispatch_dflash_generate`], adapted
//! for the Qwen35Model + HybridKvCache + Qwen35DFlashTarget surface.
//!
//! ## Why a parallel orchestrator (not a trait extension)
//!
//! `dispatch_dflash_generate` was written against `MlxModelWeights` and
//! uses inherent methods that don't exist on `Qwen35Model`:
//! - `embed_tokens` — Qwen35 now exposes `embed_tokens_gpu`
//! - `forward_prefill_batched` — Qwen35 prefill goes through
//!   `forward_gpu_with_hidden_dflash` (or the verify-batched wrapper)
//! - `per_position_argmax_from_hidden_batched_impl` — Qwen35 now exposes
//!   `per_position_argmax_from_normed_hidden`
//!
//! Extending the [`super::target::DFlashTarget`] trait with these methods
//! would require plumbing GPU state through every impl. Qwen35's GPU
//! state lives in a `thread_local` cache (`with_gpu_cache_mut`) whereas
//! MlxModelWeights' lives in `&mut self`-owned activations. Keeping the
//! orchestrators parallel sidesteps the impedance mismatch and lets each
//! one stay direct in its own native style.
//!
//! ## Algorithm (Option A: xlen verify only)
//!
//! Qwen35's DeltaNet (Gated DeltaNet) layers have non-trivial recurrent
//! state. Option C (full re-prefill from `start_pos=0` each round) would
//! force a DeltaNet state reset per round — multi-second cost. Option A
//! (incremental K+1 verify at `start_pos = output.len() - 1`, with KV
//! rollback for rejected positions) keeps state intact across rounds.
//!
//! LA-state rollback uses the per-position capture buffer (task #90
//! Step 4c). The orchestrator pre-arranges `ensure_la_capture(K+1)`
//! before any verify; `Qwen35DFlashTarget::rollback_kv` then walks the
//! captured states back to the last accepted position.
//!
//! Per-round loop:
//!
//! ```text
//!  1. Build drafter block: [last_token, mask * (K-1)]   (length K+1)
//!  2. h = model.embed_tokens_gpu(block)                  → [K+1, hs]
//!  3. target_hidden_concat = extract_drafter_concat(
//!       prior_captured, combined_capture_ids,
//!       drafter_target_layer_ids, prior_ctx_len, hs)     [new rows only]
//!  4. h_final = dispatch_dflash_model_forward(
//!       h, target_hidden_concat, drafter_*)              → [K+1, hs]
//!  5. all_argmaxes = model.per_position_argmax_from_normed_hidden(
//!       h_final, K+1)
//!  6. drafts = all_argmaxes[1..]                          → length K
//!  7. verify_input = [last_token, drafts...]              → length K+1
//!  8. Install verify capture (K+1 positions, combined ids)
//!  9. target_argmaxes = target.forward_decode_verify_batched(
//!       verify_input, start_pos=output.len()-1)
//! 10. verify_captured = target.take_dflash_capture()
//! 11. round = step_round_from_argmaxes(drafts, target_argmaxes, eos)
//! 12. rollback = drafts.len() - round.accept_count
//! 13. target.rollback_kv(rollback)                       → full+mtp+LA
//! 14. output.extend(round.committed_tokens)
//! 15. prior_captured = append_capture_positions(
//!       prior_captured, verify_captured, n_committed)
//! ```
//!
//! ## Greedy byte-identity invariant
//!
//! At temperature=0, this orchestrator emits tokens byte-identical to
//! single-token Qwen35 decode for the same prompt. Proof chain:
//! - `step_round_from_argmaxes` only accepts a draft when it equals the
//!   target's argmax at that position.
//! - The "free" continuation at `accept_count` is the target's argmax —
//!   exactly what single-token decode would emit there.
//! - `Qwen35DFlashTarget::rollback_kv` (task #78 Step 3b) discards the
//!   rejected K - accept_count positions across full-attn + MTP + LA.
//!
//! ## Non-greedy (Metropolis-Hastings) is out of scope here
//!
//! MH stochastic acceptance (task #91 / ADR-034 Step 1+2) is wired
//! through the MTP K=1 path. DFlash uses the SAME greedy accept-prefix
//! verifier (`accept_prefix_argmax`) as `dispatch_dflash_generate`. If
//! MH for DFlash is needed later, it'd hook in at step 11 above.

use anyhow::{anyhow, Context, Result};
use mlx_native::DType;

use super::config::DFlashConfig;
use super::hidden_capture::{
    append_capture_positions, extract_drafter_concat, DFlashCaptureSession,
};
use super::kv_cache::DFlashKvCache;
use super::orchestrator::step_round_from_argmaxes;
use super::qwen35_target::Qwen35DFlashTarget;
use super::target::DFlashTarget;
use super::tensors::DFlashModelTensors;
use crate::serve::gpu::GpuContext;

/// ADR-034 task #78 Step 3c.B (2026-05-21) — end-to-end Qwen35 DFlash
/// generate.
///
/// Composes initial prefill + per-round drafter→verify→accept loop into
/// one entry point. Mirrors [`super::orchestrator::dispatch_dflash_generate`]
/// for the Qwen35Model + HybridKvCache surface.
///
/// # Arguments
///
/// - `target`: pre-constructed `Qwen35DFlashTarget` wrapping mutable refs
///   to the verifier model + its kv_cache. Capture session must NOT be
///   pre-installed (this function manages capture lifetime).
/// - `drafter_*`: drafter weights / cache / config (z-lab DFlash drafter).
/// - `prompt_tokens`: text prompt, length P ≥ 1.
/// - `max_new_tokens`: budget for generated tokens (excludes prompt).
/// - `block_size`: K + 1 where K is the number of draft tokens per round
///   (typically 8, must be ≥ 2).
/// - `eos_token_ids`: stop conditions.
/// - `gpu`: `GpuContext` — present for trait API compatibility with
///   `forward_decode_verify_batched`; ignored by the Qwen35 impl
///   (Qwen35Model uses its own `thread_local` GPU_CACHE).
///
/// # Returns
///
/// `Vec<u32>` of length `prompt_tokens.len() + n_generated` where
/// `n_generated ≤ max_new_tokens`.
///
/// # Errors
///
/// - `prompt_tokens` empty or `block_size < 2`.
/// - Initial prefill or any per-round forward failure.
/// - Drafter forward / argmax failure.
/// - KV cache rollback overflow (shouldn't happen if invariants hold).
pub fn dispatch_qwen35_dflash_generate(
    target: &mut Qwen35DFlashTarget<'_>,
    drafter_tensors: &DFlashModelTensors,
    drafter_cache: &mut DFlashKvCache,
    drafter_cfg: &DFlashConfig,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    block_size: u32,
    eos_token_ids: &[u32],
    gpu: &mut GpuContext,
) -> Result<Vec<u32>> {
    if prompt_tokens.is_empty() {
        anyhow::bail!("dispatch_qwen35_dflash_generate: empty prompt");
    }
    if block_size < 2 {
        anyhow::bail!(
            "dispatch_qwen35_dflash_generate: block_size must be >= 2 (got {})",
            block_size,
        );
    }

    let hs = target.model.cfg.hidden_size as usize;
    let final_layer_idx = target.model.layers.len() - 1;

    // ─── Combined capture set: drafter target_layer_ids ∪ {final_layer_idx} ───
    let mut combined_capture_ids: Vec<usize> = drafter_cfg.target_layer_ids.clone();
    if !combined_capture_ids.contains(&final_layer_idx) {
        combined_capture_ids.push(final_layer_idx);
    }
    combined_capture_ids.sort_unstable();
    combined_capture_ids.dedup();

    // Validate target_layer_ids in-bounds vs n_layers.
    let n_layers = target.model.layers.len();
    for &lid in &combined_capture_ids {
        if lid >= n_layers {
            anyhow::bail!(
                "dispatch_qwen35_dflash_generate: combined_capture_ids contains \
                 layer {} >= n_layers={}",
                lid,
                n_layers,
            );
        }
    }

    // ─── Prime GPU cache + LA capture before any verify ─────────────
    target
        .model
        .ensure_gpu_cache_primed()
        .context("ensure_gpu_cache_primed")?;
    // `ensure_la_capture` needs the model's `MlxDevice` — fetch from
    // GPU_CACHE. block_size is the worst-case per-round verify size
    // (covers initial prefill which uses prompt_len ≥ 1; LA capture
    // window only needs to cover ONE verify forward at a time, since
    // rollback happens before the next forward).
    target.model.with_gpu_cache_mut(|device, _reg| {
        target
            .kv_cache
            .ensure_la_capture(&target.model.cfg, device, block_size)
    })?;

    let mut output: Vec<u32> = prompt_tokens.to_vec();

    // ─── Initial prefill with capture (start_pos = 0) ───────────────
    target.install_dflash_capture(DFlashCaptureSession::new(
        combined_capture_ids.clone(),
        prompt_tokens.len(),
        hs,
        false,
    ));
    let initial_argmaxes = target
        .forward_decode_verify_batched(prompt_tokens, 0, gpu)
        .context("initial prefill forward")?;
    let first_token = *initial_argmaxes
        .last()
        .ok_or_else(|| anyhow!("initial prefill: empty argmaxes"))?;
    let mut prior_captured = target
        .take_dflash_capture()
        .ok_or_else(|| anyhow!("initial prefill: capture vanished"))?;
    debug_assert_eq!(prior_captured.seq_len, prompt_tokens.len());

    output.push(first_token);
    if eos_token_ids.contains(&first_token) || max_new_tokens == 0 {
        return Ok(output);
    }

    let mut last_token = first_token;
    let drafter_target_layer_ids = drafter_cfg.target_layer_ids.clone();
    let n_target_layers = drafter_target_layer_ids.len();
    let row_stride = n_target_layers * hs;

    // ─── Multi-round loop (Option A) ────────────────────────────────
    let profile_on = std::env::var("HF2Q_DFLASH_PROFILE").as_deref() == Ok("1");
    let mut rounds_count = 0usize;
    let mut t_embed_ms = 0.0f64;
    let mut t_extract_ms = 0.0f64;
    let mut t_drafter_fwd_ms = 0.0f64;
    let mut t_drafter_argmax_ms = 0.0f64;
    let mut t_verify_ms = 0.0f64;
    let mut t_trim_ms = 0.0f64;

    while output.len() - prompt_tokens.len() < max_new_tokens {
        rounds_count += 1;

        // 1. Drafter input block: [last_token, mask × (K-1)]
        let t0 = profile_on.then(std::time::Instant::now);
        let mut block: Vec<u32> = Vec::with_capacity(block_size as usize);
        block.push(last_token);
        block.extend(
            std::iter::repeat(drafter_cfg.mask_token_id).take((block_size - 1) as usize),
        );
        let h = target
            .model
            .embed_tokens_gpu(&block)
            .context("drafter embed")?;
        if let Some(t) = t0 {
            t_embed_ms += t.elapsed().as_secs_f64() * 1000.0;
        }

        // 2. Drafter context = prior_captured (committed prefix).
        //    Incremental: drafter_cache tracks consumed prefix length.
        let prior_ctx_len = prior_captured.seq_len;
        debug_assert_eq!(
            prior_ctx_len,
            output.len() - 1,
            "prior_captured stale: seq_len={} but output.len()-1={}",
            prior_ctx_len,
            output.len() - 1,
        );
        let drafter_cached_seq_len = drafter_cache.layers[0].seq_len as usize;
        debug_assert!(
            prior_ctx_len >= drafter_cached_seq_len,
            "drafter cache regressed: cached={} prior_ctx_len={}",
            drafter_cached_seq_len,
            prior_ctx_len,
        );
        let drafter_new_rows = prior_ctx_len - drafter_cached_seq_len;

        // 3. Build target_hidden_concat on GPU (new rows only).
        let t0 = profile_on.then(std::time::Instant::now);
        let drafter_concat_full = extract_drafter_concat(
            &prior_captured.hidden_output,
            &combined_capture_ids,
            &drafter_target_layer_ids,
            prior_ctx_len,
            hs,
        )?;
        let new_rows_start = drafter_cached_seq_len * row_stride;
        let drafter_concat_new: &[f32] = &drafter_concat_full[new_rows_start..];
        debug_assert_eq!(
            drafter_concat_new.len(),
            drafter_new_rows * row_stride,
            "drafter_concat_new length mismatch",
        );
        let target_hidden_concat = target.model.with_gpu_cache_mut(|device, _reg| {
            // Allocate at least 1 row to avoid 0-sized alloc on first
            // round when drafter_new_rows might be 0 (shouldn't happen
            // — prior_ctx_len >= 1 always — but defensive).
            let mut buf = device
                .alloc_buffer(
                    drafter_concat_new.len() * 4,
                    DType::F32,
                    vec![drafter_new_rows.max(1), row_stride],
                )
                .map_err(|e| {
                    anyhow!("alloc target_hidden_concat: {e}")
                })?;
            if drafter_new_rows > 0 {
                buf.as_mut_slice::<f32>()
                    .map_err(|e| anyhow!("target_hidden_concat slice: {e}"))?
                    .copy_from_slice(drafter_concat_new);
            }
            Ok(buf)
        })?;
        if let Some(t) = t0 {
            t_extract_ms += t.elapsed().as_secs_f64() * 1000.0;
        }

        // 4. Drafter forward.
        let t0 = profile_on.then(std::time::Instant::now);
        let h_final = target.model.with_gpu_cache_mut(|device, registry| {
            super::forward::dispatch_dflash_model_forward(
                registry,
                device,
                &h,
                &target_hidden_concat,
                drafter_tensors,
                drafter_cache,
                drafter_cfg,
                block_size,
                drafter_new_rows as u32,
            )
        })
        .context("drafter forward")?;
        if let Some(t) = t0 {
            t_drafter_fwd_ms += t.elapsed().as_secs_f64() * 1000.0;
        }

        // 5. Per-position argmax on drafter's h_final via target lm_head.
        //    `per_position_argmax_from_normed_hidden` skips final_norm
        //    (drafter applied its own norm). Drafts are positions 1..K+1
        //    (position 0 is `last_token`'s prediction which we already
        //    have).
        let t0 = profile_on.then(std::time::Instant::now);
        let h_final_host: Vec<f32> = {
            let slice = h_final
                .as_slice::<f32>()
                .map_err(|e| anyhow!("h_final slice: {e}"))?;
            slice.to_vec()
        };
        let expected_h_final_len = (block_size as usize) * hs;
        if h_final_host.len() != expected_h_final_len {
            anyhow::bail!(
                "drafter h_final length {} != block_size({}) * hs({}) = {}",
                h_final_host.len(),
                block_size,
                hs,
                expected_h_final_len,
            );
        }
        let all_argmaxes = target
            .model
            .per_position_argmax_from_normed_hidden(&h_final_host, block_size)
            .context("drafter argmax")?;
        let drafts: Vec<u32> = all_argmaxes[1..].to_vec();
        debug_assert_eq!(drafts.len(), (block_size - 1) as usize);
        if let Some(t) = t0 {
            t_drafter_argmax_ms += t.elapsed().as_secs_f64() * 1000.0;
        }

        // 6. Verify forward (Option A: K+1 tokens at start_pos =
        //    output.len() - 1).
        let t0 = profile_on.then(std::time::Instant::now);
        let verify_seq_len = block_size as usize;
        target.install_dflash_capture(DFlashCaptureSession::new(
            combined_capture_ids.clone(),
            verify_seq_len,
            hs,
            false,
        ));
        let mut verify_input = Vec::with_capacity(verify_seq_len);
        verify_input.push(last_token);
        verify_input.extend(drafts.iter().copied());
        let start_pos = output.len() - 1;
        let target_argmaxes = target
            .forward_decode_verify_batched(&verify_input, start_pos, gpu)
            .context("verify forward")?;
        let verify_captured = target
            .take_dflash_capture()
            .ok_or_else(|| anyhow!("verify capture vanished"))?;
        if let Some(t) = t0 {
            t_verify_ms += t.elapsed().as_secs_f64() * 1000.0;
        }

        // 7. Accept-prefix + EOS.
        let round = step_round_from_argmaxes(&drafts, &target_argmaxes, eos_token_ids);

        if profile_on {
            eprintln!(
                "[HF2Q_DFLASH_ACCEPT qwen35] round={} accept_count={}/{} \
                 drafts={:?} target_argmaxes={:?} committed={:?}",
                rounds_count,
                round.accept_count,
                drafts.len(),
                drafts,
                target_argmaxes,
                round.committed_tokens,
            );
        }

        // 8. KV rollback (Qwen35: full_attn + mtp + LA via capture).
        let rollback = drafts.len().saturating_sub(round.accept_count);
        if rollback > 0 {
            target.rollback_kv(rollback);
        }

        // 9. Append committed + update last_token.
        let n_committed = round.committed_tokens.len();
        output.extend(round.committed_tokens.iter().copied());
        last_token = *round
            .committed_tokens
            .last()
            .ok_or_else(|| anyhow!("committed_tokens empty (round invariant violated)"))?;

        if round.hit_eos {
            break;
        }
        if output.len() - prompt_tokens.len() >= max_new_tokens {
            break;
        }

        // 10. Append accepted positions from verify_captured onto
        //     prior_captured.
        let t0 = profile_on.then(std::time::Instant::now);
        prior_captured =
            append_capture_positions(&prior_captured, &verify_captured, n_committed)
                .context("append accepted positions")?;
        if let Some(t) = t0 {
            t_trim_ms += t.elapsed().as_secs_f64() * 1000.0;
        }
    }

    if profile_on && rounds_count > 0 {
        let n = rounds_count as f64;
        eprintln!(
            "[HF2Q_DFLASH_PROFILE qwen35] rounds={} per-round-ms: \
             embed={:.2} extract={:.2} drafter_fwd={:.2} \
             drafter_argmax={:.2} verify={:.2} trim={:.2} TOTAL={:.2}",
            rounds_count,
            t_embed_ms / n,
            t_extract_ms / n,
            t_drafter_fwd_ms / n,
            t_drafter_argmax_ms / n,
            t_verify_ms / n,
            t_trim_ms / n,
            (t_embed_ms + t_extract_ms + t_drafter_fwd_ms + t_drafter_argmax_ms
                + t_verify_ms + t_trim_ms)
                / n,
        );
    }

    // Truncate any overshoot from the last round.
    let max_total = prompt_tokens.len() + max_new_tokens;
    if output.len() > max_total {
        output.truncate(max_total);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    // Real-model integration tests live under
    // /opt/hf2q/tests/ — they require GGUF + drafter weights on disk.
    // This module ships only with unit-testable scaffolding; the
    // top-level orchestrator function is exercised end-to-end by the
    // `cell_b_qwen35_dflash_e2e_2026_05_21` integration test (see
    // /opt/hf2q/tests/ for the Qwen35 DFlash harness once Step 4
    // serve/mod.rs routing lands).
    //
    // Until then, the orchestrator is type-checked by `cargo check` +
    // the building-block tests (Qwen35Model::embed_tokens_gpu,
    // per_position_argmax_from_normed_hidden, forward_gpu_with_hidden_dflash,
    // and DFlashCaptureSession::* are individually validated in their
    // own modules' tests).
}
