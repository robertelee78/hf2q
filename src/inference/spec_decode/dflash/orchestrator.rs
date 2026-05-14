//! DFlash spec-decode orchestrator (ADR-030 Phase 4).
//!
//! Wires together:
//! - Drafter model forward ([`super::forward::dispatch_dflash_model_forward`])
//! - Target forward verify (target's `forward_decode_verify_batched`,
//!   to be plumbed in subsequent iter with hidden-state capture)
//! - Accept-prefix logic (existing [`crate::inference::spec_decode::verifier::accept_prefix_argmax`])
//! - KV cache rollback (drafter cache + target cache)
//!
//! ## Phase 4 loop structure
//!
//! Per spec-decode step:
//! 1. **Propose**: drafter generates K candidate tokens. For K=0, skip
//!    drafting entirely (degrades to single-token target decode).
//! 2. **Verify**: target forward on `[last_committed, d_1, …, d_K]` (K+1
//!    positions). Captures hidden states at `target_layer_ids` for the
//!    next round's drafter input. Returns per-position argmaxes.
//! 3. **Accept**: `accept_prefix_argmax(drafts, target_argmaxes)`
//!    returns `(accept_count, model_token)`. The accepted tokens are
//!    `drafts[..accept_count]`; the final token is `model_token`
//!    (target's free continuation at the first mismatch).
//! 4. **Commit**: append `accepted_tokens + [model_token]` to the
//!    output sequence; advance target KV by `accept_count + 1`
//!    positions; advance drafter KV similarly.
//! 5. **Rollback**: target KV by `K - accept_count` positions (the
//!    rejected proposals' KV writes); drafter cache by the same.
//!
//! ## Greedy byte-identity invariant (the coherence guarantee)
//!
//! At temperature=0, this orchestrator must produce **byte-identical**
//! output to single-token target decode. The mathematical guarantee:
//! `accept_prefix_argmax` only accepts a draft token `d_i` if it equals
//! `argmax(target_logits_at_i)`. The "free" token at position
//! `accept_count` is `argmax(target_logits_at_accept_count)` — exactly
//! what greedy single-token decode would emit. KV rollback after the
//! K-positional verify undoes only the rejected portion.
//!
//! Phase 4 first piece (this commit): orchestrator state struct + the
//! pure-math step round (propose → verify result → accept → next-state).
//! Target forward integration + KV rollback wiring land in subsequent
//! commits.

use anyhow::Context;
use crate::inference::spec_decode::verifier::accept_prefix_argmax;

/// Output of one spec-decode round.
#[derive(Debug, Clone, PartialEq)]
pub struct RoundResult {
    /// Tokens to commit to the output sequence for this round.
    /// Always at least one element (the model's "free" token at the
    /// first-mismatch position).
    pub committed_tokens: Vec<u32>,
    /// How many of the proposed drafts were accepted.
    /// `committed_tokens.len() == accept_count + 1`.
    pub accept_count: usize,
    /// True if the model's free token is an EOS marker; orchestrator
    /// should stop generating.
    pub hit_eos: bool,
}

/// EOS-aware [`accept_prefix_argmax`] wrapper.
///
/// Applies the standard greedy accept rule, then:
/// - If any token in `drafts[..accept_count]` is in `eos_token_ids`, truncate
///   acceptance at the first EOS and treat it as the model's free
///   continuation (no further generation).
/// - If `model_token` is in `eos_token_ids`, mark `hit_eos = true`.
pub fn step_round_from_argmaxes(
    drafts: &[u32],
    target_argmaxes: &[u32],
    eos_token_ids: &[u32],
) -> RoundResult {
    let (accept_count, model_token) = accept_prefix_argmax(drafts, target_argmaxes);

    // Check for EOS inside the accepted prefix — if a draft emitted EOS
    // and target agreed, generation stops at that draft.
    let mut effective_accept = accept_count;
    let mut effective_final: u32 = model_token;
    let mut hit_eos = false;
    for (i, &t) in drafts.iter().take(accept_count).enumerate() {
        if eos_token_ids.contains(&t) {
            effective_accept = i;
            effective_final = t;
            hit_eos = true;
            break;
        }
    }
    if !hit_eos && eos_token_ids.contains(&model_token) {
        hit_eos = true;
    }

    let mut committed = drafts[..effective_accept].to_vec();
    committed.push(effective_final);
    RoundResult {
        committed_tokens: committed,
        accept_count: effective_accept,
        hit_eos,
    }
}

/// One target-side spec-decode round: target verify on
/// `[last_committed, d_1, ..., d_K]`, accept-prefix, KV rollback.
///
/// Composes:
/// - `MlxModelWeights::forward_decode_verify_batched` (returns K+1
///   per-position argmaxes)
/// - `step_round_from_argmaxes` (greedy accept-prefix + EOS check)
/// - `MlxModelWeights::rollback_kv` (clears the rejected K/V writes)
///
/// The DRAFTER side — generating the `drafts` slice — is the
/// orchestrator caller's responsibility (they call the drafter via
/// `dispatch_dflash_model_forward` + apply target's lm_head on the
/// drafter's output + argmax per position). This function is the
/// target-side composition.
///
/// # Arguments
///
/// - `target`: MlxModelWeights with NO dflash_capture installed
///   (this function installs/takes its own internal session)
/// - `last_committed_token`: the token most recently committed to the
///   output sequence (becomes verify_input[0])
/// - `drafts`: K candidate tokens proposed by the drafter
/// - `current_seq_pos`: target's current KV write position; verify
///   appends at this offset
/// - `eos_token_ids`: stop conditions
/// - `gpu`: GpuContext (mlx-native exec + registry)
///
/// # Returns
///
/// `RoundResult{ committed_tokens, accept_count, hit_eos }`. Committed
/// tokens = `drafts[..accept_count] + [target_continuation]`. The
/// caller appends these to the output sequence and advances seq_pos
/// by `committed_tokens.len()`.
///
/// # Greedy byte-identity guarantee
///
/// At temperature=0 (greedy), this function emits tokens
/// byte-identical to single-token target decode for the same prompt.
/// Proof: `forward_decode_verify_batched` runs target's exact
/// dispatchers on K+1 tokens; argmax at each position is what
/// single-token decode would emit at that position; `accept_prefix_argmax`
/// only accepts drafts that match those argmaxes, falling through to
/// target's own argmax at the first mismatch.
pub fn dispatch_dflash_spec_decode_round_target_side(
    target: &mut crate::serve::forward_mlx::MlxModelWeights,
    last_committed_token: u32,
    drafts: &[u32],
    current_seq_pos: usize,
    eos_token_ids: &[u32],
    gpu: &mut crate::serve::gpu::GpuContext,
) -> anyhow::Result<RoundResult> {
    // 1. Build verify input: [last_committed, d_1, …, d_K]
    let mut verify_input = Vec::with_capacity(drafts.len() + 1);
    verify_input.push(last_committed_token);
    verify_input.extend_from_slice(drafts);

    // 2. Target verify: returns per-position argmaxes
    let argmaxes = target
        .forward_decode_verify_batched(&verify_input, current_seq_pos, gpu)
        .map_err(|e| anyhow::anyhow!("spec_decode_round: verify_batched: {e}"))?;

    // 3. Accept-prefix + EOS
    let round = step_round_from_argmaxes(drafts, &argmaxes, eos_token_ids);

    // 4. KV rollback: reject K - accept_count positions
    let rollback = drafts.len().saturating_sub(round.accept_count);
    if rollback > 0 {
        target.rollback_kv(rollback);
    }

    Ok(round)
}

/// THE Phase 4 end-to-end one-round orchestrator.
///
/// Composes all the proven building blocks into one spec-decode round:
///
/// ```text
///   1. embed_tokens([last, mask*K])           → h [block_size, hs] F32
///   2. dispatch_dflash_model_forward(
///        h, target_hidden_concat, drafter ... ) → h_final [block_size, hs] F32
///   3. per_position_argmax_from_hidden_opt(
///        h_final, block_size, false, gpu)     → all_argmaxes [block_size]
///   4. drafts = all_argmaxes[1..]              (Python logits_start=1 — skip pos 0)
///   5. dispatch_dflash_spec_decode_round_target_side(
///        target, last_token, drafts, ...)    → RoundResult (target verify + accept + KV rollback)
///   6. Drafter KV rollback by K - accept_count
/// ```
///
/// Returns the RoundResult; caller appends `committed_tokens` to the
/// output sequence and advances `current_seq_pos` by
/// `committed_tokens.len()`.
///
/// # Greedy byte-identity invariant (mantra-coherence)
///
/// At temperature=0 (no sampler — pure argmax), this function's
/// committed_tokens is byte-identical to what single-token target
/// decode would emit. Proof chain:
/// - Drafter is consulted but its drafts only commit when they match
///   target's argmax (per step_round_from_argmaxes)
/// - target.forward_decode_verify_batched is bit-exact same dispatcher
///   sequence as single-token tail
/// - target.rollback_kv discards rejected positions
///
/// # Arguments
///
/// - `target`: target's MlxModelWeights, mutable for verify_batched +
///   rollback_kv + capture install
/// - `drafter_tensors`/`drafter_cache`/`drafter_cfg`: drafter state
/// - `last_committed_token`: most recent token in the output sequence
/// - `target_hidden_concat`: pre-computed by caller from a prior
///   target verify's hidden capture (permuted via PrefillCapture)
/// - `ctx_chunk_size`: number of new ctx positions this round (= 1
///   per spec-decode step, or N for the initial prompt forward)
/// - `current_seq_pos`: target's current KV write position
/// - `block_size`: K+1 (drafts K + the warmup last_committed_token slot)
/// - `eos_token_ids`: stop conditions
/// - `gpu`: shared MlxGpu context
pub fn dispatch_dflash_one_round(
    target: &mut crate::serve::forward_mlx::MlxModelWeights,
    drafter_tensors: &super::tensors::DFlashModelTensors,
    drafter_cache: &mut super::kv_cache::DFlashKvCache,
    drafter_cfg: &super::config::DFlashConfig,
    last_committed_token: u32,
    target_hidden_concat: &mlx_native::MlxBuffer,
    ctx_chunk_size: u32,
    current_seq_pos: usize,
    block_size: u32,
    eos_token_ids: &[u32],
    gpu: &mut crate::serve::gpu::GpuContext,
) -> anyhow::Result<RoundResult> {
    if block_size < 2 {
        anyhow::bail!(
            "dispatch_dflash_one_round: block_size must be >= 2; got {block_size}"
        );
    }

    // -------- Step 1: build the draft block --------
    // [last_committed, mask, mask, ..., mask]
    let mut block: Vec<u32> = Vec::with_capacity(block_size as usize);
    block.push(last_committed_token);
    block.extend(std::iter::repeat(drafter_cfg.mask_token_id).take((block_size - 1) as usize));

    // -------- Step 2: embed via target's embed_tokens --------
    let h = target
        .embed_tokens(&block, gpu)
        .map_err(|e| anyhow::anyhow!("one_round: embed_tokens: {e}"))?;

    // -------- Step 3: drafter forward --------
    let h_final = {
        let (exec, reg) = gpu.split();
        let device = exec.device();
        super::forward::dispatch_dflash_model_forward(
            reg,
            device,
            &h,
            target_hidden_concat,
            drafter_tensors,
            drafter_cache,
            drafter_cfg,
            block_size,
            ctx_chunk_size,
        )
        .map_err(|e| anyhow::anyhow!("one_round: drafter forward: {e}"))?
    };

    // -------- Step 4: per-position argmax via target's lm_head --------
    // The drafter applied its own norm; pass apply_final_norm=false.
    let all_argmaxes: Vec<u32> = {
        let h_final_slice: &[f32] = h_final
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("one_round: h_final slice: {e}"))?;
        let host_copy: Vec<f32> = h_final_slice.to_vec();
        target
            .per_position_argmax_from_hidden_opt(&host_copy, block_size, false, gpu)
            .map_err(|e| anyhow::anyhow!("one_round: per_position argmax: {e}"))?
    };

    // -------- Step 5: drafts = argmaxes at positions 1..block_size --------
    // Position 0 is the embed(last_committed_token) → drafter prediction
    // for position 0, which corresponds to Python's logits_start=1 skip.
    let drafts: Vec<u32> = all_argmaxes[1..].to_vec();

    // -------- Step 6: target-side verify + accept + KV rollback --------
    let round = dispatch_dflash_spec_decode_round_target_side(
        target,
        last_committed_token,
        &drafts,
        current_seq_pos,
        eos_token_ids,
        gpu,
    )?;

    // Step 7: drafter KV cache does NOT roll back (see correctness
    // note in dispatch_dflash_spec_decode_round_target_side above).

    Ok(round)
}

/// Multi-round end-to-end DFlash spec-decode generation.
///
/// Wraps `dispatch_dflash_one_round` in the outer loop that:
/// 1. Runs the initial prompt forward on the TARGET with capture
///    installed at the drafter's `target_layer_ids` (and final layer)
/// 2. Permutes the initial capture into `target_hidden_concat`
/// 3. Loops `dispatch_dflash_one_round` until max_new_tokens or
///    hit_eos. After each round, re-captures target hidden from the
///    verify forward (now done internally by forward_decode_verify_batched)
///    to feed the next round's drafter input.
///
/// Returns the generated token vector `[prompt_tokens..., generated...]`
/// truncated at `max_new_tokens` generated or EOS.
///
/// # Initial-prompt capture
///
/// The first round needs `target_hidden_concat` from BEFORE any
/// spec-decode steps. We get this by installing a capture session
/// targeting `cfg.target_layer_ids` (drafter's = [1, 6, 11, 17, 22, 27])
/// and running `forward_prefill_batched` on the prompt. The capture
/// session's `hidden_output` is then permuted to `target_hidden_concat`.
///
/// # Per-round re-capture
///
/// Inside `dispatch_dflash_one_round`, `forward_decode_verify_batched`
/// installs its OWN capture session (final layer only) for the
/// per-position argmax. That session is consumed and dropped. For the
/// NEXT round, we need a fresh target_hidden_concat capturing the
/// drafter's target_layer_ids at the accepted positions.
///
/// **First cut (this iter)**: handles the first round only using
/// the initial-prompt capture. Multi-round re-capture is iter-53+
/// (requires modifying forward_decode_verify_batched to support
/// the full target_layer_ids list, OR calling forward_prefill_batched
/// directly with the orchestrator's capture session). For now this
/// function runs ONE round and returns; the iter-53+ caller will
/// loop until exhaustion.
pub fn dispatch_dflash_generate_one_round_with_initial_capture(
    target: &mut crate::serve::forward_mlx::MlxModelWeights,
    drafter_tensors: &super::tensors::DFlashModelTensors,
    drafter_cache: &mut super::kv_cache::DFlashKvCache,
    drafter_cfg: &super::config::DFlashConfig,
    prompt_tokens: &[u32],
    block_size: u32,
    eos_token_ids: &[u32],
    gpu: &mut crate::serve::gpu::GpuContext,
) -> anyhow::Result<RoundResult> {
    use super::hidden_capture::{DFlashCaptureSession, PrefillCapture};

    let hs = target.hidden_size;
    let num_target_layers = target.layers.len();

    // -------- Step 1: install capture session for drafter's target_layer_ids --------
    let mut capture_layer_ids: Vec<usize> = drafter_cfg.target_layer_ids.clone();
    capture_layer_ids.sort_unstable();
    capture_layer_ids.dedup();
    for &i in &capture_layer_ids {
        if i >= num_target_layers {
            anyhow::bail!(
                "generate_one_round: target_layer_id {} >= num_target_layers {}",
                i, num_target_layers
            );
        }
    }

    let session = DFlashCaptureSession::new(
        capture_layer_ids.clone(),
        prompt_tokens.len(),
        hs,
        false, // no per-position argmaxes needed for the ctx capture
    );
    target.install_dflash_capture(session);

    // -------- Step 2: run initial prompt forward → captures pf_hidden --------
    let last_committed_token = target
        .forward_prefill_batched(prompt_tokens, 0, 0, gpu)
        .map_err(|e| anyhow::anyhow!("generate: initial prompt forward: {e}"))?;

    let captured = target
        .take_dflash_capture()
        .ok_or_else(|| anyhow::anyhow!("generate: capture session vanished after prompt forward"))?;

    // -------- Step 3: permute capture to target_hidden_concat --------
    let concat_vec: Vec<f32> = {
        let view = PrefillCapture {
            target_layer_ids: &captured.target_layer_ids,
            hidden_output: &mut captured.hidden_output.clone(),
            per_position_argmaxes: None,
        };
        view.permute_to_concat(prompt_tokens.len(), hs)
    };
    let target_hidden_concat = {
        let (exec, _reg) = gpu.split();
        let dev = exec.device();
        let mut buf = dev
            .alloc_buffer(
                concat_vec.len() * 4,
                mlx_native::DType::F32,
                vec![prompt_tokens.len(), capture_layer_ids.len() * hs],
            )
            .map_err(|e| anyhow::anyhow!("generate: alloc target_hidden_concat: {e}"))?;
        buf.as_mut_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("generate: target_hidden_concat slice: {e}"))?
            .copy_from_slice(&concat_vec);
        buf
    };

    // -------- Step 4: dispatch one spec-decode round --------
    let current_seq_pos = prompt_tokens.len();
    let round = dispatch_dflash_one_round(
        target,
        drafter_tensors,
        drafter_cache,
        drafter_cfg,
        last_committed_token,
        &target_hidden_concat,
        prompt_tokens.len() as u32,
        current_seq_pos,
        block_size,
        eos_token_ids,
        gpu,
    )?;

    Ok(round)
}

/// Multi-round end-to-end DFlash spec-decode generation.
///
/// Composes ALL Phase 4 building blocks into a complete generation
/// loop. Runs until `max_new_tokens` are emitted OR `hit_eos`.
///
/// # Multi-round invariants (surfaced during iter-58/iter-59 review)
///
/// 1. **Target KV ↔ output lag invariant**: at the start of round N,
///    target's KV ends exactly at `seq_pos`, while `output[seq_pos..]`
///    contains the last committed token (= verify_input[0] for next
///    round, fed back via embed). seq_pos += `n_committed` per round;
///    target's KV grows then truncates to the same end position via
///    `rollback_kv(K - accept_count)`.
///
/// 2. **Drafter ctx trim invariant**: `captured.seq_len` must contain
///    only TARGET-ACCEPTED positions before feeding to drafter's next
///    round (else drafter sees rejected ctx). `trim_capture_to` in
///    step 10 enforces this — mirrors Python `model_mlx.py:567`
///    `hidden = hidden[:, :accepted + 1, :]`.
///
/// 3. **Drafter cache no-rollback invariant**: drafter's `cache.seq_len`
///    only ever holds accepted ctx (consequence of #2). The K+1 prop
///    K/V lives in slack via `write_slack_kv`, never advances seq_len,
///    gets overwritten on next round. Rolling back drafter cache
///    would erroneously truncate REAL accepted ctx state.
///
/// 4. **Captured-hidden permute invariant**: combined capture's
///    `hidden_output` is row-major `[combined_layer, t, dim]`.
///    `extract_drafter_concat` produces `[t, drafter_layer, dim]`
///    flat = `[t, drafter_layer * dim]` 2D-row-major, matching what
///    `dispatch_dflash_fc` expects.
///
/// All four invariants were verified by composition + careful
/// re-reading; #1 and #4 hold by construction; #2 and #3 required
/// explicit fixes (iter-58 + iter-59 commits).
///
/// Per-round, target's forward_prefill_batched is called ONCE with a
/// combined capture session covering drafter's `target_layer_ids` ∪
/// `[final_layer_idx]`. The combined buffer is split into:
/// - drafter's input for NEXT round (via `extract_drafter_concat`)
/// - per-position argmax input for THIS round (via `extract_final_layer_slab`)
///
/// This avoids double target-forward per round.
///
/// # Greedy byte-identity invariant
///
/// At temperature = 0, the output (after `prompt_tokens.len()`) is
/// byte-identical to what single-token target decode would produce.
/// Proven by composition: `step_round_from_argmaxes` only accepts
/// drafts that match target's argmax; `target.rollback_kv` discards
/// rejected positions; per-round capture is just additive
/// instrumentation that doesn't change target's compute.
///
/// # Arguments
///
/// - `target`: MlxModelWeights for the production target
/// - `drafter_tensors` / `drafter_cache` / `drafter_cfg`: drafter state
/// - `prompt_tokens`: input prompt
/// - `max_new_tokens`: cap on generated token count
/// - `block_size`: K + 1 (drafts K + warmup slot); default 8 for K=7
/// - `eos_token_ids`: stop conditions
/// - `gpu`: shared MlxGpu context
///
/// # Returns
///
/// `Vec<u32>` = `[prompt_tokens..., generated...]`.
pub fn dispatch_dflash_generate(
    target: &mut crate::serve::forward_mlx::MlxModelWeights,
    drafter_tensors: &super::tensors::DFlashModelTensors,
    drafter_cache: &mut super::kv_cache::DFlashKvCache,
    drafter_cfg: &super::config::DFlashConfig,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    block_size: u32,
    eos_token_ids: &[u32],
    gpu: &mut crate::serve::gpu::GpuContext,
) -> anyhow::Result<Vec<u32>> {
    use super::hidden_capture::{
        extract_drafter_concat, extract_final_layer_slab, DFlashCaptureSession,
    };

    if prompt_tokens.is_empty() {
        anyhow::bail!("dispatch_dflash_generate: empty prompt");
    }
    if block_size < 2 {
        anyhow::bail!("dispatch_dflash_generate: block_size must be >= 2");
    }

    let hs = target.hidden_size;
    let final_layer_idx = target.layers.len() - 1;

    // Combined capture: drafter's target_layer_ids ∪ [final_layer_idx]
    let mut combined_capture_ids: Vec<usize> = drafter_cfg.target_layer_ids.clone();
    if !combined_capture_ids.contains(&final_layer_idx) {
        combined_capture_ids.push(final_layer_idx);
    }
    combined_capture_ids.sort_unstable();
    combined_capture_ids.dedup();

    let mut output: Vec<u32> = prompt_tokens.to_vec();

    // ADR-030 iter-76: opt-in cross-length SDPA verify path (Option A).
    // When env flag HF2Q_DFLASH_XLEN_SDPA=1 is set, the orchestrator
    // shrinks verify_input to K+1 tokens at start_pos=output.len()-1,
    // rolls back target's KV cache by (K - accept_count) per round, and
    // grows persistent_captured incrementally.  Coordinated with
    // forward_prefill_batched's resume-SDPA branch (gated on same env).
    // Default OFF for safety; when OFF, behavior is iter-72 path bit-
    // identical.
    let xlen_sdpa = std::env::var("HF2Q_DFLASH_XLEN_SDPA").as_deref() == Ok("1");

    // -------- Initial prompt forward with capture --------
    let session = DFlashCaptureSession::new(
        combined_capture_ids.clone(),
        prompt_tokens.len(),
        hs,
        false,
    );
    target.install_dflash_capture(session);
    // Size full-attention KV cache via max_decode_tokens.  The
    // worst-case write extent across all verify rounds is at the LAST
    // round: seq_pos = prompt_len + max_new_tokens - 1 (writes through
    // seq_pos + block_size - 1).  So we need
    // `linear_capacity ≥ prompt_len + max_new_tokens + block_size - 1`
    // = prompt_len + (max_new_tokens + block_size - 1).
    //
    // `linear_capacity` is computed by forward_prefill_batched as
    // `seq_len + max_decode_tokens`, so passing
    // `max_decode_tokens = max_new_tokens + block_size - 1` gives the
    // right size.  Without this the verify rounds would overflow the
    // dense cache at full-attention layers (iter-63 root-cause: the
    // `kv_capacity (P) < kv_seq_len (P+1)` assertion that the coherence
    // gate originally surfaced).
    let max_decode_for_alloc = max_new_tokens + block_size as usize - 1;
    let first_token = target
        .forward_prefill_batched(prompt_tokens, max_decode_for_alloc, 0, gpu)
        .map_err(|e| anyhow::anyhow!("generate: initial prompt forward: {e}"))?;
    // Keep the initial prompt prefill's capture as round-1's prior_ctx.
    // iter-69: instead of running a SEPARATE prior_ctx prefill each
    // round (which was ~50% of round wall-clock), we use the previous
    // verify's capture (trimmed to the accepted-positions count) as
    // the drafter's input next round.  The initial capture seeds
    // round 1.
    let mut prior_captured = target
        .take_dflash_capture()
        .ok_or_else(|| anyhow::anyhow!("generate: initial capture vanished"))?;
    debug_assert_eq!(prior_captured.seq_len, prompt_tokens.len());

    output.push(first_token);

    if eos_token_ids.contains(&first_token) || max_new_tokens == 0 {
        return Ok(output);
    }

    let mut last_token = first_token;

    // iter-70: per-stage timing (gated on HF2Q_DFLASH_PROFILE=1).  Each
    // stage accumulates total wall-clock across all rounds; printed at
    // function exit.  Production default (env unset) is bit-identical
    // since the closures wrapping each timed region don't allocate any
    // Instants when profile_on is false.
    let profile_on = std::env::var("HF2Q_DFLASH_PROFILE").as_deref() == Ok("1");
    let mut t_embed_ms = 0.0f64;
    let mut t_extract_concat_ms = 0.0f64;
    let mut t_drafter_fwd_ms = 0.0f64;
    let mut t_drafter_argmax_ms = 0.0f64;
    let mut t_verify_prefill_ms = 0.0f64;
    let mut t_target_argmax_ms = 0.0f64;
    let mut t_trim_ms = 0.0f64;
    let mut rounds_count = 0usize;

    // -------- Multi-round loop (Option C + iter-69 single-prefill) --------
    //
    // Architecture:
    // - Re-prefill `[output + drafts]` from start_pos=0 each round.
    //   The SDPA self-attention spans the whole prefix → correct
    //   cross-context attention (bypasses iter-64 Bug 5).
    // - SINGLE prefill per round (iter-69): the verify prefill ALSO
    //   captures all positions [0..output_len + K).  Hidden states for
    //   positions [0..output_len - 1) form next round's prior_ctx;
    //   target argmaxes at positions [output_len - 1 .. output_len - 1
    //   + block_size) form this round's verify result.
    //
    // Drafter cache is reset each round and re-fed the full prior ctx
    // — its work is bounded but the K/V cache lifetime is per-round.
    while output.len() - prompt_tokens.len() < max_new_tokens {
        rounds_count += 1;
        let t0_embed = if profile_on { Some(std::time::Instant::now()) } else { None };
        // 1. Build the drafter's input block: [last_token, mask × K]
        let mut block: Vec<u32> = Vec::with_capacity(block_size as usize);
        block.push(last_token);
        block.extend(
            std::iter::repeat(drafter_cfg.mask_token_id)
                .take((block_size - 1) as usize),
        );
        let h = target
            .embed_tokens(&block, gpu)
            .map_err(|e| anyhow::anyhow!("generate: embed_tokens: {e}"))?;
        if let Some(t) = t0_embed { t_embed_ms += t.elapsed().as_secs_f64() * 1000.0; }

        // 2. Drafter's context = prior_captured (= prompt capture in
        //    round 1, OR trimmed verify capture from prior round).
        //    seq_len matches output.len() - 1 (= all committed tokens
        //    EXCEPT last_token, which is drafter's block[0] query).
        let prior_ctx_len = prior_captured.seq_len;
        debug_assert_eq!(
            prior_ctx_len, output.len() - 1,
            "prior_captured stale: seq_len={} but output.len()-1={}",
            prior_ctx_len, output.len() - 1,
        );

        // 3. Extract NEW drafter_concat rows from the prior capture and
        //    upload to GPU.
        //
        // iter-71: incremental drafter cache.  Instead of resetting the
        // drafter cache each round and re-feeding the FULL prior ctx,
        // we preserve cache state across rounds and append only NEW
        // positions per round (= prior_ctx_len - cached_seq_len).  The
        // drafter's RoPE offsets (prior_offset = cache_layer.seq_len)
        // advance correctly across rounds (see
        // forward.rs:798-849).  At 0% acceptance rate, new_rows=1 per
        // round vs the previous ~P+r rows → ~5-10× faster drafter_fwd.
        let t0_extract = if profile_on { Some(std::time::Instant::now()) } else { None };
        let drafter_cached_seq_len = drafter_cache.layers[0].seq_len as usize;
        debug_assert!(
            prior_ctx_len >= drafter_cached_seq_len,
            "drafter cache state regressed: cached={} prior_ctx_len={}",
            drafter_cached_seq_len, prior_ctx_len,
        );
        let drafter_new_rows = prior_ctx_len - drafter_cached_seq_len;
        let n_target_layers = drafter_cfg.target_layer_ids.len();
        let row_stride = n_target_layers * hs;
        let drafter_concat_vec_full = extract_drafter_concat(
            &prior_captured.hidden_output,
            &combined_capture_ids,
            &drafter_cfg.target_layer_ids,
            prior_ctx_len,
            hs,
        )?;
        // Take only the LAST drafter_new_rows rows (= positions
        // [drafter_cached_seq_len..prior_ctx_len) in the prior_captured
        // slab).  drafter_concat_vec_full is [prior_ctx_len, row_stride].
        let new_rows_start = drafter_cached_seq_len * row_stride;
        let drafter_concat_new: &[f32] = &drafter_concat_vec_full[new_rows_start..];
        debug_assert_eq!(
            drafter_concat_new.len(),
            drafter_new_rows * row_stride,
            "drafter_concat_new length mismatch",
        );
        let target_hidden_concat = {
            let (exec, _reg) = gpu.split();
            let dev = exec.device();
            let mut buf = dev
                .alloc_buffer(
                    drafter_concat_new.len() * 4,
                    mlx_native::DType::F32,
                    vec![drafter_new_rows.max(1), row_stride],
                )
                .map_err(|e| anyhow::anyhow!("generate: alloc target_hidden_concat: {e}"))?;
            if drafter_new_rows > 0 {
                buf.as_mut_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("generate: target_hidden_concat slice: {e}"))?
                    .copy_from_slice(drafter_concat_new);
            }
            buf
        };
        if let Some(t) = t0_extract { t_extract_concat_ms += t.elapsed().as_secs_f64() * 1000.0; }

        // 4. Drafter forward → h_final shape [block_size, hidden]
        let t0_drafter = if profile_on { Some(std::time::Instant::now()) } else { None };
        let h_final = {
            let (exec, reg) = gpu.split();
            let device = exec.device();
            super::forward::dispatch_dflash_model_forward(
                reg,
                device,
                &h,
                &target_hidden_concat,
                drafter_tensors,
                drafter_cache,
                drafter_cfg,
                block_size,
                drafter_new_rows as u32,
            )
            .context("generate: drafter forward")?
        };
        if let Some(t) = t0_drafter { t_drafter_fwd_ms += t.elapsed().as_secs_f64() * 1000.0; }

        // 5. lm_head per position on drafter's h_final → K drafts
        //    (index 0 is for last_token's position which we already
        //    know; we want predictions at positions 1..block_size).
        //    iter-71: route through batched argmax impl (one command
        //    buffer for all block_size positions instead of K+1
        //    separate commit-and-wait syncs).  Bit-identical to the
        //    un-batched path at temp=0 (verified by the e2e gate).
        let t0_drafter_argmax = if profile_on { Some(std::time::Instant::now()) } else { None };
        let drafts: Vec<u32> = {
            let h_final_slice: &[f32] = h_final
                .as_slice::<f32>()
                .map_err(|e| anyhow::anyhow!("generate: h_final slice: {e}"))?;
            let host_copy: Vec<f32> = h_final_slice.to_vec();
            let all_argmaxes = target
                .per_position_argmax_from_hidden_batched_impl(&host_copy, block_size, false, gpu)
                .map_err(|e| anyhow::anyhow!("generate: drafter argmax: {e}"))?;
            all_argmaxes[1..].to_vec()
        };
        if let Some(t) = t0_drafter_argmax { t_drafter_argmax_ms += t.elapsed().as_secs_f64() * 1000.0; }

        // 6. Verify forward — two paths gated on HF2Q_DFLASH_XLEN_SDPA.
        //
        // Option C (default, xlen_sdpa=false): re-prefill
        //   [output + drafts] from start_pos=0.  Verify capture covers
        //   all output_len + K positions; target_argmaxes are the last
        //   block_size rows.  NO target.rollback_kv (start_pos=0 next
        //   round overwrites).  prior_captured = trim(verify_captured,
        //   output.len()-1).
        //
        // Option A (iter-76, xlen_sdpa=true): verify_input = K+1
        //   tokens at start_pos=output.len()-1.  Verify capture covers
        //   just K+1 positions; target_argmaxes = ALL K+1 argmaxes.
        //   target.rollback_kv(K - accept_count) after accept-prefix.
        //   prior_captured = append_capture_positions(prior, verify,
        //   n_committed).  Requires forward_prefill_batched's resume-
        //   SDPA branch (also gated on HF2Q_DFLASH_XLEN_SDPA).
        let (verify_captured, target_argmaxes, verify_seq_len_for_path) = if xlen_sdpa {
            // ── Option A: cross-length verify ───────────────────────
            let verify_input: Vec<u32> = std::iter::once(last_token)
                .chain(drafts.iter().copied())
                .collect();
            let verify_seq_len = verify_input.len(); // = block_size = K+1
            let start_pos = output.len() - 1;
            let verify_session = DFlashCaptureSession::new(
                combined_capture_ids.clone(),
                verify_seq_len,
                hs,
                false,
            );
            target.install_dflash_capture(verify_session);
            let t0_verify = if profile_on { Some(std::time::Instant::now()) } else { None };
            // dense_kvs_vec is fresh-allocated per forward_prefill_batched call
            // with `linear_capacity = seq_len + max_decode_tokens`.  For the
            // xlen verify call, K/V writes go to positions
            // [start_pos..start_pos+seq_len) — so the cap needs to be ≥
            // start_pos + seq_len.  Pass max_decode_tokens = start_pos so
            // cap = seq_len + start_pos exactly.
            let xlen_max_decode = start_pos;
            let _verify_last_argmax = target
                .forward_prefill_batched(&verify_input, xlen_max_decode, start_pos, gpu)
                .map_err(|e| anyhow::anyhow!("generate: verify forward (xlen): {e}"))?;
            let captured = target
                .take_dflash_capture()
                .ok_or_else(|| anyhow::anyhow!("generate: verify capture vanished (xlen)"))?;
            if let Some(t) = t0_verify { t_verify_prefill_ms += t.elapsed().as_secs_f64() * 1000.0; }

            // target_argmaxes: ALL K+1 positions of the verify capture
            // (verify_input[0] = last_token, so argmax at position 0 =
            // pred-after-last_token; argmax at position i = pred-after-
            // verify_input[i] = compare to draft[i+1] or accept as
            // free-continuation when accept_count == K).
            let t0_target_argmax = if profile_on { Some(std::time::Instant::now()) } else { None };
            let final_slab = extract_final_layer_slab(
                &captured.hidden_output,
                &combined_capture_ids,
                final_layer_idx,
                verify_seq_len,
                hs,
            )?;
            let argmaxes = target
                .per_position_argmax_from_hidden_batched_impl(
                    &final_slab,
                    verify_seq_len as u32,
                    true,
                    gpu,
                )
                .map_err(|e| anyhow::anyhow!("generate: target argmax (xlen): {e}"))?;
            if let Some(t) = t0_target_argmax { t_target_argmax_ms += t.elapsed().as_secs_f64() * 1000.0; }

            (captured, argmaxes, verify_seq_len)
        } else {
            // ── Option C: full-prefix re-prefill (iter-65..iter-72) ──
            let mut verify_prefix: Vec<u32> = output.clone();
            verify_prefix.extend(drafts.iter().copied());
            let verify_prefix_len = verify_prefix.len();
            let verify_session = DFlashCaptureSession::new(
                combined_capture_ids.clone(),
                verify_prefix_len,
                hs,
                false,
            );
            target.install_dflash_capture(verify_session);
            let t0_verify = if profile_on { Some(std::time::Instant::now()) } else { None };
            let _verify_last_argmax = target
                .forward_prefill_batched(&verify_prefix, max_decode_for_alloc, 0, gpu)
                .map_err(|e| anyhow::anyhow!("generate: verify forward: {e}"))?;
            let captured = target
                .take_dflash_capture()
                .ok_or_else(|| anyhow::anyhow!("generate: verify capture vanished"))?;
            if let Some(t) = t0_verify { t_verify_prefill_ms += t.elapsed().as_secs_f64() * 1000.0; }

            let t0_target_argmax = if profile_on { Some(std::time::Instant::now()) } else { None };
            let final_slab = extract_final_layer_slab(
                &captured.hidden_output,
                &combined_capture_ids,
                final_layer_idx,
                verify_prefix_len,
                hs,
            )?;
            let verify_start = output.len() - 1;
            let verify_end = verify_start + block_size as usize;
            debug_assert_eq!(verify_end, verify_prefix_len);
            let verify_slab_tail: &[f32] = &final_slab[verify_start * hs..verify_end * hs];
            let argmaxes = target
                .per_position_argmax_from_hidden_batched_impl(
                    verify_slab_tail,
                    block_size,
                    true,
                    gpu,
                )
                .map_err(|e| anyhow::anyhow!("generate: target argmax: {e}"))?;
            if let Some(t) = t0_target_argmax { t_target_argmax_ms += t.elapsed().as_secs_f64() * 1000.0; }

            (captured, argmaxes, verify_prefix_len)
        };

        // 8. Accept-prefix
        let round = step_round_from_argmaxes(&drafts, &target_argmaxes, eos_token_ids);

        // ADR-030 iter-86 — per-round accept_count log (env-gated).
        if std::env::var("HF2Q_DFLASH_PROFILE").as_deref() == Ok("1") {
            eprintln!(
                "[HF2Q_DFLASH_ACCEPT] round={rounds_count} accept_count={}/{} \
                 drafts={drafts:?} target_argmaxes={target_argmaxes:?} \
                 committed={:?}",
                round.accept_count, drafts.len(), round.committed_tokens,
            );
        }

        // 9. Target rollback (Option A only — Option C re-prefills at
        //    start_pos=0 so no rollback needed).
        if xlen_sdpa {
            let rollback = drafts.len().saturating_sub(round.accept_count);
            if rollback > 0 {
                target.rollback_kv(rollback);
            }
        }

        // 10. Append committed tokens + advance state
        let n_committed = round.committed_tokens.len();
        output.extend(round.committed_tokens.iter().copied());
        last_token = *round.committed_tokens.last().unwrap();

        if round.hit_eos {
            break;
        }
        if output.len() - prompt_tokens.len() >= max_new_tokens {
            break;
        }

        // 11. Update prior_captured for next round.
        let t0_trim = if profile_on { Some(std::time::Instant::now()) } else { None };
        if xlen_sdpa {
            // Option A: persistent_captured grows by n_committed
            // positions per round (the accepted positions from this
            // round's verify capture).  The verify capture covers K+1
            // verify positions; the first n_committed of those are
            // accepted into output and need to be merged into the
            // persistent prior_captured slab.
            prior_captured = super::hidden_capture::append_capture_positions(
                &prior_captured,
                &verify_captured,
                n_committed,
            )?;
        } else {
            // Option C: trim verify_captured to the new output.len()-1
            // (causal masking in the start_pos=0 prefill makes positions
            // [0..N) depend ONLY on tokens [0..N) — independent of
            // drafts at higher positions).
            let mut next_captured = verify_captured;
            let next_prior_ctx_len = output.len() - 1;
            debug_assert!(next_prior_ctx_len <= next_captured.seq_len,
                "trim target {} > current {}", next_prior_ctx_len, next_captured.seq_len);
            super::hidden_capture::trim_capture_to(&mut next_captured, next_prior_ctx_len);
            prior_captured = next_captured;
        }
        if let Some(t) = t0_trim { t_trim_ms += t.elapsed().as_secs_f64() * 1000.0; }
        let _ = verify_seq_len_for_path; // silence unused warning when not branched on
    }

    if profile_on && rounds_count > 0 {
        let n = rounds_count as f64;
        eprintln!(
            "[HF2Q_DFLASH_PROFILE] rounds={} per-round-ms: embed={:.2} extract={:.2} drafter_fwd={:.2} drafter_argmax={:.2} verify_prefill={:.2} target_argmax={:.2} trim={:.2} TOTAL={:.2}",
            rounds_count,
            t_embed_ms / n,
            t_extract_concat_ms / n,
            t_drafter_fwd_ms / n,
            t_drafter_argmax_ms / n,
            t_verify_prefill_ms / n,
            t_target_argmax_ms / n,
            t_trim_ms / n,
            (t_embed_ms + t_extract_concat_ms + t_drafter_fwd_ms + t_drafter_argmax_ms
                + t_verify_prefill_ms + t_target_argmax_ms + t_trim_ms) / n,
        );
    }

    // Truncate to max_new_tokens if we overshot in the last round
    let max_total = prompt_tokens.len() + max_new_tokens;
    if output.len() > max_total {
        output.truncate(max_total);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::spec_decode::dflash::{
        config::DFlashConfig,
        forward::dispatch_dflash_model_forward,
        kv_cache::DFlashKvCache,
        tensors::DFlashModelTensors,
        weights::{DFlashWeights, DFlashWeightsFile},
    };
    use mlx_native::{DType, KernelRegistry, MlxDevice};

    #[test]
    fn k0_empty_drafts_degrades_to_single_token() {
        // K=0: no drafts proposed. The target argmaxes have just 1
        // entry (the next-token argmax of target at the last position).
        // Round emits exactly that 1 token.
        let drafts: Vec<u32> = vec![];
        let target_argmaxes: Vec<u32> = vec![42];
        let result = step_round_from_argmaxes(&drafts, &target_argmaxes, &[]);
        assert_eq!(result.committed_tokens, vec![42]);
        assert_eq!(result.accept_count, 0);
        assert!(!result.hit_eos);
    }

    #[test]
    fn full_accept_returns_all_drafts_plus_model_token() {
        // K=3 drafts, all match target → accept all 3, plus 1 free.
        let drafts = vec![10, 20, 30];
        // Target argmaxes for K+1=4 positions: first 3 match drafts,
        // last is the model's continuation.
        let target = vec![10, 20, 30, 99];
        let result = step_round_from_argmaxes(&drafts, &target, &[]);
        assert_eq!(result.committed_tokens, vec![10, 20, 30, 99]);
        assert_eq!(result.accept_count, 3);
        assert!(!result.hit_eos);
    }

    #[test]
    fn partial_accept_truncates_at_first_mismatch() {
        // K=4 drafts; first 2 match, 3rd differs. Accept 2, model
        // continues with whatever target predicted at position 2.
        let drafts = vec![10, 20, 30, 40];
        let target = vec![10, 20, 88, 99]; // mismatch at index 2
        let result = step_round_from_argmaxes(&drafts, &target, &[]);
        assert_eq!(result.committed_tokens, vec![10, 20, 88]);
        assert_eq!(result.accept_count, 2);
        assert!(!result.hit_eos);
    }

    #[test]
    fn eos_in_accepted_prefix_stops_generation() {
        // K=3 drafts; first 2 match target, 2nd is EOS. The 3rd draft
        // (even if it matched) doesn't matter — generation stops.
        let drafts = vec![10, 7, 30];
        let target = vec![10, 7, 30];
        let result = step_round_from_argmaxes(&drafts, &target, &[7]);
        assert_eq!(result.committed_tokens, vec![10, 7]);
        assert_eq!(result.accept_count, 1); // first EOS at index 1
        assert!(result.hit_eos);
    }

    #[test]
    fn eos_as_model_free_token_sets_hit_eos() {
        // K=2 drafts, partial-accept at index 1 (first matches), model's
        // continuation at position 1 IS an EOS.
        let drafts = vec![10, 20];
        let target = vec![10, 1, 30];
        let result = step_round_from_argmaxes(&drafts, &target, &[1]);
        assert_eq!(result.committed_tokens, vec![10, 1]);
        assert_eq!(result.accept_count, 1);
        assert!(result.hit_eos);
    }

    #[test]
    fn eos_check_handles_full_accept_with_eos_continuation() {
        // K=2 drafts both accepted, model's free token is EOS.
        let drafts = vec![10, 20];
        let target = vec![10, 20, 1];
        let result = step_round_from_argmaxes(&drafts, &target, &[1]);
        assert_eq!(result.committed_tokens, vec![10, 20, 1]);
        assert_eq!(result.accept_count, 2);
        assert!(result.hit_eos);
    }

    /// Integration smoke test for orchestrator + drafter: drafter runs
    /// end-to-end producing h_final; then we SIMULATE what the target
    /// verify would produce (synthetic argmaxes) and feed both into
    /// step_round_from_argmaxes. Validates the orchestrator+drafter
    /// API surface from the caller's perspective.
    ///
    /// This is the seam where Phase 4 target integration will plug in:
    /// the synthetic argmaxes here are placeholders for what
    /// `forward_decode_verify_batched` (modified per ADR-030 §3.5
    /// Phase 4) will return.
    #[test]
    #[ignore = "requires Metal device + drafter HF cache"]
    fn smoke_orchestrator_drafter_loop_with_simulated_target() {
        let cfg = DFlashConfig::from_json_str(
            crate::inference::spec_decode::dflash::config::tests::GEMMA4_26B_A4B_DFLASH_CONFIG,
        )
        .expect("config parse");
        let device = MlxDevice::new().expect("Metal device available on M5 Max");
        let mut registry = KernelRegistry::new();

        let home = std::env::var("HOME").expect("HOME set");
        let path = format!(
            "{home}/.cache/huggingface/hub/models--z-lab--gemma-4-26B-A4B-it-DFlash/snapshots/77d4202772dfe50b2396ec7bac9cfffc7b9e7057/model.safetensors"
        );
        let file = DFlashWeightsFile::open(&path).expect("file open");
        let weights = DFlashWeights::load(file.bytes(), &cfg).expect("validated load");
        let tensors = DFlashModelTensors::upload(&device, &cfg, &weights).expect("GPU upload");
        let mut cache = DFlashKvCache::new(&device, &cfg, 128).expect("cache");

        let block_size = 8u32;
        let ctx_chunk = 4u32;
        let hidden = cfg.hidden_size as u32;
        let fc_in = cfg.fc_input_dim() as u32;

        // Build synthetic h + target_hidden_concat (caller would supply
        // these from target's embed + hidden capture).
        let h_elem = (block_size as usize) * (hidden as usize);
        let mut h = device
            .alloc_buffer(h_elem * 4, DType::F32, vec![block_size as usize, hidden as usize])
            .expect("alloc h");
        {
            let s = h.as_mut_slice::<f32>().expect("h slice");
            for v in s.iter_mut() { *v = 1.0; }
        }
        let thc_elem = (ctx_chunk as usize) * (fc_in as usize);
        let mut target_hidden = device
            .alloc_buffer(thc_elem * 4, DType::F32, vec![ctx_chunk as usize, fc_in as usize])
            .expect("alloc target_hidden");
        {
            let s = target_hidden.as_mut_slice::<f32>().expect("target_hidden slice");
            for (i, v) in s.iter_mut().enumerate() {
                *v = 0.1 + ((i % 17) as f32) / 170.0;
            }
        }

        // Run drafter forward → h_final [L, hidden]
        let h_final = dispatch_dflash_model_forward(
            &mut registry, &device, &h, &target_hidden,
            &tensors, &mut cache, &cfg, block_size, ctx_chunk,
        )
        .expect("drafter forward");
        assert_eq!(h_final.element_count(), (block_size as usize) * (hidden as usize));

        // Caller would now apply target's lm_head + softcap on h_final
        // to get K draft tokens. For this test, we SIMULATE:
        // drafts = [1, 2, 3, 4, 5, 6, 7] (K=7 placeholder tokens)
        let drafts: Vec<u32> = (1..=7).collect();
        // Simulate target verify returning argmaxes — 4 match, 1 mismatch, 3 free
        let target_argmaxes = vec![1, 2, 3, 4, 99, 100, 101, 102];

        // Apply round math
        let round = step_round_from_argmaxes(&drafts, &target_argmaxes, &[]);
        assert_eq!(round.accept_count, 4);
        assert_eq!(round.committed_tokens, vec![1, 2, 3, 4, 99]);
        assert!(!round.hit_eos);

        // Cache should have advanced by ctx_chunk on all layers (drafter
        // forward side-effect).
        for (i, l) in cache.layers.iter().enumerate() {
            assert_eq!(
                l.seq_len, ctx_chunk,
                "layer {i} cache should advance by ctx_chunk"
            );
        }
    }

    /// End-to-end GPU coherence gate (ADR-030 iter-64).
    ///
    /// Loads the real target gemma-4-26b-a4b-it Q5_K_M GGUF + the
    /// z-lab DFlash drafter safetensors, runs a **single-token decode
    /// baseline** for N tokens (forward_prefill_batched + N-1
    /// forward_decode calls), fully resets the target's KV state via
    /// [`MlxModelWeights::rollback_kv`], then runs
    /// [`dispatch_dflash_generate`] for the same N tokens.
    ///
    /// **The coherence assertion**: spec-decode output[prompt_len..]
    /// must be byte-identical to the baseline at temp=0. This is the
    /// fundamental greedy-spec-decode correctness guarantee — any
    /// divergence is a bug.
    ///
    /// If this test fails: investigate by per-token diff between
    /// baseline_new and spec_new. The first divergence position
    /// localizes the issue (typically: K/V write position offset,
    /// hidden capture layer mismatch, or accept-prefix math).
    ///
    /// Memory: loads ~20 GB target GGUF + ~820 MB drafter. Skipped by
    /// default (`#[ignore]`). Run with:
    /// ```text
    /// cargo test --release --no-default-features --features metal-shaders \
    ///     --test-threads=1 -- --ignored \
    ///     spec_decode::dflash::orchestrator::tests::e2e_dispatch_dflash_generate_gemma4_26b
    /// ```
    #[test]
    #[ignore = "requires gemma-4-26b GGUF + DFlash drafter HF cache + ~22GB RAM"]
    fn e2e_dispatch_dflash_generate_gemma4_26b() {
        use crate::inference::spec_decode::dflash::{
            kv_cache::DFlashKvCache,
            tensors::DFlashModelTensors,
            weights::{DFlashWeights, DFlashWeightsFile},
        };
        use crate::serve::{
            config::Gemma4Config,
            forward_mlx::MlxModelWeights,
            gpu::GpuContext,
            header::LoadProgress,
        };
        use std::path::PathBuf;

        // ---- Resolve paths ----
        let target_gguf = PathBuf::from(
            "/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/\
             gemma4-ara-2pass-APEX-Q5_K_M.gguf",
        );
        let tokenizer_path = PathBuf::from(
            "/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/tokenizer.json",
        );
        let home = std::env::var("HOME").expect("HOME env set");
        let drafter_dir = format!(
            "{home}/.cache/huggingface/hub/\
             models--z-lab--gemma-4-26B-A4B-it-DFlash/snapshots/\
             77d4202772dfe50b2396ec7bac9cfffc7b9e7057"
        );
        let drafter_cfg_path = format!("{drafter_dir}/config.json");
        let drafter_safetensors_path = format!("{drafter_dir}/model.safetensors");

        for p in [
            &target_gguf,
            &tokenizer_path,
            &PathBuf::from(&drafter_cfg_path),
            &PathBuf::from(&drafter_safetensors_path),
        ] {
            if !p.exists() {
                panic!(
                    "required artifact missing: {} — see test #[ignore] note",
                    p.display()
                );
            }
        }

        // ---- Init GPU + load target ----
        let mut gpu = GpuContext::new().expect("Metal device available");
        let gguf = mlx_native::gguf::GgufFile::open(&target_gguf)
            .expect("open target GGUF");
        let target_cfg = Gemma4Config::from_gguf(&gguf).expect("gemma4 cfg from gguf");
        let mut progress = LoadProgress::new(false, 0, 0);
        let mut target = MlxModelWeights::load_from_gguf(
            &gguf,
            &target_cfg,
            &mut gpu,
            &mut progress,
        )
        .expect("load target weights from GGUF");

        // ---- Load drafter ----
        let drafter_cfg = DFlashConfig::from_json_path(&drafter_cfg_path)
            .expect("drafter config.json");
        let drafter_file = DFlashWeightsFile::open(&drafter_safetensors_path)
            .expect("drafter safetensors open");
        let drafter_weights = DFlashWeights::load(drafter_file.bytes(), &drafter_cfg)
            .expect("drafter validated load");
        let drafter_tensors = {
            let (exec, _reg) = gpu.split();
            DFlashModelTensors::upload(exec.device(), &drafter_cfg, &drafter_weights)
                .expect("drafter GPU upload")
        };

        // Allocate drafter KV cache.  The cache caps total ctx the drafter
        // can absorb across rounds; for max_new_tokens=16 + ~10 prompt
        // tokens we need very little capacity, but allocate generously.
        let drafter_cache_cap: u32 = 4096;
        let mut drafter_cache = {
            let (exec, _reg) = gpu.split();
            DFlashKvCache::new(exec.device(), &drafter_cfg, drafter_cache_cap)
                .expect("drafter cache alloc")
        };

        // ---- Tokenize prompt (plain — see note below) ----
        //
        // We deliberately use a PLAIN (non-chat-templated) prompt for the
        // coherence gate.  Reason: the coherence assertion is
        // `spec_output == baseline_output` at temp=0 — it does NOT
        // require the model to be in-distribution, only that the
        // spec-decode path produces the same argmax sequence as the
        // single-token decode path.  Whether the model is confused by
        // the prompt is orthogonal.
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .expect("load tokenizer.json");
        let prompt_text = std::env::var("HF2Q_TEST_PROMPT")
            .unwrap_or_else(|_| "Q: What is 2+2?\nA:".to_string());
        let encoding = tokenizer.encode(prompt_text.as_str(), false).expect("encode");
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        assert!(!prompt_tokens.is_empty(), "prompt encoding empty");
        eprintln!(
            "[e2e] prompt={prompt_text:?} prompt_tokens.len()={}",
            prompt_tokens.len()
        );

        let prompt_len = prompt_tokens.len();
        let max_new_tokens = 16usize; // N total new tokens to compare
        let block_size = 8u32; // K=7, Phase 1.5 optimal on M5 Max

        // -----------------------------------------------------------------
        // STEP 1 — BASELINE: single-token decode (forward_prefill_batched
        // + N-1 forward_decode calls).  This is the ground-truth greedy
        // argmax sequence the spec-decode path must reproduce.
        // -----------------------------------------------------------------
        eprintln!("[e2e] BASELINE: single-token decode for N={max_new_tokens} tokens");
        let t_baseline = std::time::Instant::now();
        // Size full-attention KV cache to accommodate BOTH the baseline
        // forward_decode loop AND the subsequent SPEC re-prefill (which
        // grows to prompt_len + max_new_tokens + block_size - 1 in the
        // worst-case round).  Use the larger of the two so hybrid_kv
        // (lazy-alloc'd on first prefill, preserved by iter-63 fix) is
        // big enough for both paths.
        let max_decode_for_alloc = max_new_tokens + block_size as usize - 1;
        let first_token_baseline = target
            .forward_prefill_batched(&prompt_tokens, max_decode_for_alloc, 0, &mut gpu)
            .expect("baseline initial prefill");
        let mut baseline_new: Vec<u32> = vec![first_token_baseline];
        let mut last_tok = first_token_baseline;
        // We emit `max_new_tokens` new tokens total: first_token_baseline
        // + (max_new_tokens - 1) forward_decode calls.  Each
        // forward_decode writes KV at the supplied seq_pos and returns
        // the next argmax.
        for step in 0..(max_new_tokens - 1) {
            let seq_pos = prompt_len + step;
            let mut prof: Option<crate::serve::forward_mlx::TokenProfile> = None;
            let next = target
                .forward_decode(last_tok, seq_pos, &mut gpu, &mut prof)
                .expect("baseline forward_decode");
            baseline_new.push(next);
            last_tok = next;
        }
        let baseline_elapsed = t_baseline.elapsed();
        eprintln!(
            "[e2e] BASELINE done: {:.2}s, tokens={baseline_new:?}",
            baseline_elapsed.as_secs_f64(),
        );

        // -----------------------------------------------------------------
        // STEP 2 — RESET target's KV cache fully back to seq_len=0.
        //
        // After baseline: kv_caches[].seq_len = prompt_len + (N-1).  Each
        // forward_decode wrote 1 position; the final argmax (last_tok =
        // baseline_new[N-1]) was the RETURNED prediction and is NOT yet
        // in the cache.  So we need to roll back by `prompt_len + N - 1`
        // to fully reset.  rollback_kv handles sliding and full-attn
        // layers per the verifier::rollback_kv_state tests.
        let rollback_count = prompt_len + max_new_tokens - 1;
        target.rollback_kv(rollback_count);
        eprintln!("[e2e] target.rollback_kv({rollback_count}) → seq_len=0");

        // -----------------------------------------------------------------
        // STEP 3 — SPEC: dispatch_dflash_generate on the same prompt for
        // the same N new tokens.  Uses the SAME target instance (now
        // logically reset).  hybrid_kv storage at positions [0..N) still
        // has baseline data but that's invisible — forward kernels only
        // read positions [0..seq_len) and writes overwrite at start_pos.
        // -----------------------------------------------------------------
        let eos_token_ids: Vec<u32> = vec![1, 106];
        let t_spec = std::time::Instant::now();
        let spec_output = match dispatch_dflash_generate(
            &mut target,
            &drafter_tensors,
            &mut drafter_cache,
            &drafter_cfg,
            &prompt_tokens,
            max_new_tokens,
            block_size,
            &eos_token_ids,
            &mut gpu,
        ) {
            Ok(toks) => toks,
            Err(e) => {
                eprintln!("[e2e] dispatch_dflash_generate FAILED chain:");
                for (i, cause) in e.chain().enumerate() {
                    eprintln!("[e2e]   #{i}: {cause}");
                }
                panic!("dispatch_dflash_generate end-to-end (see chain above)");
            }
        };
        let spec_elapsed = t_spec.elapsed();
        eprintln!(
            "[e2e] SPEC done: {:.2}s, output.len()={} (prompt={prompt_len}, new<={max_new_tokens})",
            spec_elapsed.as_secs_f64(),
            spec_output.len(),
        );

        // Pipeline-runs assertions.
        assert!(
            spec_output.len() > prompt_len,
            "spec output must emit ≥ 1 new token; got len={}",
            spec_output.len()
        );
        let spec_new = &spec_output[prompt_len..];
        assert!(
            spec_new.len() <= max_new_tokens,
            "spec must not exceed max_new_tokens={max_new_tokens}; got {}",
            spec_new.len()
        );

        // Diff print for debugging (before assertion so failures are
        // diagnosable from the test output).
        let n_compare = baseline_new.len().min(spec_new.len());
        eprintln!("[e2e] baseline_new = {baseline_new:?}");
        eprintln!("[e2e] spec_new     = {spec_new:?}");
        for i in 0..n_compare {
            let mark = if baseline_new[i] == spec_new[i] { "✓" } else { "✗" };
            eprintln!(
                "[e2e]   pos {i}: baseline={} spec={} {mark}",
                baseline_new[i], spec_new[i]
            );
        }

        // -----------------------------------------------------------------
        // COHERENCE GATE: spec must produce byte-identical output to
        // baseline at temp=0.  This is the greedy-spec-decode
        // correctness invariant: accept_prefix_argmax only accepts a
        // draft when it matches target's argmax, so committed tokens
        // are always identical to single-token decode.
        // -----------------------------------------------------------------
        assert_eq!(
            spec_new.len(),
            baseline_new.len(),
            "spec emitted {} tokens, baseline emitted {} — length mismatch",
            spec_new.len(),
            baseline_new.len(),
        );
        for (i, (b, s)) in baseline_new.iter().zip(spec_new.iter()).enumerate() {
            assert_eq!(
                b, s,
                "coherence gate FAILED at new-token position {i}: baseline={b} spec={s} \
                 (first {i} tokens matched). Investigate orchestrator at this round.",
            );
        }

        // Decode for visual inspection (only runs if assertions pass).
        let decoded = tokenizer
            .decode(spec_new, /*skip_special=*/ false)
            .unwrap_or_else(|e| format!("<decode failed: {e}>"));
        eprintln!("[e2e] COHERENCE PASS: spec_new == baseline_new for all {n_compare} tokens");
        eprintln!("[e2e] decoded = {decoded:?}");
    }

    /// Iter-67 dual-axis coherence diagnostic on Gemma chat-templated prompt.
    ///
    /// Hardcoded 24-token sequence captured from cmd_generate via
    /// HF2Q_DUMP_PROMPT_TOKENS — Gemma's gguf-embedded chat template applied
    /// to the canonical "Q: What is 2+2?\nA:" prompt.  Contains BOS=2,
    /// `<|turn>`=105, `<turn|>`=106, `\n`=107, `<|channel>`=100,
    /// `<channel|>`=101 — Gemma's chat-template special-token set.
    ///
    /// ## Finding (iter-67)
    ///
    /// On THIS prompt, two orthogonal coherence axes diverge:
    ///
    /// 1. **spec-decode vs single-token forward_decode** — FAILS at pos 3+.
    /// 2. **`forward_prefill_batched` vs `forward_decode`** — ALSO FAILS at
    ///    the same positions (L=27,28,29).  See the DIAG output: running
    ///    forward_prefill_batched on `[prompt + baseline_new[..i]]` and
    ///    taking the last-position argmax DIVERGES from
    ///    `forward_decode(baseline_new[i-1], prompt_len + i - 1)`.
    /// 3. **spec-decode vs forward_prefill_batched** — PASSES (orchestrator
    ///    is internally consistent with batched-prefill).
    ///
    /// Conclusion: the spec-decode orchestrator FAITHFULLY reproduces what
    /// `forward_prefill_batched` computes.  The "incoherence" surfaced by the
    /// HF2Q_SPEC_DFLASH=1 CLI on chat-templated prompts is rooted in
    /// `forward_prefill_batched`'s coherence with `forward_decode`, not in
    /// the orchestrator.  This is consistent with the pre-existing
    /// `coherence_smoke_all_cells` failure on gemma4-apex prompts.
    ///
    /// **Test is intentionally RED**: it documents the chain
    /// (batched_prefill bug → spec-decode inherits) so a future
    /// `forward_prefill_batched` fix is detected as turning this green.
    #[test]
    #[ignore = "iter-67 dual-axis diagnostic; requires gemma-4-26b GGUF + DFlash drafter"]
    fn e2e_coherence_gemma4_chat_templated_prompt() {
        use crate::inference::spec_decode::dflash::{
            kv_cache::DFlashKvCache,
            tensors::DFlashModelTensors,
            weights::{DFlashWeights, DFlashWeightsFile},
        };
        use crate::serve::{
            config::Gemma4Config,
            forward_mlx::MlxModelWeights,
            gpu::GpuContext,
            header::LoadProgress,
        };
        use std::path::PathBuf;

        let target_gguf = PathBuf::from(
            "/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/\
             gemma4-ara-2pass-APEX-Q5_K_M.gguf",
        );
        let home = std::env::var("HOME").expect("HOME env set");
        let drafter_dir = format!(
            "{home}/.cache/huggingface/hub/\
             models--z-lab--gemma-4-26B-A4B-it-DFlash/snapshots/\
             77d4202772dfe50b2396ec7bac9cfffc7b9e7057"
        );
        let drafter_cfg_path = format!("{drafter_dir}/config.json");
        let drafter_safetensors_path = format!("{drafter_dir}/model.safetensors");
        for p in [
            target_gguf.to_string_lossy().to_string(),
            drafter_cfg_path.clone(),
            drafter_safetensors_path.clone(),
        ] {
            if !std::path::Path::new(&p).exists() {
                panic!("required artifact missing: {p}");
            }
        }

        let mut gpu = GpuContext::new().expect("Metal device");
        let gguf = mlx_native::gguf::GgufFile::open(&target_gguf).expect("open gguf");
        let target_cfg = Gemma4Config::from_gguf(&gguf).expect("gemma4 cfg");
        let mut progress = LoadProgress::new(false, 0, 0);
        let mut target = MlxModelWeights::load_from_gguf(&gguf, &target_cfg, &mut gpu, &mut progress)
            .expect("load target");

        let drafter_cfg = DFlashConfig::from_json_path(&drafter_cfg_path).expect("drafter cfg");
        let drafter_file = DFlashWeightsFile::open(&drafter_safetensors_path).expect("drafter file");
        let drafter_weights = DFlashWeights::load(drafter_file.bytes(), &drafter_cfg).expect("drafter weights");
        let drafter_tensors = {
            let (exec, _reg) = gpu.split();
            DFlashModelTensors::upload(exec.device(), &drafter_cfg, &drafter_weights)
                .expect("drafter upload")
        };
        let drafter_cache_cap: u32 = 4096;
        let mut drafter_cache = {
            let (exec, _reg) = gpu.split();
            DFlashKvCache::new(exec.device(), &drafter_cfg, drafter_cache_cap)
                .expect("drafter cache")
        };

        // The exact 24-token sequence cmd_generate produces for the
        // canonical "Q: What is 2+2?\nA:" prompt via render_chat_template.
        let prompt_tokens: Vec<u32> = vec![
            2, 105, 2364, 107, 236935, 236787, 2900, 563, 236743, 236778,
            236862, 236778, 105470, 169631, 236787, 106, 107, 105, 4368,
            107, 100, 45518, 107, 101,
        ];
        let prompt_len = prompt_tokens.len();
        let max_new_tokens = 8usize;
        let block_size = 8u32;
        eprintln!("[e2e-tmpl] prompt_tokens.len()={prompt_len}");

        // BASELINE single-token decode for N tokens.
        let max_decode_for_alloc = max_new_tokens + block_size as usize - 1;
        let first_token_baseline = target
            .forward_prefill_batched(&prompt_tokens, max_decode_for_alloc, 0, &mut gpu)
            .expect("baseline prefill");
        let mut baseline_new: Vec<u32> = vec![first_token_baseline];
        let mut last_tok = first_token_baseline;
        for step in 0..(max_new_tokens - 1) {
            let mut prof: Option<crate::serve::forward_mlx::TokenProfile> = None;
            let next = target
                .forward_decode(last_tok, prompt_len + step, &mut gpu, &mut prof)
                .expect("baseline forward_decode");
            baseline_new.push(next);
            last_tok = next;
        }
        eprintln!("[e2e-tmpl] BASELINE = {baseline_new:?}");

        // Reset target for diagnostic runs.
        target.rollback_kv(prompt_len + max_new_tokens - 1);

        // DIAGNOSTIC: forward_prefill_batched at start_pos=0 on
        // incremental prefixes.  At each prefix length L =
        // prompt_len + i, the LAST-POSITION argmax (returned as
        // first_token) should equal baseline_new[i] — proving
        // forward_prefill_batched's self-attention pipeline is
        // coherent for the chat-templated prompt content at every
        // incremental length.  If this diverges, the bug is in the
        // prefill+self-attention path, not the orchestrator.
        eprintln!("[e2e-tmpl] DIAG: forward_prefill_batched at incremental prefix lengths");
        for i in 0..max_new_tokens {
            let mut prefix: Vec<u32> = prompt_tokens.clone();
            prefix.extend(&baseline_new[..i]);
            let argmax = target
                .forward_prefill_batched(&prefix, max_decode_for_alloc, 0, &mut gpu)
                .expect("diag forward_prefill_batched");
            let mark = if argmax == baseline_new[i] { "✓" } else { "✗" };
            eprintln!(
                "[e2e-tmpl]   DIAG L={} argmax={} baseline_new[{}]={} {}",
                prefix.len(), argmax, i, baseline_new[i], mark,
            );
            target.rollback_kv(prefix.len());
        }

        // SPEC: dispatch_dflash_generate.
        let eos_token_ids: Vec<u32> = vec![]; // mirror --ignore-eos to avoid early-stop
        let spec_output = dispatch_dflash_generate(
            &mut target,
            &drafter_tensors,
            &mut drafter_cache,
            &drafter_cfg,
            &prompt_tokens,
            max_new_tokens,
            block_size,
            &eos_token_ids,
            &mut gpu,
        )
        .expect("dispatch_dflash_generate");
        let spec_new = &spec_output[prompt_len..];
        eprintln!("[e2e-tmpl] SPEC     = {spec_new:?}");

        let n_compare = baseline_new.len().min(spec_new.len());
        for i in 0..n_compare {
            let mark = if baseline_new[i] == spec_new[i] { "✓" } else { "✗" };
            eprintln!(
                "[e2e-tmpl]   pos {i}: baseline={} spec={} {mark}",
                baseline_new[i], spec_new[i]
            );
        }

        // -----------------------------------------------------------------
        // ORCHESTRATOR INTERNAL-CONSISTENCY DIAGNOSTIC
        //
        // Now verify the third coherence axis: spec_new should match what
        // forward_prefill_batched produces when called on the SPEC's own
        // chain — i.e. spec-decode is faithful to batched-prefill.  This
        // SHOULD pass even when the batched_prefill vs forward_decode
        // axis fails.
        // -----------------------------------------------------------------
        eprintln!("[e2e-tmpl] DIAG: forward_prefill_batched on SPEC's own chain");
        let mut spec_self_consistent = true;
        target.rollback_kv(prompt_len + max_new_tokens - 1);
        for i in 0..max_new_tokens {
            let mut prefix: Vec<u32> = prompt_tokens.clone();
            prefix.extend(&spec_new[..i]);
            let argmax = target
                .forward_prefill_batched(&prefix, max_decode_for_alloc, 0, &mut gpu)
                .expect("self-consistency prefill");
            let mark = if argmax == spec_new[i] { "✓" } else { "✗" };
            if argmax != spec_new[i] {
                spec_self_consistent = false;
            }
            eprintln!(
                "[e2e-tmpl]   SELF L={} argmax={} spec_new[{}]={} {}",
                prefix.len(), argmax, i, spec_new[i], mark,
            );
            target.rollback_kv(prefix.len());
        }

        // Spec-decode MUST be internally consistent with batched_prefill —
        // this is the orchestrator's REAL coherence guarantee.  Even if
        // baseline forward_decode disagrees with batched_prefill (axis 2),
        // the orchestrator must faithfully reproduce batched_prefill's
        // chain (axis 3).
        assert!(
            spec_self_consistent,
            "ORCHESTRATOR self-consistency FAILED: spec_new should match \
             forward_prefill_batched(prompt + spec_new[..i]).first_token at \
             every i (see SELF rows above for first mismatch).",
        );
        eprintln!(
            "[e2e-tmpl] ORCHESTRATOR self-consistency PASS (spec-decode faithful to batched_prefill)"
        );

        // The "spec vs forward_decode baseline" axis — currently fails due
        // to an underlying forward_prefill_batched vs forward_decode
        // divergence (see DIAG rows above).  Test is intentionally RED on
        // this axis until the batched_prefill bug is fixed separately.
        assert_eq!(spec_new.len(), baseline_new.len(), "length mismatch");
        for (i, (b, s)) in baseline_new.iter().zip(spec_new.iter()).enumerate() {
            assert_eq!(b, s,
                "coherence gate FAILED on chat-templated prompt at new-token position {i}: \
                 baseline={b} spec={s} — root cause is forward_prefill_batched coherence \
                 (see DIAG output for axis-2 failures).");
        }
        eprintln!("[e2e-tmpl] FULL COHERENCE PASS for chat-templated prompt");
    }
}
