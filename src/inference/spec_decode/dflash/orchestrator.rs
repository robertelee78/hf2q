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

    // -------- Step 7: drafter KV rollback (mirror of target's) --------
    let rollback = drafts.len().saturating_sub(round.accept_count) as u32;
    if rollback > 0 {
        for layer in drafter_cache.layers.iter_mut() {
            layer.rollback(rollback);
        }
    }

    Ok(round)
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
}
