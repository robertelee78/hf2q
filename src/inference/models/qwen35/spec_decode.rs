//! Greedy T=0 speculative decoding for Qwen3.5 MTP.
//!
//! The verifier remains the normal Qwen3.5 GPU forward path. The draft model
//! is the appended MTP block loaded in [`super::mtp`].

use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::ops::argmax::dispatch_argmax_f32;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};
use std::time::{Duration, Instant};

// NOTE on device sharing: SpecDecode and the verifier MUST run on the SAME
// `MlxDevice`. The global `MlxBufferPool` keeps residency-enabled devices
// keyed by owner; mixing two `MlxDevice::new()` instances triggers
// "MlxBufferPool cannot mix residency-enabled devices" at the first MTP
// alloc. We therefore reuse the verifier's cached device via
// `Qwen35Model::with_gpu_cache_mut`. The cache is primed by the prefill
// call before any MTP path runs.

/// Slice the last token's hidden-state row out of a `[seq_len, hidden_size]`
/// residual buffer returned by `forward_gpu_with_hidden`.
///
/// `forward_gpu_with_hidden` returns the FULL residual stream
/// (`element_count() == seq_len * hidden_size`) regardless of how many tokens
/// were processed. `MtpWeights::forward_draft` requires exactly `[1, H]`
/// (`element_count() == hidden_size`). For prefill (seq_len = prompt.len())
/// we must slice the final row; for the verifier per-step call (seq_len = 1)
/// this is an identity slice. Mirrors `apply_output_head_gpu_last` which
/// performs the same slice for the lm_head path.
fn last_hidden_row(hidden: &MlxBuffer, hidden_size: u32) -> Result<MlxBuffer> {
    let h = hidden_size as usize;
    let total = hidden.element_count();
    ensure!(
        total % h == 0 && total >= h,
        "last_hidden_row: hidden buffer element_count {} not a positive multiple of hidden_size {}",
        total,
        h
    );
    let seq_len = (total / h) as u64;
    let byte_offset = (seq_len - 1) * (h as u64) * 4; // F32 = 4 bytes
    Ok(hidden.slice_view(byte_offset, h))
}

/// ADR-028 iter-171: slice the Nth row out of a `[seq_len, hidden_size]`
/// residual buffer. `row=0` → first token's hidden state, `row=seq_len-1`
/// → last (== `last_hidden_row`). Used by K=1 batched-verify to extract
/// the token-position-specific hidden state for next iter's MTP draft.
fn nth_hidden_row(hidden: &MlxBuffer, hidden_size: u32, row: u64) -> Result<MlxBuffer> {
    let h = hidden_size as usize;
    let total = hidden.element_count();
    ensure!(
        total % h == 0 && total >= h,
        "nth_hidden_row: hidden buffer element_count {} not a positive multiple of hidden_size {}",
        total, h
    );
    let seq_len = (total / h) as u64;
    ensure!(
        row < seq_len,
        "nth_hidden_row: row {} out of range (seq_len {})",
        row, seq_len
    );
    let byte_offset = row * (h as u64) * 4; // F32 = 4 bytes
    Ok(hidden.slice_view(byte_offset, h))
}

/// Argmax over a vocab-length logits slice. Used by K=1 batched-verify
/// to extract per-position predicted tokens from a multi-row logits
/// buffer (caller slices `logits[row*vocab..(row+1)*vocab]`).
fn greedy_argmax_slice(logits_row: &[f32]) -> u32 {
    debug_assert!(!logits_row.is_empty());
    logits_row
        .iter()
        .enumerate()
        .fold((0u32, f32::NEG_INFINITY), |(best_i, best_v), (i, &v)| {
            if v > best_v { (i as u32, v) } else { (best_i, best_v) }
        })
        .0
}

/// ADR-034 task #91 (2026-05-21) — sample a token from a probability
/// distribution via inverse-CDF (single uniform draw). Used by the
/// stochastic MH paths to (a) sample the drafter's proposed token from
/// the MTP softmax, (b) sample the "bonus" / next-token after accept,
/// and (c) sample replacement on residual-distribution reject.
///
/// Caller must ensure `probs` sums to ~1.0 (we normalize defensively to
/// guard against accumulated rounding when computed via
/// `softmax_with_temp`). Empty input or all-zero probs returns 0.
fn sample_from_probs(probs: &[f32], rng: &mut rand::rngs::StdRng) -> u32 {
    use rand::Rng;
    if probs.is_empty() {
        return 0;
    }
    let total: f32 = probs.iter().sum();
    if total <= 0.0 {
        return 0;
    }
    let u: f32 = rng.gen::<f32>() * total;
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u < acc {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

use super::gpu_full_attn::upload_f32;
use super::io_heads::greedy_argmax_last_token;
use super::kv_cache::HybridKvCache;
use super::model::Qwen35Model;
// ADR-040 Phase B4a (2026-05-23): spec-decode entry points are gated to
// `SlotId(0)` for B4a; multi-slot spec-decode (drafter + verifier
// per-slot KV) lands in Phase B4d alongside the EAGLE3/DFlash drafter
// cache multi-seq lift (ADR-040 §6.1.4).
use crate::serve::multi_seq_kv::SlotId;

/// ADR-034 task #91 (2026-05-21) — Sampler configuration for stochastic
/// (temp > 0) Metropolis-Hastings spec-decode acceptance.
///
/// At `temperature <= 0.0` the runner uses greedy argmax matching at every
/// accept site (byte-identical to the legacy behavior shipped at HEAD
/// `3be36936`). At `temperature > 0.0` the runner uses Leviathan-2023
/// §2.3 stochastic acceptance via [`super::super::super::spec_decode::dflash::rejection_sampler::leviathan_step`]
/// — accept probability `min(1, p/q)` with residual-sampling on reject.
///
/// `seed` is for reproducibility: same seed + same prompt → same generated
/// trajectory (in stochastic mode). The runner uses `StdRng::seed_from_u64`.
#[derive(Debug, Clone, Copy)]
pub struct SpecSampler {
    pub temperature: f32,
    pub seed: u64,
}

impl SpecSampler {
    /// Greedy sampler (no MH; argmax at every accept site). Default.
    pub const fn greedy() -> Self {
        Self { temperature: 0.0, seed: 0 }
    }

    /// Stochastic sampler with the given temperature + seed.
    pub const fn new(temperature: f32, seed: u64) -> Self {
        Self { temperature, seed }
    }

    /// True if MH stochastic acceptance applies (temperature > 0).
    pub fn is_stochastic(&self) -> bool {
        self.temperature > 0.0
    }
}

impl Default for SpecSampler {
    fn default() -> Self {
        Self::greedy()
    }
}

#[derive(Debug, Clone, Default)]
pub struct SpecDecodeStats {
    pub accepted: usize,
    pub rejected: usize,
    pub proposed: usize,
    pub prefill_elapsed: Duration,
    pub decode_elapsed: Duration,
}

impl SpecDecodeStats {
    pub fn acceptance_rate_pct(&self) -> f64 {
        if self.proposed == 0 {
            0.0
        } else {
            (self.accepted as f64) * 100.0 / (self.proposed as f64)
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpecDecodeResult {
    pub tokens: Vec<u32>,
    pub stats: SpecDecodeStats,
}

pub struct SpecDecode<'a> {
    verifier: &'a Qwen35Model,
    kv_cache: HybridKvCache,
    /// Set of token IDs that terminate generation.  Multi-token to
    /// support qwen3.5/3.6 GGUFs whose chat template uses `<|im_end|>`
    /// (not in `tokenizer.ggml.eos_token_id` for some converted GGUFs)
    /// alongside or instead of the canonical `<|endoftext|>`.
    /// ADR-028 iter-266: was `Option<u32>` — caused MTP K1 to ignore
    /// `<|im_end|>` when GGUF metadata only listed `<|endoftext|>`
    /// (or omitted the key entirely).
    eos_token_ids: Vec<u32>,
    stats: SpecDecodeStats,
    /// ADR-034 task #91 (2026-05-21) — see [`SpecSampler`]. Default
    /// is `greedy()` for byte-identical behavior with pre-#91 code.
    sampler: SpecSampler,
}

impl<'a> SpecDecode<'a> {
    pub fn new(
        verifier: &'a Qwen35Model,
        max_seq_len: u32,
        eos_token_id: Option<u32>,
    ) -> Result<Self> {
        let eos_token_ids = eos_token_id.into_iter().collect();
        Self::new_with_eos_set(verifier, max_seq_len, eos_token_ids)
    }

    /// ADR-028 iter-266: multi-EOS variant.  Use this when the caller
    /// has the full set (e.g., qwen3 has both `<|endoftext|>` 151643
    /// and `<|im_end|>` 151645 / 248046).
    pub fn new_with_eos_set(
        verifier: &'a Qwen35Model,
        max_seq_len: u32,
        eos_token_ids: Vec<u32>,
    ) -> Result<Self> {
        ensure!(verifier.mtp.is_some(), "SpecDecode requires MTP weights");
        // Prime the verifier's GPU_CACHE so HybridKvCache and MTP
        // forward_draft allocate on the SAME `MlxDevice`. Two
        // residency-enabled devices in one process trip the global
        // `MlxBufferPool` ("cannot mix residency-enabled devices").
        verifier
            .ensure_gpu_cache_primed()
            .context("SpecDecode::new ensure_gpu_cache_primed")?;
        let kv_cache = verifier.with_gpu_cache_mut(|device, _registry| {
            HybridKvCache::new(&verifier.cfg, device, max_seq_len, 1)
                .context("SpecDecode HybridKvCache::new")
        })?;
        ensure!(
            kv_cache.mtp_slot.is_some(),
            "SpecDecode requires HybridKvCache.mtp_slot"
        );
        Ok(Self {
            verifier,
            kv_cache,
            eos_token_ids,
            stats: SpecDecodeStats::default(),
            sampler: SpecSampler::greedy(),
        })
    }

    /// ADR-034 task #91 (2026-05-21) — builder-style sampler installer.
    /// Call after [`Self::new_with_eos_set`]; default sampler is
    /// [`SpecSampler::greedy()`] (byte-identical to pre-#91 behavior).
    pub fn with_sampler(mut self, sampler: SpecSampler) -> Self {
        self.sampler = sampler;
        self
    }

    /// ADR-034 task #91 (2026-05-21) — multi-EOS constructor with sampler.
    /// At `sampler.temperature > 0` the runner uses Metropolis-Hastings
    /// stochastic acceptance (Leviathan-2023 §2.3) at every accept site
    /// in the K=1 BATCHED path. At `temperature <= 0` runs greedy
    /// argmax-match (default).
    pub fn new_with_sampler_eos_set(
        verifier: &'a Qwen35Model,
        max_seq_len: u32,
        eos_token_ids: Vec<u32>,
        sampler: SpecSampler,
    ) -> Result<Self> {
        Ok(Self::new_with_eos_set(verifier, max_seq_len, eos_token_ids)?
            .with_sampler(sampler))
    }

    /// ADR-034 task #91 (2026-05-21) — sampler-aware entry point.
    /// Mirrors [`Self::run_with_eos_set`] + threads `sampler`.
    pub fn run_with_sampler_eos_set(
        verifier: &'a Qwen35Model,
        prompt: &[u32],
        max_new: usize,
        eos_token_ids: Vec<u32>,
        max_seq_len: u32,
        sampler: SpecSampler,
    ) -> Result<SpecDecodeResult> {
        let mut runner =
            Self::new_with_sampler_eos_set(verifier, max_seq_len, eos_token_ids, sampler)?;
        runner.run_prompt(prompt, max_new)
    }

    pub fn run(verifier: &'a Qwen35Model, prompt: &[u32], max_new: usize) -> Result<Vec<u32>> {
        let max_seq = (prompt.len() + max_new + 64).max(128) as u32;
        let mut runner = Self::new(verifier, max_seq, None)?;
        Ok(runner.run_prompt(prompt, max_new)?.tokens)
    }

    pub fn run_with_eos(
        verifier: &'a Qwen35Model,
        prompt: &[u32],
        max_new: usize,
        eos_token_id: Option<u32>,
        max_seq_len: u32,
    ) -> Result<SpecDecodeResult> {
        let mut runner = Self::new(verifier, max_seq_len, eos_token_id)?;
        runner.run_prompt(prompt, max_new)
    }

    /// ADR-028 iter-266: multi-EOS variant of [`run_with_eos`].
    ///
    /// Pass the full set of stop-token IDs (e.g., both `<|endoftext|>`
    /// and `<|im_end|>` for qwen3 chat templates).  Generation
    /// terminates when the next token matches ANY id in the set.
    /// Fixes MTP K1 path running past `<|im_end|>` when GGUF only
    /// lists `<|endoftext|>` (or neither — see ADR-028 iter-265).
    pub fn run_with_eos_set(
        verifier: &'a Qwen35Model,
        prompt: &[u32],
        max_new: usize,
        eos_token_ids: Vec<u32>,
        max_seq_len: u32,
    ) -> Result<SpecDecodeResult> {
        let mut runner = Self::new_with_eos_set(verifier, max_seq_len, eos_token_ids)?;
        runner.run_prompt(prompt, max_new)
    }

    pub fn run_prompt(&mut self, prompt: &[u32], max_new: usize) -> Result<SpecDecodeResult> {
        ensure!(!prompt.is_empty(), "SpecDecode prompt must not be empty");
        let mtp = self
            .verifier
            .mtp
            .as_ref()
            .ok_or_else(|| anyhow!("SpecDecode requires MTP weights"))?;

        let mut generated = Vec::with_capacity(max_new);
        if max_new == 0 {
            return Ok(SpecDecodeResult {
                tokens: generated,
                stats: self.stats.clone(),
            });
        }

        let prefill_positions = positions_for_range(0, prompt.len());
        let prefill_start = Instant::now();
        let (prefill_logits, prefill_hidden) = self
            .verifier
            .forward_gpu_with_hidden(prompt, &prefill_positions, &mut self.kv_cache, SlotId(0))
            .context("SpecDecode verifier prefill")?;
        self.stats.prefill_elapsed = prefill_start.elapsed();
        // forward_gpu_with_hidden returns the full [seq_len, H] residual; MTP
        // forward_draft expects only the last row. Use slice_view (zero-copy
        // view + offset-aware setBuffer:offset:) — same pattern as
        // apply_output_head_gpu_last.
        let mut hidden_t = last_hidden_row(&prefill_hidden, self.verifier.cfg.hidden_size)
            .context("SpecDecode prefill last_hidden_row slice")?;

        let vocab = self.verifier.cfg.vocab_size;
        let mut logits_t = last_logits(&prefill_logits, vocab)?.to_vec();
        let mut hidden_pos = prompt.len() as i32 - 1;
        let mut preemitted_argmax = false;

        // ADR-034 task #91 (2026-05-21) — stochastic MH state. The runner
        // hoists the RNG to function scope so deterministic-seed runs
        // produce byte-identical output across the entire generation.
        // When `sampler.temperature <= 0` `is_mh` is false and the RNG
        // is never read — greedy paths are byte-identical to pre-#91.
        let is_mh = self.sampler.is_stochastic();
        let sampler_temp = self.sampler.temperature;
        let mut rng = {
            use rand::SeedableRng;
            rand::rngs::StdRng::seed_from_u64(self.sampler.seed)
        };

        let decode_start = Instant::now();
        while generated.len() < max_new {
            // ADR-028 iter-159: whole-iter timer to find loop overhead.
            let mtp_profile_iter = std::env::var("HF2Q_MTP_PROFILE").as_deref() == Ok("1");
            let iter_t0 = if mtp_profile_iter { Some(Instant::now()) } else { None };

            // ADR-034 task #91 (2026-05-21) — MH stochastic mode requires
            // the per-iter `token_next` to MATCH the previously-preemitted
            // token (when one was preemitted), because the next-iter
            // sampling result is non-deterministic from logits_t alone.
            // In greedy mode argmax(logits_t) deterministically reconstructs
            // the preemitted token, so this branch is a no-op for greedy.
            let token_next = if preemitted_argmax {
                *generated.last().expect("preemitted_argmax implies generated non-empty")
            } else if is_mh {
                let probs = crate::inference::spec_decode::dflash
                    ::rejection_sampler::softmax_with_temp(&logits_t, sampler_temp);
                sample_from_probs(&probs, &mut rng)
            } else {
                greedy_argmax_last_token(&logits_t, vocab)
            };
            if !preemitted_argmax {
                generated.push(token_next);
            }
            preemitted_argmax = false;
            if generated.len() >= max_new || self.is_eos(token_next) {
                break;
            }

            let next_pos = hidden_pos + 1;
            // MTP draft step runs on the verifier's cached `MlxDevice` so
            // it shares the global pool's residency-set owner.
            let cfg = self.verifier.cfg.clone();
            let mtp_vocab = mtp.vocab_size;
            let token_embd = &self.verifier.token_embd;
            // ADR-028 iter-154: per-step MTP profile gated on HF2Q_MTP_PROFILE=1.
            let mtp_profile = std::env::var("HF2Q_MTP_PROFILE").as_deref() == Ok("1");

            // ADR-034 SOTA path (2026-05-21): K=N chained MTP draft.
            //
            // HF2Q_SPEC_DECODE_K=N (N >= 2) enables DeepSeek-V3 / MTPLX-style
            // multi-step lookahead: chain MTP forward N times (each step
            // re-uses the prior step's inner-FFN hidden state as `prev_hidden`),
            // then do ONE batched verify of [token_next, draft_0, ..., draft_{N-1}]
            // and sequential accept-prefix walk.
            //
            // Speedup model (vs K=0 single-token verify):
            //   E[tokens/iter] = (1 + p + p^2 + ... + p^N) at acceptance rate p
            //   T_per_iter ≈ T_v(N+1) + N * T_d  (one batched verify + N MTP drafts)
            //
            // At N=2, p=0.75: 2.31 tokens / iter; verify cost ~1.3*T_v(1); MTP cost
            // ~2*T_d. Empirical reference: MTPLX `draft2_fn` at
            // `/opt/MTPLX/mtplx/generation.py:2153`.
            //
            // KV semantics on partial-accept: same pattern as the existing K=1
            // reject path (no explicit cache rollback; next iter's verifier
            // forward overwrites stale positions because each forward writes at
            // the explicit `next_pos` it's given).
            let spec_k: usize = std::env::var("HF2Q_SPEC_DECODE_K")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);

            if spec_k >= 2 {
                // ADR-034 task #91 Step 4 (2026-05-21) — K=N path supports MH
                // stochastic acceptance via leviathan_accept_prefix. When
                // sampler.temperature > 0, chain drafts are sampled
                // stochastically from softmax(draft_logits, temp); the
                // accept walk uses Leviathan-2023 §2.3 per-position
                // residual-sampling. Greedy temp <= 0 path is
                // byte-identical to pre-Step-4 K=N behavior.
                // ADR-034 task #90 Step 4 (2026-05-21) — lazy allocate the
                // per-LA-slot capture buffer ONCE on first K=N iter. Subsequent
                // iters reuse (ensure_la_capture is idempotent at same
                // n_tokens_max). The buffer size covers spec_k+1 positions
                // (1 verified token + spec_k drafts batched). Once allocated
                // forward_gpu_impl detects la_capture_active and routes the
                // LA layer dispatch through dispatch_gated_delta_net_decode_with_capture
                // via build_delta_net_layer (Step 3 wiring). On partial-reject
                // below we copy capture[accepted_idx] → recurrent.
                if self.stats.proposed == 0 {
                    self.verifier.with_gpu_cache_mut(|device, _registry| {
                        self.kv_cache
                            .ensure_la_capture(&self.verifier.cfg, device, (spec_k + 1) as u32)
                            .context("K=N: ensure_la_capture")
                    })?;
                }
                let hidden_size_u32 = self.verifier.cfg.hidden_size;
                let vsz = vocab as usize;
                let mtp_t0 = if mtp_profile { Some(Instant::now()) } else { None };

                // Snapshot pre-iter slot lengths so we can roll back precisely
                // on partial reject. Verifier full-attn slots start at
                // prompt_len after prefill; MTP slot starts at 0 (MTP isn't
                // exercised during prefill). The K=N path advances each by a
                // KNOWN amount (spec_k+1 entries for verifier batched verify,
                // spec_k+1 for MTP chained draft + catch-up).
                let prior_full_attn_len: u32 = self
                    .kv_cache
                    .full_attn
                    .first()
                    .map(|s| s.current_len[0])
                    .unwrap_or(0);
                let prior_mtp_len: u32 = self
                    .kv_cache
                    .mtp_slot
                    .as_ref()
                    .map(|s| s.current_len[0])
                    .unwrap_or(0);

                // Chain `spec_k + 1` MTP forward steps:
                //   - steps 0..spec_k-1 produce drafts[0..spec_k-1] AND write MTP
                //     KV slot at positions [next_pos..next_pos+spec_k-1]
                //   - step spec_k is a CATCH-UP that writes MTP slot at
                //     `next_pos+spec_k` (the position of drafts[spec_k-1]) so
                //     the MTP slot's `current_len` matches the verifier slot's
                //     `current_len` after batched verify (both at
                //     prior_len + spec_k + 1). Without this, MTP slot falls
                //     behind by 1 per iter and attention at next iter's MTP
                //     step 0 sees a positional gap → quality collapses.
                //     The catch-up step's logits are discarded; the bonus
                //     token (if all drafts accepted) is taken from the
                //     verifier's stronger row-spec_k prediction instead.
                // ADR-034 task #91 Step 4 (2026-05-21) — chained MTP draft
                // loop now captures the per-position draft probability
                // distribution when `is_mh`, so the accept walk below can
                // call leviathan_accept_prefix. In greedy mode
                // `draft_probs_per_pos` is left empty (zero-cost).
                let (drafts, draft_probs_per_pos): (Vec<u32>, Vec<Vec<f32>>) = {
                    let hidden_ref = &hidden_t;
                    let kv_cache_ref = &mut self.kv_cache;
                    let rng_ref = &mut rng;
                    self.verifier.with_gpu_cache_mut(|device, registry| {
                        let mut out = Vec::with_capacity(spec_k);
                        let mut out_probs: Vec<Vec<f32>> = if is_mh {
                            Vec::with_capacity(spec_k)
                        } else {
                            Vec::new()
                        };
                        let mut chain_hidden: Option<MlxBuffer> = None;
                        let mut chain_token = token_next;
                        for k in 0..=spec_k {
                            let embed = embed_token_on_device(
                                token_embd,
                                chain_token,
                                cfg.hidden_size,
                                device,
                            )?;
                            let mtp_pos = next_pos + k as i32;
                            let prev_h = chain_hidden.as_ref().unwrap_or(hidden_ref);
                            let (draft_logits, draft_hidden) = mtp
                                .forward_draft_with_hidden(
                                    prev_h,
                                    &embed,
                                    kv_cache_ref,
                                    &[mtp_pos; 4],
                                    device,
                                    registry,
                                    &cfg,
                                )
                                .with_context(|| {
                                    format!(
                                        "SpecDecode K={spec_k} chained MTP step {k} pos {mtp_pos}"
                                    )
                                })?;
                            if k < spec_k {
                                let tok = if is_mh {
                                    // MH: download F32 draft_logits, softmax with
                                    // temp, sample stochastically. Store the
                                    // probability vector so leviathan_accept_prefix
                                    // below has q_v at each draft position.
                                    let logits_cpu = super::gpu_full_attn::download_f32(&draft_logits)?;
                                    let probs = crate::inference::spec_decode::dflash
                                        ::rejection_sampler::softmax_with_temp(
                                            &logits_cpu,
                                            sampler_temp,
                                        );
                                    let t = sample_from_probs(&probs, rng_ref);
                                    out_probs.push(probs);
                                    t
                                } else {
                                    argmax_logits_gpu(
                                        device,
                                        registry,
                                        &draft_logits,
                                        mtp_vocab,
                                    )?
                                };
                                out.push(tok);
                                chain_hidden = Some(draft_hidden);
                                chain_token = tok;
                            }
                            // k == spec_k: catch-up step. Slot write only.
                        }
                        Ok::<(Vec<u32>, Vec<Vec<f32>>), anyhow::Error>((out, out_probs))
                    })?
                };
                let mtp_ms = mtp_t0.map(|t| t.elapsed().as_secs_f64() * 1000.0);
                self.stats.proposed += spec_k;

                // Single batched verify of K+1 tokens.
                let verify_t0 = if mtp_profile { Some(Instant::now()) } else { None };
                let mut verify_input = Vec::with_capacity(spec_k + 1);
                verify_input.push(token_next);
                verify_input.extend(drafts.iter().copied());
                let verify_positions = positions_for_range(next_pos, spec_k + 1);
                let (verify_logits, verify_hidden) = self
                    .verifier
                    .forward_gpu_with_hidden(
                        &verify_input,
                        &verify_positions,
                        &mut self.kv_cache,
                        SlotId(0),
                    )
                    .with_context(|| {
                        format!("SpecDecode K={spec_k} batched verify pos {next_pos}")
                    })?;
                let v_ms = verify_t0.map(|t| t.elapsed().as_secs_f64() * 1000.0);
                ensure!(
                    verify_logits.len() == (spec_k + 1) * vsz,
                    "K={spec_k} batched verify: expected {} logits, got {}",
                    (spec_k + 1) * vsz,
                    verify_logits.len()
                );

                // Accept-prefix walk. Stop at first mismatch; emit accepted
                // drafts + 1 corrected token at the mismatch position.
                //
                // ADR-034 task #91 Step 4 (2026-05-21) — At temp > 0, replace
                // strict argmax-match with Leviathan-2023 §2.3 stochastic
                // MH per-position acceptance via `leviathan_accept_prefix`.
                // At temp <= 0 (greedy), byte-identical to pre-Step-4 walk.
                let mut accepted = 0usize;
                let mut emitted_corrected = false;
                let mut next_iter_logits: Vec<f32> = Vec::new();
                let mut next_iter_hidden_row: u64 = 0;
                let mut hit_eos = false;
                if is_mh {
                    // Build target_probs for all spec_k+1 positions (one per
                    // verify row). leviathan_accept_prefix contract requires
                    // exactly drafts.len() + 1 target probability vectors.
                    let target_probs_per_pos: Vec<Vec<f32>> = (0..=spec_k)
                        .map(|i| {
                            let row = &verify_logits[i * vsz..(i + 1) * vsz];
                            crate::inference::spec_decode::dflash
                                ::rejection_sampler::softmax_with_temp(row, sampler_temp)
                        })
                        .collect();
                    let (accept_count, continuation) =
                        crate::inference::spec_decode::dflash
                            ::rejection_sampler::leviathan_accept_prefix(
                                &drafts,
                                &target_probs_per_pos,
                                &draft_probs_per_pos,
                                &mut rng,
                            );
                    // Push accepted drafts; honor max_new / EOS as in greedy.
                    for i in 0..accept_count {
                        if generated.len() >= max_new {
                            break;
                        }
                        generated.push(drafts[i]);
                        self.stats.accepted += 1;
                        accepted += 1;
                        if self.is_eos(drafts[i]) {
                            hit_eos = true;
                            break;
                        }
                    }
                    if !hit_eos && generated.len() < max_new {
                        if accept_count == spec_k {
                            // Full accept — `continuation` is the stochastic
                            // bonus sampled from target_probs_per_pos[spec_k].
                            // Apply the same row-cap to next_iter_hidden_row
                            // that the greedy path uses (Step 6 Strategy A).
                            //
                            // KNOWN ARCHITECTURAL LIMITATION (not a bug):
                            // hidden_t (next_iter) comes from verify_hidden[0]
                            // (capped) while `continuation` is sampled from
                            // row spec_k's distribution. Greedy has the same
                            // mismatch (row spec_k argmax vs row 0 hidden);
                            // MH amplifies it via temperature variance. The
                            // K=2 attractor collapse documented at
                            // project_adr034_task91_step4_falsified_2026_05_21
                            // is the empirical consequence. Closing this
                            // requires task #89 (cross-length SDPA to unify
                            // batched vs decode kernel) or tree decoding
                            // (multiple candidates avoid compounding-draft).
                            generated.push(continuation);
                            let hidden_row_cap: u64 = match std::env::var(
                                "HF2Q_SPEC_DECODE_KN_HIDDEN_ROW_CAP",
                            )
                            .as_deref()
                            {
                                Ok("off") | Ok("max") => spec_k as u64,
                                Ok(other) => other.parse::<u64>().unwrap_or(0),
                                _ => 0,
                            };
                            next_iter_hidden_row =
                                (spec_k as u64).min(hidden_row_cap);
                            next_iter_logits = verify_logits
                                [spec_k * vsz..(spec_k + 1) * vsz]
                                .to_vec();
                            if self.is_eos(continuation) {
                                hit_eos = true;
                            }
                        } else {
                            // Partial reject at index accept_count.
                            // `continuation` is the residual-sampled
                            // replacement token at that position; push it
                            // (pre-emit) so the next iter's token_next
                            // reuses it.
                            generated.push(continuation);
                            self.stats.rejected += 1;
                            emitted_corrected = true;
                            next_iter_hidden_row = accept_count as u64;
                            next_iter_logits = verify_logits
                                [accept_count * vsz..(accept_count + 1) * vsz]
                                .to_vec();
                            if self.is_eos(continuation) {
                                hit_eos = true;
                            }
                        }
                    }
                } else {
                    for i in 0..spec_k {
                        let row = &verify_logits[i * vsz..(i + 1) * vsz];
                        let target_pred = greedy_argmax_slice(row);
                        if target_pred == drafts[i] {
                            generated.push(drafts[i]);
                            self.stats.accepted += 1;
                            accepted += 1;
                            if self.is_eos(drafts[i]) {
                                hit_eos = true;
                                break;
                            }
                            if generated.len() >= max_new {
                                break;
                            }
                        } else {
                            // Reject: emit corrected token; pre-emit so next iter
                            // doesn't re-draft from this position.
                            generated.push(target_pred);
                            self.stats.rejected += 1;
                            emitted_corrected = true;
                            next_iter_hidden_row = i as u64;
                            next_iter_logits = row.to_vec();
                            if self.is_eos(target_pred) {
                                hit_eos = true;
                            }
                            break;
                        }
                    }
                }
                if !is_mh && accepted == spec_k && !emitted_corrected && !hit_eos {
                    // GREEDY full-accept: emit the argmax bonus token. MH
                    // full-accept already pushed `continuation` above (the
                    // stochastic bonus sampled from target_probs[spec_k]) and
                    // set next_iter_hidden_row / next_iter_logits, so this
                    // block must NOT re-run for MH.
                    //
                    // All drafts accepted: emit the bonus token (verifier's
                    // prediction at row spec_k, predicting position
                    // next_pos+spec_k+1). No truncate — verifier slot wrote
                    // spec_k+1 valid entries [next_pos..next_pos+spec_k] AND
                    // MTP slot also wrote spec_k+1 valid entries via the
                    // catch-up step, so both slots are aligned at
                    // (prior_len + spec_k + 1) with no stale tail.
                    let bonus_row =
                        &verify_logits[spec_k * vsz..(spec_k + 1) * vsz];
                    let bonus = greedy_argmax_slice(bonus_row);
                    if generated.len() < max_new {
                        generated.push(bonus);
                    }
                    // ADR-034 task #90 Step 6 (2026-05-21) — Strategy A row-cap:
                    // hidden_t row used to propagate next iter MUST stay
                    // bounded to avoid the compounding row-N divergence drift
                    // documented at [[project_adr034_step5_k_sweep_finding_2026_05_21]].
                    //
                    // The BONUS TOKEN itself still comes from row spec_k
                    // (verifier's argmax over its own logits at the highest
                    // batch row) — that's the verifier's ground-truth output
                    // and is logit-correct. Only the HIDDEN state used to
                    // seed the next iter's MTP draft is capped to a low row
                    // (default 0 — the safest "first-row" hidden that
                    // matches single-token-decode behavior more closely).
                    //
                    // A/B verified at HEAD (Qwen 3.6 27B, K=2, 200 tok):
                    //   cap=max (legacy):  31.1 t/s @ 79.2% — PERMANENT attractor
                    //   cap=1:             22.9 t/s @ 45.7% — stutter + recovers
                    //   cap=0 (NEW DEFAULT): 23.8 t/s @ 48.5% — FULLY COHERENT
                    //
                    // Env override `HF2Q_SPEC_DECODE_KN_HIDDEN_ROW_CAP`:
                    //   unset / default: cap at 0 (most conservative)
                    //   "off" or "max": legacy row-spec_k (degenerates on K>=2)
                    //   "1": cap at 1 (intermediate)
                    //   "<N>": cap at <N>
                    let hidden_row_cap: u64 = match std::env::var(
                        "HF2Q_SPEC_DECODE_KN_HIDDEN_ROW_CAP",
                    ).as_deref() {
                        Ok("off") | Ok("max") => spec_k as u64,
                        Ok(other) => other.parse::<u64>().unwrap_or(0),
                        _ => 0,
                    };
                    next_iter_hidden_row = (spec_k as u64).min(hidden_row_cap);
                    next_iter_logits = bonus_row.to_vec();
                    if self.is_eos(bonus) {
                        hit_eos = true;
                    }
                } else if emitted_corrected {
                    // Partial reject at draft index `accepted`. Verifier
                    // and MTP slots each wrote spec_k+1 entries this iter
                    // but only the first `accepted + 1` are valid in each
                    // (input chain stayed valid only as long as drafts
                    // were accepted). Roll BOTH back to
                    // `prior_slot_len + accepted + 1` — slot-specific
                    // because verifier and MTP have different base
                    // offsets (verifier starts at prompt_len, MTP at 0).
                    let valid_count = (accepted as u32) + 1;
                    self.kv_cache.truncate_full_attn_to(
                        prior_full_attn_len + valid_count,
                    );
                    self.kv_cache.truncate_mtp_to(
                        prior_mtp_len + valid_count,
                    );
                    // ADR-034 task #90 Step 4 (2026-05-21) — roll back the
                    // DeltaNet recurrent state via the per-position capture
                    // buffer. accepted_idx = `accepted` (the LA state AFTER
                    // the `accepted`-th token of the batch, which is the
                    // last token whose prefix was correct). The next iter's
                    // forward will read `recurrent` (active) and start fresh
                    // from that state.
                    //
                    // Without this, K>=2 on hybrid Qwen 3.5/3.6 degenerates
                    // because the DeltaNet recurrent state has been advanced
                    // by spec_k+1 tokens (during the batched verify) but
                    // only `accepted+1` of those tokens are valid; the next
                    // iter would attend over a stale state ahead by
                    // `spec_k - accepted` steps. See task #86 root-cause
                    // memo for the "the the the..." attractor empirics.
                    // ADR-040 Phase A2b (2026-05-29) — rollback_la_to now
                    // takes an explicit slot. Spec-decode hot path is
                    // single-seq today (`HybridKvCache::n_seqs == 1`), so
                    // we route through SlotId(0). When iter-A2b-cont (or a
                    // future iter) lifts the spec-decode loop to multi-seq,
                    // this site will thread the active slot id from the
                    // batched-verify dispatcher.
                    self.kv_cache
                        .rollback_la_to(
                            crate::serve::multi_seq_kv::SlotId(0),
                            accepted as u32,
                        )
                        .context("K=N partial-reject: rollback_la_to")?;
                }

                // State update for next iter.
                hidden_pos = next_pos + accepted as i32;
                hidden_t = nth_hidden_row(
                    &verify_hidden,
                    hidden_size_u32,
                    next_iter_hidden_row,
                )
                .with_context(|| {
                    format!(
                        "SpecDecode K={spec_k} accept-walk row {} hidden slice",
                        next_iter_hidden_row
                    )
                })?;
                logits_t = next_iter_logits;
                // Both full-accept (bonus emitted) and partial-reject
                // (corrected emitted) paths just pushed a token at the
                // tail. Next iter computes `token_next = argmax(logits_t)`
                // which yields the SAME token; mark preemitted so it
                // doesn't get pushed twice.
                preemitted_argmax = true;

                if mtp_profile {
                    let iter_ms = iter_t0
                        .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0);
                    eprintln!(
                        "[MTP_PROFILE_KN] iter K={} accepted={}/{}: mtp={:.2} ver={:.2} ITER={:.2}",
                        spec_k,
                        accepted,
                        spec_k,
                        mtp_ms.unwrap_or(0.0),
                        v_ms.unwrap_or(0.0),
                        iter_ms,
                    );
                }
                if hit_eos || generated.len() >= max_new {
                    break;
                }
                continue;
            }

            // ---- legacy K=0 / K=1 path (single MTP draft + 1-or-2-token verify) ----
            let mtp_t0 = if mtp_profile { Some(Instant::now()) } else { None };
            let kv_cache_ref = &mut self.kv_cache;
            let hidden_ref = &hidden_t;
            // ADR-034 task #91 (2026-05-21): MH path needs the FULL draft
            // distribution (q_v at the proposed token + residual support
            // for reject), not just argmax. Greedy path is unchanged.
            //
            // Return tuple: (proposed_token, optional_draft_probs).
            //   - Greedy (is_mh=false): (argmax(draft_logits), None)
            //   - MH (is_mh=true):     (sampled_token, Some(softmax(draft_logits, temp)))
            // ADR-034 task #91 — closure returns the draft_logits buffer
            // and (in MH mode) a CPU-downloaded copy of the F32 logits.
            // Stochastic sampling + softmax happen OUTSIDE the GPU
            // closure to avoid borrowing `rng` across the closure boundary.
            let (draft_token_argmax, draft_logits_cpu_opt): (Option<u32>, Option<Vec<f32>>) =
                self.verifier.with_gpu_cache_mut(|device, registry| {
                    let embed_next = embed_token_on_device(
                        token_embd, token_next, cfg.hidden_size, device,
                    )?;
                    let draft_logits = mtp
                        .forward_draft(
                            hidden_ref,
                            &embed_next,
                            kv_cache_ref,
                            &[next_pos; 4],
                            device,
                            registry,
                            &cfg,
                        )
                        .context("SpecDecode MTP forward_draft")?;
                    if is_mh {
                        let logits_cpu = super::gpu_full_attn::download_f32(&draft_logits)?;
                        Ok::<_, anyhow::Error>((None, Some(logits_cpu)))
                    } else {
                        let tok = argmax_logits_gpu(device, registry, &draft_logits, mtp_vocab)?;
                        Ok::<_, anyhow::Error>((Some(tok), None))
                    }
                })?;
            let (proposed, draft_probs_opt): (u32, Option<Vec<f32>>) = if let Some(
                logits_cpu,
            ) = draft_logits_cpu_opt
            {
                let draft_probs = crate::inference::spec_decode::dflash
                    ::rejection_sampler::softmax_with_temp(&logits_cpu, sampler_temp);
                let sampled = sample_from_probs(&draft_probs, &mut rng);
                (sampled, Some(draft_probs))
            } else {
                (draft_token_argmax.expect("argmax token present in greedy mode"), None)
            };
            let mtp_ms = mtp_t0.map(|t| t.elapsed().as_secs_f64() * 1000.0);
            self.stats.proposed += 1;

            // ADR-028 iter-171: K=1 batched verify path.
            //
            // HF2Q_SPEC_DECODE_K1=1 enables Leviathan-style batched verify:
            // 2-token forward [token_next, proposed] at positions
            // [next_pos, next_pos+1]. Per iter-170 bench: T_v(2)=40ms vs
            // T_v(1)=34ms = +18% verifier cost for +78% accepted token
            // throughput → 1.37× greedy speedup at 78% accept.
            //
            // Reject path: position next_pos+1's KV is stale (computed
            // with the wrong draft_1 token). The next iter's verifier
            // call writes pos next_pos+1 with the corrected token,
            // OVERWRITING the stale K/V. No explicit GPU rollback —
            // hidden_pos = next_pos (not next_pos+1) ensures only
            // [0..=next_pos] is read in the meantime.
            //
            // ADR-034 iter 2026-05-21 (task #87): K=1 BATCHED is now the
            // default for DENSE MTP variant (empirically 1.08-1.13× on
            // Qwen 3.6 27B Q8_0 — see project_adr034_k1_batched_shipped
            // memory entry). For MoE MTP variant (Qwen 3.5/3.6 35B-A3B)
            // the 2-token batched verify costs T_v(2)/T_v(1)=2.4× —
            // unprofitable until task #89 fusion lands. Env override
            // HF2Q_SPEC_DECODE_K1=1 forces ON; =0 forces OFF; unset →
            // auto per variant.
            let k1_batched = match std::env::var("HF2Q_SPEC_DECODE_K1").as_deref() {
                Ok("1") => true,
                Ok("0") => false,
                _ => matches!(mtp.ffn_kind(), super::mtp::MtpFfnKind::Dense),
            };

            let verify_t0 = if mtp_profile { Some(Instant::now()) } else { None };
            let hidden_size_u32 = self.verifier.cfg.hidden_size;
            let vsz = vocab as usize;

            // ADR-028 iter-175: TWO_CALLS_PROPER bisect interleaves the
            // accept/reject decision BETWEEN forward A and forward B, so
            // forward B only writes K[N+1] when accept is confirmed (with
            // the correct token = proposed = verified_at_n1). On reject,
            // forward B is skipped — next iter's verifier writes K[N+1]
            // with the corrected token.
            let two_calls = k1_batched
                && std::env::var("HF2Q_SPEC_DECODE_K1_TWO_CALLS")
                    .as_deref() == Ok("1");

            if two_calls {
                // ADR-034 task #91 (2026-05-21) codex review #3 —
                // K1_TWO_CALLS path is greedy-only; MH would require
                // sampling decisions between forward A and B which the
                // current "interleaved accept/reject" doesn't support.
                // Warn once per generation so operators see when their
                // sampler is silently ignored.
                if is_mh && self.stats.accepted == 0 && self.stats.rejected == 0 {
                    eprintln!(
                        "[hf2q WARN] HF2Q_SPEC_DECODE_K1_TWO_CALLS=1 forces the K=1 two-call \
                         interleaved path which is GREEDY-ONLY; --temperature {} sampler is \
                         ignored on this path. Use HF2Q_SPEC_DECODE_K1=1 (without TWO_CALLS) \
                         for MH stochastic acceptance.",
                        self.sampler.temperature,
                    );
                }
                // --- Step A: forward [token_next] at next_pos ---
                let pos_a = vec![next_pos; 4];
                let (logits_a, hidden_a) = self
                    .verifier
                    .forward_gpu_with_hidden(
                        &[token_next], &pos_a, &mut self.kv_cache, SlotId(0),
                    )
                    .with_context(|| {
                        format!("K1 TWO_CALLS_PROPER A pos {next_pos}")
                    })?;
                let last_a = last_logits(&logits_a, vocab)?.to_vec();
                let verified_at_n1 = greedy_argmax_slice(&last_a);

                if std::env::var("HF2Q_SPEC_DECODE_K1_TRACE").as_deref()
                    == Ok("1")
                {
                    eprintln!(
                        "[K1_TRACE_TC] iter={} pos={} tn={} prop={} v_at_n1={} match={}",
                        self.stats.proposed,
                        next_pos,
                        token_next,
                        proposed,
                        verified_at_n1,
                        verified_at_n1 == proposed,
                    );
                }
                let v_ms = verify_t0.map(|t| t.elapsed().as_secs_f64() * 1000.0);

                if verified_at_n1 == proposed {
                    // --- Step B (ACCEPT only): forward [proposed] at N+1 ---
                    let pos_b = vec![next_pos + 1; 4];
                    let (logits_b, hidden_b) = self
                        .verifier
                        .forward_gpu_with_hidden(
                            &[proposed], &pos_b, &mut self.kv_cache, SlotId(0),
                        )
                        .with_context(|| {
                            format!("K1 TWO_CALLS_PROPER B pos {}", next_pos + 1)
                        })?;
                    let last_b = last_logits(&logits_b, vocab)?.to_vec();
                    let next_iter_token_next = greedy_argmax_slice(&last_b);

                    let no_amort = std::env::var("HF2Q_SPEC_DECODE_K1_NO_AMORT")
                        .as_deref() == Ok("1");
                    generated.push(proposed);
                    if !no_amort
                        && generated.len() < max_new
                        && !self.is_eos(proposed)
                    {
                        generated.push(next_iter_token_next);
                    }
                    preemitted_argmax = !no_amort;
                    self.stats.accepted += 1;
                    if self.is_eos(proposed)
                        || (!no_amort && self.is_eos(next_iter_token_next))
                    {
                        break;
                    }
                    if generated.len() >= max_new {
                        break;
                    }
                    hidden_pos = next_pos + 1;
                    hidden_t = last_hidden_row(&hidden_b, hidden_size_u32)
                        .context("K1 TWO_CALLS_PROPER ACCEPT hidden_b last_row")?;
                    logits_t = last_b;
                } else {
                    // REJECT: skip step B. K[N+1] not written this iter;
                    // next iter writes K[N+1] with verified_at_n1.
                    generated.push(verified_at_n1);
                    preemitted_argmax = true;
                    self.stats.rejected += 1;
                    if self.is_eos(verified_at_n1) {
                        break;
                    }
                    hidden_pos = next_pos;
                    hidden_t = last_hidden_row(&hidden_a, hidden_size_u32)
                        .context("K1 TWO_CALLS_PROPER REJECT hidden_a last_row")?;
                    logits_t = last_a;
                }

                if mtp_profile {
                    let iter_ms = iter_t0
                        .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0);
                    eprintln!(
                        "[MTP_PROFILE_K1_TC] iter {}: mtp={:.2} ver={:.2} ITER={:.2}",
                        self.stats.proposed,
                        mtp_ms.unwrap_or(0.0),
                        v_ms.unwrap_or(0.0),
                        iter_ms,
                    );
                }
                // Skip the rest of the K=1 (2-token forward) branch.
                continue;
            }

            if k1_batched {
                // hidden_row_0 / hidden_row_1: pre-extracted hidden rows.
                let mut hidden_row_0: Option<MlxBuffer> = None;
                let mut hidden_row_1: Option<MlxBuffer> = None;
                let (verify_logits, verify_hidden) = {
                    let verify_positions_2 = positions_for_range(next_pos, 2);
                    self
                        .verifier
                        .forward_gpu_with_hidden(
                            &[token_next, proposed],
                            &verify_positions_2,
                            &mut self.kv_cache,
                            SlotId(0),
                        )
                        .with_context(|| {
                            format!("SpecDecode K1 verifier step pos {next_pos}")
                        })?
                };
                let v_ms = verify_t0.map(|t| t.elapsed().as_secs_f64() * 1000.0);

                ensure!(
                    verify_logits.len() == 2 * vsz,
                    "SpecDecode K1: expected 2*vocab={} logits, got {}",
                    2 * vsz, verify_logits.len()
                );
                let logits_row0 = &verify_logits[0..vsz];
                let logits_row1 = &verify_logits[vsz..2 * vsz];

                // ADR-034 task #91 (2026-05-21) — Metropolis-Hastings
                // acceptance branch. At temp > 0, instead of strict
                // argmax-match, use Leviathan-2023 §2.3:
                //   accept_prob = min(1, p_target(proposed) / q_draft(proposed))
                //   on accept:  emit proposed + sampled bonus from softmax(logits_row1, temp)
                //   on reject:  emit replacement sampled from residual max(0, p-q)
                //
                // At temp <= 0 (sampler.is_stochastic() == false), this
                // computes the same greedy verified_at_n1 as before —
                // byte-identical to pre-#91 behavior.
                let (accepted_mh, k1_replacement_or_verified): (bool, u32) = if is_mh {
                    let draft_probs = draft_probs_opt
                        .as_ref()
                        .expect("MH mode must have draft_probs from MTP draft sampling");
                    let target_probs = crate::inference::spec_decode::dflash
                        ::rejection_sampler::softmax_with_temp(logits_row0, sampler_temp);
                    let step = crate::inference::spec_decode::dflash
                        ::rejection_sampler::leviathan_step(
                            proposed,
                            &target_probs,
                            draft_probs,
                            &mut rng,
                        );
                    match step {
                        crate::inference::spec_decode::dflash::rejection_sampler::SampleStep::Accept => {
                            (true, proposed)
                        }
                        crate::inference::spec_decode::dflash::rejection_sampler::SampleStep::Reject {
                            replacement_token,
                        } => (false, replacement_token),
                    }
                } else {
                    let v = greedy_argmax_slice(logits_row0);
                    (v == proposed, v)
                };
                let verified_at_n1 = k1_replacement_or_verified;
                if std::env::var("HF2Q_SPEC_DECODE_K1_TRACE").as_deref() == Ok("1") {
                    let h_count = verify_hidden.element_count();
                    let next_iter_tn_dbg = greedy_argmax_slice(logits_row1);
                    eprintln!(
                        "[K1_TRACE] iter={} pos={} tn={} prop={} v_at_n1={} (match={}) nitn={} verify_hidden_elems={} (expected 2*h={}) mh={}",
                        self.stats.proposed,
                        next_pos,
                        token_next,
                        proposed,
                        verified_at_n1,
                        accepted_mh,
                        next_iter_tn_dbg,
                        h_count,
                        2 * hidden_size_u32 as usize,
                        is_mh,
                    );
                }

                if accepted_mh {
                    // ACCEPT: draft_1 was correct.
                    // Emit BOTH proposed (=token at N+1, draft confirmed)
                    // AND argmax(logits_row1) (=token at N+2, "free" since
                    // verifier processed pos N+1 with the correct token).
                    // This is the Leviathan amortization: per-iter output =
                    // 1 (verifier's own next prediction) + 1 (draft accepted).
                    //
                    // HF2Q_SPEC_DECODE_K1_NO_AMORT=1 disables the "free
                    // token" push for bisect: keeps the 2-token verifier
                    // forward but emits only proposed (so K=1 should
                    // produce the same trajectory as K=0). If output is
                    // STILL wrong with NO_AMORT, the bug is in the 2-token
                    // verifier state propagation. If output is CORRECT
                    // with NO_AMORT, the bug is in the speculative push.
                    let no_amort = std::env::var("HF2Q_SPEC_DECODE_K1_NO_AMORT")
                        .as_deref() == Ok("1");
                    // ADR-034 task #91 — stochastic bonus token in MH mode.
                    let next_iter_token_next = if is_mh {
                        let row1_probs = crate::inference::spec_decode::dflash
                            ::rejection_sampler::softmax_with_temp(logits_row1, sampler_temp);
                        sample_from_probs(&row1_probs, &mut rng)
                    } else {
                        greedy_argmax_slice(logits_row1)
                    };
                    generated.push(proposed);
                    if !no_amort
                        && generated.len() < max_new
                        && !self.is_eos(proposed)
                    {
                        generated.push(next_iter_token_next);
                    }
                    // preemitted=true if we pushed next_iter_token_next.
                    // In no_amort mode we DIDN'T push it, so next iter SHOULD
                    // push token_next at start (= the same N+2 prediction).
                    preemitted_argmax = !no_amort;
                    self.stats.accepted += 1;
                    if self.is_eos(proposed)
                        || (!no_amort && self.is_eos(next_iter_token_next))
                    {
                        break;
                    }
                    if generated.len() >= max_new {
                        break;
                    }
                    hidden_pos = next_pos + 1;
                    hidden_t = if let Some(h1) = hidden_row_1.take() {
                        h1
                    } else {
                        nth_hidden_row(&verify_hidden, hidden_size_u32, 1)
                            .with_context(|| format!("K1 ACCEPT row=1 pos {next_pos}"))?
                    };
                    logits_t = logits_row1.to_vec();
                } else {
                    // REJECT: emit the corrected token at N+1. KV at pos
                    // N+1 is stale (draft_1's contribution); next iter
                    // overwrites it via verifier.forward at pos N+1 with
                    // verified_at_n1. hidden_pos = next_pos (not +1)
                    // ensures attention-read range covers only [0..=N].
                    generated.push(verified_at_n1);
                    preemitted_argmax = true;
                    self.stats.rejected += 1;
                    if self.is_eos(verified_at_n1) {
                        break;
                    }
                    hidden_pos = next_pos;
                    hidden_t = if let Some(h0) = hidden_row_0.take() {
                        h0
                    } else {
                        nth_hidden_row(&verify_hidden, hidden_size_u32, 0)
                            .with_context(|| format!("K1 REJECT row=0 pos {next_pos}"))?
                    };
                    logits_t = logits_row0.to_vec();
                }

                if mtp_profile {
                    let iter_ms = iter_t0
                        .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0);
                    eprintln!(
                        "[MTP_PROFILE_K1] iter {}: mtp={:.2} ver={:.2} ITER={:.2}",
                        self.stats.proposed,
                        mtp_ms.unwrap_or(0.0),
                        v_ms.unwrap_or(0.0),
                        iter_ms,
                    );
                }
            } else {
                // Legacy K=0 path: 1-token verify at next_pos.
                let verify_positions = vec![next_pos; 4];
                let (verify_logits, verify_hidden) = self
                    .verifier
                    .forward_gpu_with_hidden(
                        &[token_next], &verify_positions, &mut self.kv_cache, SlotId(0),
                    )
                    .with_context(|| format!("SpecDecode verifier step pos {next_pos}"))?;
                let v_ms = verify_t0.map(|t| t.elapsed().as_secs_f64() * 1000.0);

                let post_t0 = if mtp_profile { Some(Instant::now()) } else { None };
                // ADR-034 task #91 Step 3 (2026-05-21) — MH stochastic
                // acceptance for K=0 path. At temp > 0:
                //   accept iff u < min(1, p_target(proposed) / q_draft(proposed))
                //   on reject: emit replacement from residual max(0, p - q)
                // Greedy temp=0 path is byte-identical to pre-Step-3.
                let last_verify_logits = last_logits(&verify_logits, vocab)?;
                let (verified, mh_accepted_k0): (u32, Option<bool>) = if is_mh {
                    let draft_probs = draft_probs_opt
                        .as_ref()
                        .expect("MH mode must have draft_probs from MTP draft sampling");
                    let target_probs = crate::inference::spec_decode::dflash
                        ::rejection_sampler::softmax_with_temp(last_verify_logits, sampler_temp);
                    let step = crate::inference::spec_decode::dflash
                        ::rejection_sampler::leviathan_step(
                            proposed,
                            &target_probs,
                            draft_probs,
                            &mut rng,
                        );
                    match step {
                        crate::inference::spec_decode::dflash::rejection_sampler::SampleStep::Accept => {
                            (proposed, Some(true))
                        }
                        crate::inference::spec_decode::dflash::rejection_sampler::SampleStep::Reject {
                            replacement_token,
                        } => (replacement_token, Some(false)),
                    }
                } else {
                    (greedy_argmax_last_token(&verify_logits, vocab), None)
                };
                let argmax_ms = post_t0.map(|t| t.elapsed().as_secs_f64() * 1000.0);

                let slice_t0 = if mtp_profile { Some(Instant::now()) } else { None };
                hidden_t = last_hidden_row(&verify_hidden, hidden_size_u32)
                    .with_context(|| {
                        format!("SpecDecode verify last_hidden_row slice pos {next_pos}")
                    })?;
                let slice_ms = slice_t0.map(|t| t.elapsed().as_secs_f64() * 1000.0);

                let copy_t0 = if mtp_profile { Some(Instant::now()) } else { None };
                logits_t = last_logits(&verify_logits, vocab)?.to_vec();
                let copy_ms = copy_t0.map(|t| t.elapsed().as_secs_f64() * 1000.0);
                hidden_pos = next_pos;

                if mtp_profile {
                    let iter_ms = iter_t0
                        .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0);
                    let summed = mtp_ms.unwrap_or(0.0) + v_ms.unwrap_or(0.0)
                        + argmax_ms.unwrap_or(0.0) + slice_ms.unwrap_or(0.0)
                        + copy_ms.unwrap_or(0.0);
                    eprintln!(
                        "[MTP_PROFILE] iter {}: mtp={:.2} ver={:.2} arg={:.2} sl={:.2} cp={:.2} summed={:.2} ITER={:.2} delta={:.2}",
                        self.stats.proposed,
                        mtp_ms.unwrap_or(0.0), v_ms.unwrap_or(0.0),
                        argmax_ms.unwrap_or(0.0), slice_ms.unwrap_or(0.0), copy_ms.unwrap_or(0.0),
                        summed, iter_ms, iter_ms - summed,
                    );
                }

                // ADR-034 task #91 Step 3 — accept/reject decision now
                // sourced from MH step at temp>0, falls back to greedy
                // proposed==verified at temp=0 (byte-identical).
                let k0_accepted = match mh_accepted_k0 {
                    Some(b) => b,                     // MH path: accept iff leviathan_step returned Accept
                    None => proposed == verified,     // greedy: argmax-match
                };
                if k0_accepted && generated.len() < max_new {
                    generated.push(verified);
                    preemitted_argmax = true;
                    self.stats.accepted += 1;
                    if self.is_eos(verified) {
                        break;
                    }
                } else if !k0_accepted && is_mh && generated.len() < max_new {
                    // MH reject: push the residual replacement so the next
                    // iter's token_next reuses it (preemitted invariant).
                    generated.push(verified);
                    preemitted_argmax = true;
                    self.stats.rejected += 1;
                    if self.is_eos(verified) {
                        break;
                    }
                } else {
                    // Greedy reject: don't push; next iter's token_next will
                    // be argmax(logits_t) = verified deterministically.
                    self.stats.rejected += 1;
                }
            }
        }
        self.stats.decode_elapsed = decode_start.elapsed();

        // ADR-028 iter-170: verifier(N) scaling bench (HF2Q_VERIFIER_NBENCH=1).
        //
        // Empirical T_v(N) for N=1..4. Used to pick K for the iter-162+
        // batched-verify refactor. Runs N forward calls back-to-back with
        // synthetic tokens at sequential positions after the main loop —
        // doesn't affect generated output, just adds bench latency.
        //
        // Speedup formula at K=N-1:
        //   spec speedup = (1 + a × ... × a^(N-1)) × T_v(1) / (T_v(N) + T_d)
        // where a = chained accept rate (~0.78 measured), T_d = MTP draft
        // time (~4ms). Pick K maximizing the ratio.
        if std::env::var("HF2Q_VERIFIER_NBENCH").as_deref() == Ok("1") {
            let bench_start_pos = hidden_pos + 1;
            eprintln!(
                "[VERIFIER_NBENCH] starting bench at pos {bench_start_pos}"
            );
            let mut cumulative_pos = bench_start_pos;
            for n in 1..=4usize {
                let synth_tokens: Vec<u32> =
                    (0..n).map(|i| (i as u32) % 100).collect();
                let synth_positions =
                    positions_for_range(cumulative_pos, n);
                let t0 = Instant::now();
                let _ = self
                    .verifier
                    .forward_gpu_with_hidden(
                        &synth_tokens,
                        &synth_positions,
                        &mut self.kv_cache,
                        SlotId(0),
                    )
                    .with_context(|| {
                        format!("VerifierN bench N={n}")
                    })?;
                let elapsed_ms =
                    t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!(
                    "[VERIFIER_NBENCH] N={} T_v={:.2}ms per-tok={:.2}ms",
                    n,
                    elapsed_ms,
                    elapsed_ms / n as f64
                );
                cumulative_pos += n as i32;
            }
        }

        Ok(SpecDecodeResult {
            tokens: generated,
            stats: self.stats.clone(),
        })
    }

    fn is_eos(&self, token: u32) -> bool {
        self.eos_token_ids.contains(&token)
    }
}

fn embed_token_on_device(
    token_embd: &[f32],
    token: u32,
    hidden_size: u32,
    device: &MlxDevice,
) -> Result<MlxBuffer> {
    let h = hidden_size as usize;
    let token = token as usize;
    let start = token
        .checked_mul(h)
        .ok_or_else(|| anyhow!("SpecDecode token index overflow"))?;
    let end = start + h;
    ensure!(
        end <= token_embd.len(),
        "SpecDecode token {} outside token_embd rows",
        token
    );
    upload_f32(&token_embd[start..end], device).context("SpecDecode upload token embedding")
}

pub fn positions_for_range(start_pos: i32, seq_len: usize) -> Vec<i32> {
    let mut flat = vec![0i32; 4 * seq_len];
    for axis in 0..4 {
        for t in 0..seq_len {
            flat[axis * seq_len + t] = start_pos + t as i32;
        }
    }
    flat
}

fn last_logits(logits: &[f32], vocab_size: u32) -> Result<&[f32]> {
    let v = vocab_size as usize;
    ensure!(logits.len() >= v, "logits shorter than vocab_size");
    Ok(&logits[logits.len() - v..])
}

fn argmax_logits_gpu(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    logits: &MlxBuffer,
    vocab_size: u32,
) -> Result<u32> {
    let out_index = device
        .alloc_buffer(4, DType::U32, vec![1])
        .map_err(|e| anyhow!("SpecDecode alloc argmax index: {e}"))?;
    let out_value = device
        .alloc_buffer(4, DType::F32, vec![1])
        .map_err(|e| anyhow!("SpecDecode alloc argmax value: {e}"))?;
    let mut params = device
        .alloc_buffer(4, DType::U32, vec![1])
        .map_err(|e| anyhow!("SpecDecode alloc argmax params: {e}"))?;
    params
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow!("SpecDecode argmax params slice: {e}"))?[0] = vocab_size;
    let mut enc = device.command_encoder().context("SpecDecode enc argmax")?;
    dispatch_argmax_f32(
        &mut enc,
        registry,
        device.metal_device(),
        logits,
        &out_index,
        &out_value,
        &params,
        vocab_size,
    )
    .context("SpecDecode dispatch argmax")?;
    enc.commit_and_wait().context("SpecDecode commit argmax")?;
    Ok(out_index
        .as_slice::<u32>()
        .map_err(|e| anyhow!("SpecDecode argmax index slice: {e}"))?[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::{Qwen35Config, Qwen35Variant};

    #[test]
    fn positions_are_axis_major() {
        assert_eq!(
            positions_for_range(7, 3),
            vec![7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9]
        );
    }

    #[test]
    fn last_hidden_row_slices_final_token_from_prefill_shape() {
        // Reproduces the production caller's shape contract:
        // forward_gpu_with_hidden returns [seq_len, H] for prefill;
        // forward_draft requires [1, H]. last_hidden_row must produce a
        // slice whose GPU-visible region (byte_offset + element_count)
        // points at the final row. Validation is on element_count +
        // byte_offset (the GPU contract honored via set_buffer:offset:),
        // NOT CPU as_slice (which ignores byte_offset).
        let device = MlxDevice::new().expect("MlxDevice for slice test");
        let h: u32 = 8;
        let seq_len: usize = 5;
        let n = seq_len * h as usize;
        let buf = device
            .alloc_buffer(n * 4, DType::F32, vec![seq_len, h as usize])
            .expect("alloc residual buffer");
        let last = last_hidden_row(&buf, h).expect("last_hidden_row");
        assert_eq!(
            last.element_count(),
            h as usize,
            "shape must be [H] so MTP forward_draft's element_count check passes"
        );
        // Final row offset = (seq_len - 1) * H * sizeof(F32) = 4 * 8 * 4 = 128.
        assert_eq!(
            last.byte_offset(),
            ((seq_len - 1) * h as usize * 4) as u64,
            "GPU set_buffer:offset: must point at final row"
        );
        // Storage is shared (zero-copy view).
        assert_eq!(
            last.metal_buffer().length(),
            buf.metal_buffer().length(),
            "slice_view shares storage with parent"
        );
    }

    #[test]
    fn last_hidden_row_handles_seq_len_one_identity() {
        // Verifier per-step path returns seq_len=1 already; slice must be
        // [H]-shaped with byte_offset 0.
        let device = MlxDevice::new().expect("MlxDevice for identity test");
        let h: u32 = 4;
        let buf = device
            .alloc_buffer((h as usize) * 4, DType::F32, vec![1, h as usize])
            .expect("alloc one-token residual");
        let last = last_hidden_row(&buf, h).expect("last_hidden_row identity");
        assert_eq!(last.element_count(), h as usize);
        assert_eq!(last.byte_offset(), 0, "seq_len=1 → no offset");
    }

    #[test]
    fn last_hidden_row_rejects_misaligned_buffer() {
        let device = MlxDevice::new().expect("MlxDevice for reject test");
        // 7 elements with hidden_size=4 — not a multiple.
        let buf = device
            .alloc_buffer(7 * 4, DType::F32, vec![7])
            .expect("alloc misaligned");
        let err = last_hidden_row(&buf, 4).expect_err("misaligned must error");
        assert!(err.to_string().contains("not a positive multiple"));
    }

    #[test]
    fn run_rejects_missing_mtp_before_gpu_alloc() {
        let cfg = Qwen35Config {
            variant: Qwen35Variant::Dense,
            hidden_size: 32,
            num_hidden_layers: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 32,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 32,
            linear_value_head_dim: 32,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 1,
            layer_types: vec![],
            partial_rotary_factor: 1.0,
            rope_theta: 1_000_000.0,
            rotary_dim: 32,
            mrope_section: [8, 8, 8, 8],
            mrope_interleaved: true,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 128,
            vocab_size: 64,
            attn_output_gate: true,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: true,
            intermediate_size: Some(32),
            moe: None,
        };
        let model = Qwen35Model::empty_from_cfg(cfg);
        let err = SpecDecode::run(&model, &[1], 1).expect_err("missing MTP must fail");
        assert!(err.to_string().contains("requires MTP"));
    }
}
