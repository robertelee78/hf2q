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

use super::gpu_full_attn::upload_f32;
use super::io_heads::greedy_argmax_last_token;
use super::kv_cache::HybridKvCache;
use super::model::Qwen35Model;

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
        })
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
            .forward_gpu_with_hidden(prompt, &prefill_positions, &mut self.kv_cache)
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

        let decode_start = Instant::now();
        while generated.len() < max_new {
            // ADR-028 iter-159: whole-iter timer to find loop overhead.
            let mtp_profile_iter = std::env::var("HF2Q_MTP_PROFILE").as_deref() == Ok("1");
            let iter_t0 = if mtp_profile_iter { Some(Instant::now()) } else { None };

            let token_next = greedy_argmax_last_token(&logits_t, vocab);
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
            let kv_cache_ref = &mut self.kv_cache;
            let hidden_ref = &hidden_t;
            let token_embd = &self.verifier.token_embd;
            // ADR-028 iter-154: per-step MTP profile gated on HF2Q_MTP_PROFILE=1.
            let mtp_profile = std::env::var("HF2Q_MTP_PROFILE").as_deref() == Ok("1");
            let mtp_t0 = if mtp_profile { Some(Instant::now()) } else { None };
            let proposed = self.verifier.with_gpu_cache_mut(|device, registry| {
                let embed_next = embed_token_on_device(token_embd, token_next, cfg.hidden_size, device)?;
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
                argmax_logits_gpu(device, registry, &draft_logits, mtp_vocab)
            })?;
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
            let k1_batched =
                std::env::var("HF2Q_SPEC_DECODE_K1").as_deref() == Ok("1");

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
                // --- Step A: forward [token_next] at next_pos ---
                let pos_a = vec![next_pos; 4];
                let (logits_a, hidden_a) = self
                    .verifier
                    .forward_gpu_with_hidden(
                        &[token_next], &pos_a, &mut self.kv_cache,
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
                            &[proposed], &pos_b, &mut self.kv_cache,
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
                let verified_at_n1 = greedy_argmax_slice(logits_row0);
                if std::env::var("HF2Q_SPEC_DECODE_K1_TRACE").as_deref() == Ok("1") {
                    let h_count = verify_hidden.element_count();
                    let next_iter_tn_dbg = greedy_argmax_slice(logits_row1);
                    eprintln!(
                        "[K1_TRACE] iter={} pos={} tn={} prop={} v_at_n1={} (match={}) nitn={} verify_hidden_elems={} (expected 2*h={})",
                        self.stats.proposed,
                        next_pos,
                        token_next,
                        proposed,
                        verified_at_n1,
                        verified_at_n1 == proposed,
                        next_iter_tn_dbg,
                        h_count,
                        2 * hidden_size_u32 as usize,
                    );
                }

                if verified_at_n1 == proposed {
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
                    let next_iter_token_next = greedy_argmax_slice(logits_row1);
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
                        &[token_next], &verify_positions, &mut self.kv_cache,
                    )
                    .with_context(|| format!("SpecDecode verifier step pos {next_pos}"))?;
                let v_ms = verify_t0.map(|t| t.elapsed().as_secs_f64() * 1000.0);

                let post_t0 = if mtp_profile { Some(Instant::now()) } else { None };
                let verified = greedy_argmax_last_token(&verify_logits, vocab);
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

                if proposed == verified && generated.len() < max_new {
                    generated.push(verified);
                    preemitted_argmax = true;
                    self.stats.accepted += 1;
                    if self.is_eos(verified) {
                        break;
                    }
                } else {
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
