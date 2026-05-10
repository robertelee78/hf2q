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
    eos_token_id: Option<u32>,
    stats: SpecDecodeStats,
}

impl<'a> SpecDecode<'a> {
    pub fn new(
        verifier: &'a Qwen35Model,
        max_seq_len: u32,
        eos_token_id: Option<u32>,
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
            eos_token_id,
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

            let verify_positions = vec![next_pos; 4];
            let verify_t0 = if mtp_profile { Some(Instant::now()) } else { None };
            let (verify_logits, verify_hidden) = self
                .verifier
                .forward_gpu_with_hidden(&[token_next], &verify_positions, &mut self.kv_cache)
                .with_context(|| format!("SpecDecode verifier step pos {next_pos}"))?;
            if mtp_profile {
                let v_ms = verify_t0.unwrap().elapsed().as_secs_f64() * 1000.0;
                eprintln!("[MTP_PROFILE] iter {}: mtp_draft={:.2}ms verifier={:.2}ms",
                    self.stats.proposed, mtp_ms.unwrap_or(0.0), v_ms);
            }
            let verified = greedy_argmax_last_token(&verify_logits, vocab);

            // Verifier step is a single-token decode (seq_len=1); slice is
            // an identity-shaped view but makes the [1, H] contract explicit.
            hidden_t = last_hidden_row(&verify_hidden, self.verifier.cfg.hidden_size)
                .with_context(|| format!("SpecDecode verify last_hidden_row slice pos {next_pos}"))?;
            logits_t = last_logits(&verify_logits, vocab)?.to_vec();
            hidden_pos = next_pos;

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
        self.stats.decode_elapsed = decode_start.elapsed();

        Ok(SpecDecodeResult {
            tokens: generated,
            stats: self.stats.clone(),
        })
    }

    fn is_eos(&self, token: u32) -> bool {
        self.eos_token_id == Some(token)
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
