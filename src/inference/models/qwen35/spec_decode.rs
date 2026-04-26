//! Greedy T=0 speculative decoding for Qwen3.5 MTP.
//!
//! The verifier remains the normal Qwen3.5 GPU forward path. The draft model
//! is the appended MTP block loaded in [`super::mtp`].

use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::ops::argmax::dispatch_argmax_f32;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};
use std::time::{Duration, Instant};

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
    device: MlxDevice,
    registry: KernelRegistry,
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
        let device = MlxDevice::new().map_err(|e| anyhow!("SpecDecode MlxDevice::new: {e}"))?;
        let registry = KernelRegistry::new();
        let kv_cache = HybridKvCache::new(&verifier.cfg, &device, max_seq_len, 1)
            .context("SpecDecode HybridKvCache::new")?;
        ensure!(
            kv_cache.mtp_slot.is_some(),
            "SpecDecode requires HybridKvCache.mtp_slot"
        );
        Ok(Self {
            verifier,
            device,
            registry,
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
        let (prefill_logits, mut hidden_t) = self
            .verifier
            .forward_gpu_with_hidden(prompt, &prefill_positions, &mut self.kv_cache)
            .context("SpecDecode verifier prefill")?;
        self.stats.prefill_elapsed = prefill_start.elapsed();

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
            let embed_next = self.embed_token(token_next)?;
            let draft_logits = mtp
                .forward_draft(
                    &hidden_t,
                    &embed_next,
                    &mut self.kv_cache,
                    &[next_pos; 4],
                    &self.device,
                    &mut self.registry,
                    &self.verifier.cfg,
                )
                .context("SpecDecode MTP forward_draft")?;
            let proposed = argmax_logits_gpu(
                &self.device,
                &mut self.registry,
                &draft_logits,
                mtp.vocab_size,
            )?;
            self.stats.proposed += 1;

            let verify_positions = vec![next_pos; 4];
            let (verify_logits, hidden_next) = self
                .verifier
                .forward_gpu_with_hidden(&[token_next], &verify_positions, &mut self.kv_cache)
                .with_context(|| format!("SpecDecode verifier step pos {next_pos}"))?;
            let verified = greedy_argmax_last_token(&verify_logits, vocab);

            hidden_t = hidden_next;
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

    fn embed_token(&self, token: u32) -> Result<MlxBuffer> {
        let h = self.verifier.cfg.hidden_size as usize;
        let token = token as usize;
        let start = token
            .checked_mul(h)
            .ok_or_else(|| anyhow!("SpecDecode token index overflow"))?;
        let end = start + h;
        ensure!(
            end <= self.verifier.token_embd.len(),
            "SpecDecode token {} outside token_embd rows",
            token
        );
        upload_f32(&self.verifier.token_embd[start..end], &self.device)
            .context("SpecDecode upload token embedding")
    }

    fn is_eos(&self, token: u32) -> bool {
        self.eos_token_id == Some(token)
    }
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
            intermediate_size: Some(32),
            moe: None,
        };
        let model = Qwen35Model::empty_from_cfg(cfg);
        let err = SpecDecode::run(&model, &[1], 1).expect_err("missing MTP must fail");
        assert!(err.to_string().contains("requires MTP"));
    }
}
