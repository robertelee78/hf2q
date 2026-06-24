//! ADR-040 Phase F M2.2 / S1 — batched lm_head over N gathered decode slots.
//!
//! The single largest matmul in gemma4 decode is the lm_head (Q6_K
//! [2816, 262144] = 605 MB weight read). Under continuous batching, N decode
//! slots each run their own scalar lm_head → N independent 605 MB reads. This
//! module computes the head ONCE for N gathered hidden rows (`m=N`), amortizing
//! that weight read across slots — the first slice (S1) of the batched decode
//! forward.
//!
//! Correctness rests on the proven kernel property H-S1-rowparity
//! (mlx-native `bench_lmhead_batch_row_parity`, commit 272b2a3): the `mv`
//! quantized matmul produces output row r BIT-IDENTICALLY at `m=N` and `m=1`,
//! so the per-row logits this produces equal the scalar head's logits exactly.
//! The end-to-end gate is the existing `slot_aware_*` byte-equivalence suite.
//!
//! Production regime only: Q6_K lm_head (the gemma4 default, ADR-028 iter-188).
//! Non-Q6_K heads return an error rather than silently diverging (no fallback —
//! mantra). The shared per-row finalize (argmax + Q6_K rerank) lands in S1b.

use anyhow::Result;
use mlx_native::DType;

use crate::quantize::imatrix::ImatrixHint;
use crate::serve::forward_mlx_shared::dispatch_qmatmul;
use crate::serve::gpu::GpuContext;

use super::model::MlxModelWeights;

impl MlxModelWeights {
    /// Batched final-norm + lm_head (`m=N`) + softcap over `n` gathered hidden
    /// rows. `hidden_rows` is the row-major `[n, hidden_size]` F32 final hidden
    /// state (one row per slot, captured before final-norm — i.e. the value at
    /// `self.activations.hidden` after the layer loop). Returns the row-major
    /// `[n, vocab_size]` F32 logits, post-softcap, ready for per-row finalize.
    ///
    /// Per-row BIT-IDENTICAL to the scalar head (H-S1-rowparity). Q6_K only.
    pub fn lm_head_batched(
        &self,
        hidden_rows: &[f32],
        n: usize,
        gpu: &mut GpuContext,
    ) -> Result<Vec<f32>> {
        let hs = self.hidden_size;
        let vocab = self.vocab_size;
        if n == 0 {
            return Ok(Vec::new());
        }
        if hidden_rows.len() != n * hs {
            anyhow::bail!(
                "lm_head_batched: hidden_rows len {} != n*hidden_size {}*{}",
                hidden_rows.len(), n, hs
            );
        }
        // S1 production path: Q6_K lm_head only. Other quant heads (Q8/F16) are
        // not yet wired for the batched path; erroring here is intentional (the
        // SlotAware worker routes only Q6_K-lm_head models through this) — no
        // silent scalar fallback that would mask a misroute.
        let q6k = self.lm_head_q6k.as_ref().ok_or_else(|| {
            anyhow::anyhow!("lm_head_batched requires a Q6_K lm_head (production decode path)")
        })?;

        let (exec, reg) = gpu.split();
        let dev = exec.device();
        let metal_dev = dev.metal_device();

        // Batched scratch (allocated per call for S1 bring-up; cache on the
        // model once the path is proven — see S1 follow-up).
        let mut hidden_b = dev
            .alloc_buffer(n * hs * 4, DType::F32, vec![n * hs])
            .map_err(|e| anyhow::anyhow!("lm_head_batched alloc hidden_b: {e}"))?;
        hidden_b
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("lm_head_batched write hidden_b: {e}"))?
            .copy_from_slice(hidden_rows);
        let normed_b = dev
            .alloc_buffer(n * hs * 4, DType::F32, vec![n * hs])
            .map_err(|e| anyhow::anyhow!("lm_head_batched alloc normed_b: {e}"))?;
        let logits_b = dev
            .alloc_buffer(n * vocab * 4, DType::F32, vec![n * vocab])
            .map_err(|e| anyhow::anyhow!("lm_head_batched alloc logits_b: {e}"))?;

        let mut s = exec
            .begin()
            .map_err(|e| anyhow::anyhow!("lm_head_batched session begin: {e}"))?;

        // Batched final RMS norm: [n,hidden] -> [n,hidden]. norm_params ([eps,dim])
        // is per-element and identical for every row, so it is reused as-is.
        s.barrier_between(&[&hidden_b, &self.final_norm], &[&normed_b]);
        s.rms_norm(
            reg, metal_dev,
            &hidden_b,
            &self.final_norm,
            &normed_b,
            &self.activations.norm_params,
            n as u32, hs as u32,
        )
        .map_err(|e| anyhow::anyhow!("lm_head_batched final norm: {e}"))?;

        // Batched lm_head: [n,hidden] x Q6_K[hidden,vocab] -> [n,vocab].
        s.barrier_between(&[&normed_b, &q6k.buffer], &[&logits_b]);
        dispatch_qmatmul(
            &mut s, reg, dev,
            &normed_b,
            q6k,
            &logits_b,
            n as u32,
            ImatrixHint::Global("output.weight"),
        )?;

        // Softcap (elementwise; applied per logit, so the flat [n*vocab] view is
        // correct for all rows) when configured.
        if let Some(cap) = self.final_logit_softcapping {
            s.barrier_between(&[&logits_b], &[&logits_b]);
            mlx_native::ops::softcap::dispatch_softcap(
                s.encoder_mut(), reg, metal_dev,
                &logits_b,
                &logits_b,
                &self.activations.softcap_params,
                cap,
            )
            .map_err(|e| anyhow::anyhow!("lm_head_batched softcap: {e}"))?;
        }

        s.finish()
            .map_err(|e| anyhow::anyhow!("lm_head_batched session finish: {e}"))?;

        let out: &[f32] = logits_b
            .as_slice()
            .map_err(|e| anyhow::anyhow!("lm_head_batched read logits_b: {e}"))?;
        Ok(out.to_vec())
    }
}
