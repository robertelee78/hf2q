//! Logit heads, argmax, and NLL scoring for the Gemma 4 forward pass.
//!
//! Moved from `src/serve/forward_mlx.rs` by ADR-038 Step 3.

use anyhow::Result;

use super::model::MlxModelWeights;

impl MlxModelWeights {
    /// ADR-030 Phase 4 — compute per-position argmaxes from a
    /// post-last-layer hidden state buffer.
    ///
    /// Convenience wrapper around `per_position_argmax_from_hidden_opt`
    /// with `apply_final_norm = true` (matches the target's tail).
    ///
    /// Takes `hidden` of shape `[seq_len, hidden_size]` F32 (the
    /// `pf_hidden` content after all decoder layers ran, captured via
    /// the DFlash capture hook with the FINAL layer index included
    /// in `target_layer_ids`). For each row, runs final_norm +
    /// lm_head + softcap + argmax. Returns `[seq_len]` u32.
    ///
    /// Uses the same model state as the existing last-row tail in
    /// `forward_prefill_batched`. Bit-exact same dispatch ordering —
    /// guarantees the LAST-position argmax matches first_token.
    ///
    /// **Not on the production hot path.** Spec-decode verify only.
    pub fn per_position_argmax_from_hidden(
        &mut self,
        hidden: &[f32],
        seq_len: u32,
        gpu: &mut crate::serve::gpu::GpuContext,
    ) -> anyhow::Result<Vec<u32>> {
        self.per_position_argmax_from_hidden_opt(hidden, seq_len, true, gpu)
    }

    /// ADR-030 Phase 4 — per-position argmax with optional final_norm.
    ///
    /// `apply_final_norm = true` mirrors target's tail (used for
    /// `forward_decode_verify_batched`).
    ///
    /// `apply_final_norm = false` is the drafter-side path: the drafter
    /// applies its own `norm` (drafter's final_norm) inside
    /// `dispatch_dflash_model_forward`; the orchestrator then takes
    /// the drafter's h_final and runs target's lm_head + softcap +
    /// argmax via THIS method with apply_final_norm=false. Mirrors
    /// Python `model_mlx.py:194` — `logits = self.lm_head(self.norm(h))`
    /// where `self.norm` is the drafter's, `self.lm_head` is target's
    /// (shared via `bind()`).
    pub fn per_position_argmax_from_hidden_opt(
        &mut self,
        hidden: &[f32],
        seq_len: u32,
        apply_final_norm: bool,
        gpu: &mut crate::serve::gpu::GpuContext,
    ) -> anyhow::Result<Vec<u32>> {
        let hs = self.hidden_size;
        let vocab_size = self.vocab_size;
        let expected = (seq_len as usize) * hs;
        if hidden.len() != expected {
            anyhow::bail!(
                "per_position_argmax_from_hidden: hidden len {} != seq_len({}) * hs({}) = {}",
                hidden.len(), seq_len, hs, expected
            );
        }
        // ADR-030 iter-70: HF2Q_DFLASH_BATCH_ARGMAX=1 (opt-in) routes to
        // the batched implementation that:
        // 1. CPU-uploads ALL hidden rows in one shot (no per-iter CPU writes)
        // 2. Runs all seq_len iterations in ONE command buffer (one finish())
        // 3. Reads all argmaxes from a seq_len-sized output buffer at end
        // Saves ~K * sync_overhead per call.  Profile data at N=16 shows
        // target_argmax = 51 ms/round → expected ~10 ms/round after batching.
        if std::env::var("HF2Q_DFLASH_BATCH_ARGMAX").as_deref() == Ok("1") {
            return self.per_position_argmax_from_hidden_batched_impl(
                hidden, seq_len, apply_final_norm, gpu,
            );
        }
        let mut argmaxes = Vec::with_capacity(seq_len as usize);

        let (exec, reg) = gpu.split();
        let dev = exec.device();
        let metal_dev = dev.metal_device();

        for pos in 0..(seq_len as usize) {
            // Copy hidden[pos] into activations.hidden via CPU→GPU upload.
            {
                let slice: &mut [f32] = self
                    .activations
                    .hidden
                    .as_mut_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("activations.hidden slice: {e}"))?;
                slice[..hs].copy_from_slice(&hidden[pos * hs..(pos + 1) * hs]);
            }

            // Open session and run final_norm + lm_head + softcap + argmax.
            let mut s = exec
                .begin()
                .map_err(|e| anyhow::anyhow!("per_pos session begin: {e}"))?;

            // norm_out source: either final_norm(hidden) when caller
            // requests target's final_norm, OR a direct copy of hidden
            // when caller has already applied the drafter's final_norm
            // externally.
            if apply_final_norm {
                s.barrier_between(
                    &[&self.activations.hidden, &self.final_norm],
                    &[&self.activations.norm_out],
                );
                s.rms_norm(
                    reg,
                    metal_dev,
                    &self.activations.hidden,
                    &self.final_norm,
                    &self.activations.norm_out,
                    &self.activations.norm_params,
                    1,
                    hs as u32,
                )
                .map_err(|e| anyhow::anyhow!("per_pos final_norm: {e}"))?;
            } else {
                // Copy hidden → norm_out (already pre-normed by drafter)
                s.barrier_between(
                    &[&self.activations.hidden],
                    &[&self.activations.norm_out],
                );
                mlx_native::ops::copy::dispatch_copy_f32(
                    s.encoder_mut(),
                    reg,
                    metal_dev,
                    &self.activations.hidden,
                    &self.activations.norm_out,
                    0,
                    0,
                    hs,
                )
                .map_err(|e| anyhow::anyhow!("per_pos pre-normed copy: {e}"))?;
            }

            if let Some(ref q6k) = self.lm_head_q6k {
                s.barrier_between(
                    &[&self.activations.norm_out, &q6k.buffer],
                    &[&self.activations.logits],
                );
                crate::serve::forward_mlx_shared::dispatch_qmatmul(
                    &mut s,
                    reg,
                    dev,
                    &self.activations.norm_out,
                    q6k,
                    &mut self.activations.logits,
                    1,
                    crate::quantize::imatrix::ImatrixHint::Global("output.weight"),
                )
                .map_err(|e| anyhow::anyhow!("per_pos lm_head Q6_K: {e}"))?;
            } else if let Some(ref q8) = self.lm_head_q8 {
                s.barrier_between(
                    &[&self.activations.norm_out, &q8.buffer],
                    &[&self.activations.logits],
                );
                crate::serve::forward_mlx_shared::dispatch_qmatmul(
                    &mut s,
                    reg,
                    dev,
                    &self.activations.norm_out,
                    q8,
                    &mut self.activations.logits,
                    1,
                    crate::quantize::imatrix::ImatrixHint::Global("output.weight"),
                )
                .map_err(|e| anyhow::anyhow!("per_pos lm_head Q8: {e}"))?;
            } else if let Some(ref lm_head_f16) = self.lm_head_f16 {
                s.barrier_between(
                    &[&self.activations.norm_out, lm_head_f16],
                    &[&self.activations.logits],
                );
                mlx_native::ops::dense_gemm::dispatch_dense_matvec_f16w_f32io(
                    s.encoder_mut(),
                    reg,
                    metal_dev,
                    &self.activations.norm_out,
                    lm_head_f16,
                    &self.activations.logits,
                    &mlx_native::ops::dense_gemm::DenseGemmF16Params {
                        m: 1,
                        n: vocab_size as u32,
                        k: hs as u32,
                    },
                )
                .map_err(|e| anyhow::anyhow!("per_pos lm_head f16: {e}"))?;
            } else {
                anyhow::bail!("per_position_argmax_from_hidden requires lm_head_q6k / q8 / f16");
            }

            if let Some(cap) = self.final_logit_softcapping {
                s.barrier_between(
                    &[&self.activations.logits],
                    &[&self.activations.logits],
                );
                mlx_native::ops::softcap::dispatch_softcap(
                    s.encoder_mut(),
                    reg,
                    metal_dev,
                    &self.activations.logits,
                    &self.activations.logits,
                    &self.activations.softcap_params,
                    cap,
                )
                .map_err(|e| anyhow::anyhow!("per_pos softcap: {e}"))?;
            }

            s.barrier_between(
                &[&self.activations.logits],
                &[&self.activations.argmax_index, &self.activations.argmax_value],
            );
            mlx_native::ops::argmax::dispatch_argmax_f32(
                s.encoder_mut(),
                reg,
                metal_dev,
                &self.activations.logits,
                &self.activations.argmax_index,
                &self.activations.argmax_value,
                &self.activations.argmax_params,
                vocab_size as u32,
            )
            .map_err(|e| anyhow::anyhow!("per_pos argmax: {e}"))?;

            s.finish()
                .map_err(|e| anyhow::anyhow!("per_pos session finish: {e}"))?;

            let argmax_val: u32 = {
                let idx: &[u32] = self
                    .activations
                    .argmax_index
                    .as_slice()
                    .map_err(|e| anyhow::anyhow!("per_pos argmax read: {e}"))?;
                idx[0]
            };
            argmaxes.push(argmax_val);
        }

        Ok(argmaxes)
    }

    /// ADR-030 iter-70 — batched per-position argmax.
    ///
    /// Equivalent semantically to [`per_position_argmax_from_hidden_opt`]
    /// but:
    /// 1. Uploads ALL hidden rows once (one bulk CPU→GPU copy).
    /// 2. Allocates seq_len-element argmax_index/value output buffers.
    /// 3. Runs all `seq_len` chains (copy → norm → lm_head → softcap →
    ///    argmax) inside ONE command buffer.  Shared scratch
    ///    (activations.hidden / norm_out / logits) is reused per
    ///    iteration with `barrier_between` ensuring iter i's reads of
    ///    a shared buffer complete before iter i+1's writes.
    /// 4. Single `finish()` at end → reads all argmaxes from the
    ///    per-position output buffer view.
    ///
    /// Eliminates `seq_len - 1` `commit_and_wait` syncs.  Profile data
    /// shows ~5-7 ms per sync, so for seq_len=8 we expect ~35-50 ms
    /// savings per call (validated by iter-71 bench).
    pub(crate) fn per_position_argmax_from_hidden_batched_impl(
        &mut self,
        hidden: &[f32],
        seq_len: u32,
        apply_final_norm: bool,
        gpu: &mut crate::serve::gpu::GpuContext,
    ) -> anyhow::Result<Vec<u32>> {
        let hs = self.hidden_size;
        let vocab_size = self.vocab_size;
        let n = seq_len as usize;
        let expected = n * hs;
        if hidden.len() != expected {
            anyhow::bail!(
                "per_position_argmax_batched: hidden len {} != seq_len({}) * hs({}) = {}",
                hidden.len(), seq_len, hs, expected
            );
        }
        let (exec, reg) = gpu.split();
        let dev = exec.device();
        let metal_dev = dev.metal_device();

        // (1) Bulk upload hidden → GPU.
        let mut gpu_hidden_all = dev
            .alloc_buffer(n * hs * 4, mlx_native::DType::F32, vec![n, hs])
            .map_err(|e| anyhow::anyhow!("alloc gpu_hidden_all: {e}"))?;
        gpu_hidden_all
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("gpu_hidden_all slice: {e}"))?
            .copy_from_slice(hidden);

        // (2) Per-position argmax output buffers.
        let argmax_index_all = dev
            .alloc_buffer(n * 4, mlx_native::DType::U32, vec![n])
            .map_err(|e| anyhow::anyhow!("alloc argmax_index_all: {e}"))?;
        let argmax_value_all = dev
            .alloc_buffer(n * 4, mlx_native::DType::F32, vec![n])
            .map_err(|e| anyhow::anyhow!("alloc argmax_value_all: {e}"))?;

        // iter-72: truly batched processing.  One rms_norm with rows=n,
        // one lm_head matmul with m=n, one softcap over n*vocab, then
        // n argmax dispatches with logits views per row.  Replaces the
        // iter-70 sequential loop on shared scratch.
        let norm_out_batched = dev
            .alloc_buffer(n * hs * 4, mlx_native::DType::F32, vec![n, hs])
            .map_err(|e| anyhow::anyhow!("alloc norm_out_batched: {e}"))?;
        let mut logits_batched = dev
            .alloc_buffer(
                n * (vocab_size as usize) * 4,
                mlx_native::DType::F32,
                vec![n, vocab_size as usize],
            )
            .map_err(|e| anyhow::anyhow!("alloc logits_batched: {e}"))?;

        let mut s = exec
            .begin()
            .map_err(|e| anyhow::anyhow!("batched argmax session begin: {e}"))?;

        // (a) ONE rms_norm or copy with rows=n
        if apply_final_norm {
            s.barrier_between(
                &[&gpu_hidden_all, &self.final_norm],
                &[&norm_out_batched],
            );
            s.rms_norm(
                reg,
                metal_dev,
                &gpu_hidden_all,
                &self.final_norm,
                &norm_out_batched,
                &self.activations.norm_params,
                n as u32,
                hs as u32,
            )
            .map_err(|e| anyhow::anyhow!("batched arg rms_norm: {e}"))?;
        } else {
            s.barrier_between(
                &[&gpu_hidden_all],
                &[&norm_out_batched],
            );
            mlx_native::ops::copy::dispatch_copy_f32(
                s.encoder_mut(),
                reg,
                metal_dev,
                &gpu_hidden_all,
                &norm_out_batched,
                0,
                0,
                n * hs,
            )
            .map_err(|e| anyhow::anyhow!("batched arg pre-norm copy: {e}"))?;
        }

        // (b) ONE lm_head dispatch with m=n.  dispatch_qmatmul routes
        //     m<=8 through mat-vec (multiple matvecs per dispatch) and
        //     m>8 through mat-mat (simdgroup MMA, tile kernel).
        if let Some(ref q6k) = self.lm_head_q6k {
            s.barrier_between(
                &[&norm_out_batched, &q6k.buffer],
                &[&logits_batched],
            );
            crate::serve::forward_mlx_shared::dispatch_qmatmul(
                &mut s,
                reg,
                dev,
                &norm_out_batched,
                q6k,
                &mut logits_batched,
                n as u32,
                crate::quantize::imatrix::ImatrixHint::Global("output.weight"),
            )
            .map_err(|e| anyhow::anyhow!("batched arg lm_head Q6_K: {e}"))?;
        } else if let Some(ref q8) = self.lm_head_q8 {
            s.barrier_between(
                &[&norm_out_batched, &q8.buffer],
                &[&logits_batched],
            );
            crate::serve::forward_mlx_shared::dispatch_qmatmul(
                &mut s,
                reg,
                dev,
                &norm_out_batched,
                q8,
                &mut logits_batched,
                n as u32,
                crate::quantize::imatrix::ImatrixHint::Global("output.weight"),
            )
            .map_err(|e| anyhow::anyhow!("batched arg lm_head Q8: {e}"))?;
        } else if let Some(ref lm_head_f16) = self.lm_head_f16 {
            s.barrier_between(
                &[&norm_out_batched, lm_head_f16],
                &[&logits_batched],
            );
            mlx_native::ops::dense_gemm::dispatch_dense_matvec_f16w_f32io(
                s.encoder_mut(),
                reg,
                metal_dev,
                &norm_out_batched,
                lm_head_f16,
                &logits_batched,
                &mlx_native::ops::dense_gemm::DenseGemmF16Params {
                    m: n as u32,
                    n: vocab_size as u32,
                    k: hs as u32,
                },
            )
            .map_err(|e| anyhow::anyhow!("batched arg lm_head F16: {e}"))?;
        } else {
            anyhow::bail!(
                "per_position_argmax_batched requires lm_head_q6k / q8 / f16"
            );
        }

        // (c) ONE softcap on the full n*vocab logits (element-wise).
        //     ADR-040 §0.16: the softcap kernel early-returns `if id >= params[1]`,
        //     and the shared `self.activations.softcap_params` carries
        //     `params[1] = vocab_size` (single-row count). For this batched
        //     [n, vocab] buffer that would softcap ONLY position 0 and leave
        //     positions ≥1 RAW (the same defect fixed in lm_head_batched). Use a
        //     per-call params buffer with the full `n * vocab` element count.
        if let Some(cap) = self.final_logit_softcapping {
            let total = n
                .checked_mul(vocab_size as usize)
                .expect("per_position softcap: n*vocab overflow");
            let mut softcap_params_b = dev
                .alloc_buffer(8, mlx_native::DType::F32, vec![2])
                .map_err(|e| anyhow::anyhow!("batched arg softcap params alloc: {e}"))?;
            {
                let p: &mut [f32] = softcap_params_b
                    .as_mut_slice()
                    .map_err(|e| anyhow::anyhow!("batched arg softcap params slice: {e}"))?;
                p[0] = cap;
                p[1] = f32::from_bits(total as u32);
            }
            s.barrier_between(
                &[&logits_batched],
                &[&logits_batched],
            );
            mlx_native::ops::softcap::dispatch_softcap(
                s.encoder_mut(),
                reg,
                metal_dev,
                &logits_batched,
                &logits_batched,
                &softcap_params_b,
                cap,
            )
            .map_err(|e| anyhow::anyhow!("batched arg softcap: {e}"))?;
        }

        // (d) Per-row argmax (the kernel itself isn't batchable; this
        //     stage is cheap compared to lm_head).  We dispatch within
        //     the same session — n small dispatches, one finish().
        for pos in 0..n {
            let logits_row = logits_batched
                .slice_view((pos * (vocab_size as usize) * 4) as u64, vocab_size as usize);
            let argmax_idx_view = argmax_index_all.slice_view((pos * 4) as u64, 1);
            let argmax_val_view = argmax_value_all.slice_view((pos * 4) as u64, 1);
            s.barrier_between(
                &[&logits_batched],
                &[&argmax_idx_view, &argmax_val_view],
            );
            mlx_native::ops::argmax::dispatch_argmax_f32(
                s.encoder_mut(),
                reg,
                metal_dev,
                &logits_row,
                &argmax_idx_view,
                &argmax_val_view,
                &self.activations.argmax_params,
                vocab_size as u32,
            )
            .map_err(|e| anyhow::anyhow!("batched arg argmax L{pos}: {e}"))?;
        }

        // (4) Single commit + wait for ALL iterations.
        s.finish()
            .map_err(|e| anyhow::anyhow!("batched argmax session finish: {e}"))?;

        // (5) Read all argmaxes in one shot.
        let argmaxes: Vec<u32> = argmax_index_all
            .as_slice::<u32>()
            .map_err(|e| anyhow::anyhow!("batched argmax read: {e}"))?
            .iter()
            .take(n)
            .copied()
            .collect();
        Ok(argmaxes)
    }
    /// Read the `[vocab_size]` F32 logits buffer that was produced by the
    /// most recent `forward_decode` (or `forward_prefill`) call.
    ///
    /// Returns a borrowed slice into `self.activations.logits` (no copy).
    /// Caller holds the borrow until they drop the reference; no further
    /// `self.activations` reads invalidate this slice (the underlying
    /// `MlxBuffer` is reused across decode steps but writes only happen
    /// inside the next `forward_decode` invocation).
    ///
    /// **Phase 2a Task #5 / #7 hook (iter-94).**  This is the logits-side
    /// surface that the chat decode loop consults when any Tier 2/3/4
    /// sampling field (`temperature`, `top_p`, `top_k`, `repetition_penalty`,
    /// `logit_bias`) is non-default — it bypasses `forward_decode`'s on-GPU
    /// greedy argmax and runs the pure-Rust `sampler_pure::sample_token`
    /// over these logits instead.  When grammar lands (iter-95+), the same
    /// hook supplies the logits to the GBNF mask before sampling.
    ///
    /// The logits include any post-softcap that the kernel applied
    /// (`final_logit_softcapping` from the GGUF metadata) — sampling
    /// operates on the same logits the on-GPU argmax would have seen.
    ///
    /// # Errors
    /// Forwarded from `MlxBuffer::as_slice` — fails only if the buffer
    /// is in an unreadable state (typically: never written by any
    /// preceding forward call).
    pub fn logits_view(&self) -> Result<&[f32]> {
        let slice: &[f32] = self.activations.logits.as_slice()
            .map_err(|e| anyhow::anyhow!("logits_view read: {e}"))?;
        let v = self.vocab_size;
        anyhow::ensure!(
            slice.len() >= v,
            "logits_view: buffer length {} < vocab_size {}",
            slice.len(), v
        );
        Ok(&slice[..v])
    }

    /// Compute NLL (negative log-likelihood) for `token_id` from the logits buffer
    /// that was produced by the most recent `forward_decode` call.
    ///
    /// Uses log-sum-exp for numerical stability. The logits may include a soft-cap
    /// (tanh * 30) already applied by the kernel; we use them as-is since both
    /// dense and TQ paths apply the same cap, so the relative NLL is fair.
    ///
    /// Returns: -log P(token_id) under the softmax distribution.
    /// Call ONLY immediately after `forward_decode`; the logits buffer is live.
    ///
    /// Public surface for downstream eval/scoring crates; no internal caller.
    #[allow(dead_code)]
    pub fn token_nll_from_logits(&self, token_id: u32) -> Result<f32> {
        let logits: &[f32] = self.activations.logits.as_slice()
            .map_err(|e| anyhow::anyhow!("token_nll logits read: {e}"))?;
        let v = self.vocab_size;
        anyhow::ensure!(
            (token_id as usize) < v,
            "token_nll: token_id {token_id} >= vocab_size {v}"
        );
        let slice = &logits[..v];
        // Log-sum-exp with max subtraction for numerical stability.
        let max_logit = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f64 = slice.iter()
            .map(|&l| ((l - max_logit) as f64).exp())
            .sum();
        let log_sum_exp = max_logit as f64 + sum_exp.ln();
        let log_prob = (logits[token_id as usize] as f64) - log_sum_exp;
        Ok(-log_prob as f32)
    }

}
