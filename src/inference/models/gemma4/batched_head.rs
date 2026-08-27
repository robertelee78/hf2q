//! ADR-040 Phase F M2.2 / S1 — batched lm_head over N gathered decode slots.
//!
//! The single largest matmul in Gemma decode is the artifact-declared output
//! head. Under continuous batching, N scalar calls would reread that matrix N
//! times. This module computes the head over N gathered hidden rows while
//! preserving scalar-row reduction order — the first slice (S1) of the batched
//! decode forward.
//!
//! Quantized types use their row-identical continuous-width route. Dense
//! F32/F16/BF16 types execute one scalar row at a time because their tile
//! reduction order differs. The end-to-end gate is the `slot_aware_*`
//! byte-equivalence suite.

use anyhow::Result;
use mlx_native::graph::GraphSession;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use crate::quantize::imatrix::ImatrixHint;
use crate::serve::gpu::GpuContext;

use super::batched_body::dispatch_dense_rowident;
use super::model::MlxModelWeights;

/// DIAGNOSTIC: one-shot gate so the HF2Q_MVN_ENCODE_TRACE dump fires for only
/// the FIRST lm_head_batched call (avoids 1000s of head calls' worth of noise).
static FIRST_HEAD_TRACE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Output of [`MlxModelWeights::lm_head_batched`]: row-major per-slot logits.
pub struct BatchedHeadOut {
    /// `[n, vocab_size]` post-softcap logits.
    logits: Vec<f32>,
}

impl BatchedHeadOut {
    pub(crate) fn from_logits(logits: Vec<f32>) -> Result<Self> {
        crate::inference::argmax::validate_finite_logits(&logits, "Gemma batched output head")?;
        Ok(Self { logits })
    }

    pub(crate) fn logits(&self) -> &[f32] {
        &self.logits
    }
}

#[cfg(test)]
mod host_payload_contract {
    use super::BatchedHeadOut;

    #[test]
    fn batched_head_host_payload_contains_logits_only() {
        // Deliberately destructure without `..`: adding a second host payload
        // (for example the retired post-norm hidden rows) must break this gate
        // until the transfer is explicitly reviewed and qualified.
        let BatchedHeadOut { logits } = BatchedHeadOut::from_logits(vec![1.0, 2.0]).unwrap();
        assert_eq!(logits, [1.0, 2.0]);
    }

    #[test]
    fn batched_head_host_payload_rejects_nonfinite_logits() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(BatchedHeadOut::from_logits(vec![0.0, value]).is_err());
        }
    }
}

impl MlxModelWeights {
    /// Batched final-norm + lm_head (`m=N`) + softcap over `n` gathered hidden
    /// rows. `hidden_rows` is the row-major `[n, hidden_size]` F32 final hidden
    /// state (one row per slot, captured before final-norm — i.e. the value at
    /// `self.activations.hidden` after the layer loop). Returns the row-major
    /// `[n, vocab_size]` F32 logits, post-softcap, ready for per-row finalize.
    ///
    /// Per-row bit-identical to the scalar head by row-identical dispatch.
    pub fn lm_head_batched(
        &self,
        hidden_rows: &[f32],
        n: usize,
        gpu: &mut GpuContext,
    ) -> Result<BatchedHeadOut> {
        let hs = self.hidden_size;
        let vocab = self.vocab_size;
        if n == 0 {
            return BatchedHeadOut::from_logits(Vec::new());
        }
        if hidden_rows.len() != n * hs {
            anyhow::bail!(
                "lm_head_batched: hidden_rows len {} != n*hidden_size {}*{}",
                hidden_rows.len(),
                n,
                hs
            );
        }
        let lm_head = self.resolved_lm_head();

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

        // ADR-040 §0.16 (2026-06-25) — per-call softcap params with n_elements =
        // n*vocab. ROOT CAUSE of the batched-determinism residual: the shared
        // `self.activations.softcap_params` carries `params[1] = vocab` (the
        // single-row count, init at model.rs:1441), and the softcap kernel
        // (shaders/softcap.metal) early-returns `if id >= params[1]`. In the
        // batched head the logits buffer is `n*vocab`, so ONLY row 0 (id < vocab)
        // got softcapped — rows ≥1 kept their RAW (larger) logits. Row 0 == the
        // serial reference; rows ≥1 diverged, which surfaced as the ~13%
        // staggered-eviction flake (a request landing in a slot ≥1 whose
        // un-softcapped logits flipped a near-tie argmax). Affected BOTH decode
        // paths (per-slot and batched-body both call this with n=N). Fix: a
        // per-call `[cap, n*vocab]` params buffer so every row is softcapped.
        let mut softcap_params_b = dev
            .alloc_buffer(8, DType::F32, vec![2])
            .map_err(|e| anyhow::anyhow!("lm_head_batched alloc softcap params: {e}"))?;
        if let Some(cap) = self.final_logit_softcapping {
            let p: &mut [f32] = softcap_params_b
                .as_mut_slice()
                .map_err(|e| anyhow::anyhow!("lm_head_batched softcap params slice: {e}"))?;
            let total = n
                .checked_mul(vocab)
                .expect("lm_head softcap: n*vocab overflow");
            p[0] = cap;
            p[1] = f32::from_bits(total as u32);
        }

        let mut s = exec
            .begin()
            .map_err(|e| anyhow::anyhow!("lm_head_batched session begin: {e}"))?;

        // Batched final RMS norm: [n,hidden] -> [n,hidden]. norm_params ([eps,dim])
        // is per-element and identical for every row.
        let _bc_dbg = std::env::var("HF2Q_MVN_BARRIER_TRACE").as_deref() == Ok("1");
        let _bc_before = if _bc_dbg {
            mlx_native::barrier_count()
        } else {
            0
        };
        // DIAGNOSTIC (HF2Q_MVN_ENCODE_TRACE=1): scope the mlx-native encode trace
        // (per-dispatch + per-barrier with encoder pointer) to JUST this head
        // call, so codex can read the command-stream order around mN→softcap
        // without the 30-layer body noise. Only the FIRST head call to keep it short.
        let _enc_trace = std::env::var("HF2Q_MVN_ENCODE_TRACE").as_deref() == Ok("1")
            && FIRST_HEAD_TRACE.swap(false, std::sync::atomic::Ordering::Relaxed);
        if _enc_trace {
            eprintln!("[ENCODE-TRACE] === lm_head_batched BEGIN n={} ===", n);
            mlx_native::set_encode_trace(true);
        }
        s.barrier_between(&[&hidden_b, &self.final_norm], &[&normed_b]);
        s.rms_norm(
            reg,
            metal_dev,
            &hidden_b,
            &self.final_norm,
            &normed_b,
            &self.activations.norm_params,
            n as u32,
            hs as u32,
        )
        .map_err(|e| anyhow::anyhow!("lm_head_batched final norm: {e}"))?;

        // Batched lm_head executes the artifact's declared representation.
        s.barrier_between(&[&normed_b, &lm_head.buffer], &[&logits_b]);
        dispatch_dense_rowident(
            &mut s,
            reg,
            dev,
            &normed_b,
            lm_head,
            &logits_b,
            n,
            hs,
            vocab,
            ImatrixHint::Global("output.weight"),
        )?;

        // ADR-040 §0.21c: the lm_head→softcap (and all other) cross-dispatch
        // ordering is now handled correctly by the mlx-native encoder-RETAIN root
        // fix (the autoreleased concurrent encoder was being held via a borrowed
        // pointer → its `memoryBarrierWithScope` did not reliably order a slow
        // producer like the Q6_K mvN lm_head before its in-place softcap consumer;
        // retaining it like the peer fixes it engine-wide). The prior local
        // commit_wait order-fence here is therefore removed — no per-tick CPU stall,
        // mvN keeps full concurrency. See ADR-040 §0.21c-track2 / the encoder-retain
        // commit; verified ≥20 consecutive green runtime-compile + -O3 with the local
        // fence OFF and the retain fix ON.
        // Softcap all n*vocab logits in place (per-call params count above).
        if let Some(cap) = self.final_logit_softcapping {
            s.barrier_between(&[&logits_b], &[&logits_b]);
            mlx_native::ops::softcap::dispatch_softcap(
                s.encoder_mut(),
                reg,
                metal_dev,
                &logits_b,
                &logits_b,
                &softcap_params_b,
                cap,
            )
            .map_err(|e| anyhow::anyhow!("lm_head_batched softcap: {e}"))?;
        }
        if _enc_trace {
            mlx_native::set_encode_trace(false);
            eprintln!("[ENCODE-TRACE] === lm_head_batched END n={} ===", n);
        }
        if _bc_dbg {
            let after = mlx_native::barrier_count();
            eprintln!(
                "[BARRIER-TRACE] lm_head_batched n={} barriers_emitted={}",
                n,
                after - _bc_before
            );
        }
        use crate::inference::models::gemma4::batched_body::host_phases;
        let _hp = std::time::Instant::now();
        s.finish()
            .map_err(|e| anyhow::anyhow!("lm_head_batched session finish: {e}"))?;
        host_phases::add(
            host_phases::Phase::LmheadWait,
            _hp.elapsed().as_nanos() as u64,
        );

        let _hp = std::time::Instant::now();
        let logits: Vec<f32> = logits_b
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("lm_head_batched read logits: {e}"))?
            .to_vec();
        host_phases::add(
            host_phases::Phase::LmheadReadback,
            _hp.elapsed().as_nanos() as u64,
        );
        BatchedHeadOut::from_logits(logits)
    }

    /// ADR-040 §25 iter-L — encode the lm_head (final RMS-norm + native matmul +
    /// softcap) into an EXISTING session `s`, reading `hidden_buf` (the on-GPU
    /// `[n,hidden]` body output) DIRECTLY — no host round-trip and NO separate
    /// session/commit. BYTE-IDENTICAL to [`Self::lm_head_batched`]'s encode (same
    /// dispatches, order, and per-call `[cap, n*vocab]` softcap params — see the
    /// §0.16 root-cause note there). Does NOT finish/readback: the caller appends
    /// this as the final chunk of the body's CB pipeline, then does the single
    /// `commit_and_wait` and reads only `logits_b`. `normed_b` and the third
    /// returned buffer (`softcap_params_b`) are handed back solely so they
    /// outlive the GPU read; neither crosses the host boundary. The caller
    /// currently enables this fused command-buffer path only for its qualified
    /// Q6_K regime.
    pub(crate) fn encode_lm_head_into(
        &self,
        s: &mut GraphSession,
        hidden_buf: &MlxBuffer,
        n: usize,
        dev: &MlxDevice,
        reg: &mut KernelRegistry,
    ) -> Result<(MlxBuffer, MlxBuffer, MlxBuffer)> {
        let hs = self.hidden_size;
        let vocab = self.vocab_size;
        let metal_dev = dev.metal_device();
        let lm_head = self.resolved_lm_head();

        let normed_b = dev
            .alloc_buffer(n * hs * 4, DType::F32, vec![n * hs])
            .map_err(|e| anyhow::anyhow!("encode_lm_head_into alloc normed_b: {e}"))?;
        let logits_b = dev
            .alloc_buffer(n * vocab * 4, DType::F32, vec![n * vocab])
            .map_err(|e| anyhow::anyhow!("encode_lm_head_into alloc logits_b: {e}"))?;
        // Per-call softcap params with n_elements = n*vocab (§0.16 root cause —
        // the shared self.activations.softcap_params carries vocab, softcapping
        // ONLY row 0). MUST be per-call here too.
        let mut softcap_params_b = dev
            .alloc_buffer(8, DType::F32, vec![2])
            .map_err(|e| anyhow::anyhow!("encode_lm_head_into alloc softcap params: {e}"))?;
        if let Some(cap) = self.final_logit_softcapping {
            let p: &mut [f32] = softcap_params_b
                .as_mut_slice()
                .map_err(|e| anyhow::anyhow!("encode_lm_head_into softcap params slice: {e}"))?;
            let total = n
                .checked_mul(vocab)
                .expect("encode_lm_head_into softcap: n*vocab overflow");
            p[0] = cap;
            p[1] = f32::from_bits(total as u32);
        }

        // RAW: the body's final layer wrote `hidden_buf` into the SAME session's
        // tracker, so this barrier_between (reads hidden_buf) emits the memory
        // barrier that orders the body-hidden-write before the final norm.
        s.barrier_between(&[hidden_buf, &self.final_norm], &[&normed_b]);
        s.rms_norm(
            reg,
            metal_dev,
            hidden_buf,
            &self.final_norm,
            &normed_b,
            &self.activations.norm_params,
            n as u32,
            hs as u32,
        )
        .map_err(|e| anyhow::anyhow!("encode_lm_head_into final norm: {e}"))?;

        s.barrier_between(&[&normed_b, &lm_head.buffer], &[&logits_b]);
        dispatch_dense_rowident(
            s,
            reg,
            dev,
            &normed_b,
            lm_head,
            &logits_b,
            n,
            hs,
            vocab,
            ImatrixHint::Global("output.weight"),
        )?;

        if let Some(cap) = self.final_logit_softcapping {
            s.barrier_between(&[&logits_b], &[&logits_b]);
            mlx_native::ops::softcap::dispatch_softcap(
                s.encoder_mut(),
                reg,
                metal_dev,
                &logits_b,
                &logits_b,
                &softcap_params_b,
                cap,
            )
            .map_err(|e| anyhow::anyhow!("encode_lm_head_into softcap: {e}"))?;
        }

        Ok((logits_b, normed_b, softcap_params_b))
    }
}
