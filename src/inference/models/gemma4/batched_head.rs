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
use mlx_native::graph::GraphSession;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use crate::debug::INVESTIGATION_ENV;
use crate::quantize::imatrix::ImatrixHint;
use crate::serve::forward_mlx_shared::dispatch_qmatmul;
use crate::serve::gpu::GpuContext;

use super::model::MlxModelWeights;

/// DIAGNOSTIC: one-shot gate so the HF2Q_MVN_ENCODE_TRACE dump fires for only
/// the FIRST lm_head_batched call (avoids 1000s of head calls' worth of noise).
static FIRST_HEAD_TRACE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

// ADR-040 §26 — gated rerank profiling (HF2Q_RERANK_PROFILE=1).
static RERANK_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static RERANK_CAND: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static RERANK_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static RERANK_PROFILE_ON: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var("HF2Q_RERANK_PROFILE").as_deref() == Ok("1"));
/// (total_rerank_ns, total_candidates, calls) since last reset.
pub fn rerank_profile() -> (u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (RERANK_NS.load(Relaxed), RERANK_CAND.load(Relaxed), RERANK_CALLS.load(Relaxed))
}
pub fn rerank_profile_reset() {
    use std::sync::atomic::Ordering::Relaxed;
    RERANK_NS.store(0, Relaxed); RERANK_CAND.store(0, Relaxed); RERANK_CALLS.store(0, Relaxed);
}

/// Output of [`MlxModelWeights::lm_head_batched`]: row-major per-slot logits and
/// the post-final-norm hidden rows (the exact-F32 rerank operand that
/// [`MlxModelWeights::finalize_token_from_logits`] needs).
pub struct BatchedHeadOut {
    /// `[n, vocab_size]` post-softcap logits.
    pub logits: Vec<f32>,
    /// `[n, hidden_size]` post-final-norm hidden (the `hidden·embed` operand).
    pub normed: Vec<f32>,
}

impl MlxModelWeights {
    /// ADR-040 S1b — shared per-row finalize: Q8/Q6_K coarse-logit → exact-F32
    /// argmax rerank. Pure CPU. Extracted verbatim from the scalar
    /// `forward_decode` tail (forward_gpu.rs:1008–1077) so the scalar and
    /// batched-lm_head paths produce BIT-IDENTICAL tokens by construction.
    ///
    /// Inputs are per-row slices so either path can supply its own buffers:
    /// - `logits_row`: post-softcap logits `[vocab_size]`.
    /// - `normed_row`: post-final-norm hidden `[hidden_size]` (the `hidden·embed`
    ///   rerank operand — scalar passes `norm_out`, batched passes its `normed_b`
    ///   row).
    /// - `gpu_top1` / `top1_val`: the GPU-argmax index+value over `logits_row`
    ///   (both paths run the SAME `dispatch_argmax_f32` kernel, so these match).
    ///
    /// When rerank is inactive (F16 head, or `HF2Q_LMHEAD_RERANK=0`) returns
    /// `gpu_top1` unchanged — identical to the scalar `else` arm.
    pub(crate) fn finalize_token_from_logits(
        &self,
        logits_row: &[f32],
        normed_row: &[f32],
        gpu_top1: u32,
        top1_val: f32,
    ) -> Result<u32> {
        let vocab_size = self.vocab_size;
        let hs = self.hidden_size;
        let rerank_active = (self.lm_head_q8.is_some() || self.lm_head_q6k.is_some())
            && !INVESTIGATION_ENV.lmhead_rerank_disabled;
        if !rerank_active {
            return Ok(gpu_top1);
        }
        // Headroom for Q8 noise (≈5e-3); delta=0.5 keeps the candidate set small
        // while guaranteeing the true winner is included. Verbatim from scalar.
        let delta: f32 = 0.5;
        let threshold = top1_val - delta;

        let embed_f32: &[f32] = self
            .embed_weight
            .as_slice()
            .map_err(|e| anyhow::anyhow!("finalize rerank embed read: {e}"))?;

        let mut candidates: Vec<u32> = Vec::with_capacity(64);
        for (i, &v) in logits_row[..vocab_size].iter().enumerate() {
            if v >= threshold {
                candidates.push(i as u32);
            }
        }
        for sp in [0u32, 1, 2, 105, 106] {
            if (sp as usize) < vocab_size {
                candidates.push(sp);
            }
        }
        candidates.sort_unstable();
        candidates.dedup();

        // ADR-040 §26 — gated profiling (HF2Q_RERANK_PROFILE=1): is finalize's cost
        // the candidate full-vocab scan (movable to GPU, F32) or the F64 rerank dots
        // (Metal-no-F64, stuck on host)? Counts candidates + times the rerank loop.
        let _rr_prof = *RERANK_PROFILE_ON;
        let _rr_t = if _rr_prof { Some(std::time::Instant::now()) } else { None };
        // Exact F32 rerank via hidden · embed_row. Softcap is monotonic so
        // skipping it doesn't change argmax order. F64 accumulator for precision.
        let mut best_tok: u32 = gpu_top1;
        let mut best_logit: f32 = f32::NEG_INFINITY;
        for &tok in &candidates {
            let row_off = (tok as usize) * hs;
            if row_off + hs > embed_f32.len() {
                continue;
            }
            let row = &embed_f32[row_off..row_off + hs];
            let mut acc: f64 = 0.0;
            for i in 0..hs {
                acc += (normed_row[i] as f64) * (row[i] as f64);
            }
            let l = acc as f32;
            if l > best_logit {
                best_logit = l;
                best_tok = tok;
            }
        }
        if let Some(t) = _rr_t {
            RERANK_NS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            RERANK_CAND.fetch_add(candidates.len() as u64, std::sync::atomic::Ordering::Relaxed);
            RERANK_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(best_tok)
    }

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
    ) -> Result<BatchedHeadOut> {
        let hs = self.hidden_size;
        let vocab = self.vocab_size;
        if n == 0 {
            return Ok(BatchedHeadOut { logits: Vec::new(), normed: Vec::new() });
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
            let total = n.checked_mul(vocab).expect("lm_head softcap: n*vocab overflow");
            p[0] = cap;
            p[1] = f32::from_bits(total as u32);
        }

        let mut s = exec
            .begin()
            .map_err(|e| anyhow::anyhow!("lm_head_batched session begin: {e}"))?;

        // Batched final RMS norm: [n,hidden] -> [n,hidden]. norm_params ([eps,dim])
        // is per-element and identical for every row.
        let _bc_dbg = std::env::var("HF2Q_MVN_BARRIER_TRACE").as_deref() == Ok("1");
        let _bc_before = if _bc_dbg { mlx_native::barrier_count() } else { 0 };
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

        // ADR-040 §0.21c: the lm_head→softcap (and all other) cross-dispatch
        // ordering is now handled correctly by the mlx-native encoder-RETAIN root
        // fix (the autoreleased concurrent encoder was being held via a borrowed
        // pointer → its `memoryBarrierWithScope` did not reliably order a slow
        // producer like the Q6_K mvN lm_head before its in-place softcap consumer;
        // retaining it like llama fixes it engine-wide). The prior local
        // commit_wait order-fence here is therefore removed — no per-tick CPU stall,
        // mvN keeps full concurrency. See ADR-040 §0.21c-track2 / the encoder-retain
        // commit; verified ≥20 consecutive green runtime-compile + -O3 with the local
        // fence OFF and the retain fix ON.
        // Softcap all n*vocab logits in place (per-call params count above).
        if let Some(cap) = self.final_logit_softcapping {
            s.barrier_between(&[&logits_b], &[&logits_b]);
            mlx_native::ops::softcap::dispatch_softcap(
                s.encoder_mut(), reg, metal_dev,
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
            eprintln!("[BARRIER-TRACE] lm_head_batched n={} barriers_emitted={}", n, after - _bc_before);
        }
        use crate::inference::models::gemma4::batched_body::host_phases;
        let _hp = std::time::Instant::now();
        s.finish()
            .map_err(|e| anyhow::anyhow!("lm_head_batched session finish: {e}"))?;
        host_phases::add(host_phases::Phase::LmheadWait, _hp.elapsed().as_nanos() as u64);

        let _hp = std::time::Instant::now();
        let logits: Vec<f32> = logits_b
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("lm_head_batched read logits: {e}"))?
            .to_vec();
        let normed: Vec<f32> = normed_b
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("lm_head_batched read normed_b: {e}"))?
            .to_vec();
        host_phases::add(host_phases::Phase::LmheadReadback, _hp.elapsed().as_nanos() as u64);
        Ok(BatchedHeadOut { logits, normed })
    }

    /// ADR-040 §25 iter-L — encode the lm_head (final RMS-norm + Q6_K matmul +
    /// softcap) into an EXISTING session `s`, reading `hidden_buf` (the on-GPU
    /// `[n,hidden]` body output) DIRECTLY — no host round-trip and NO separate
    /// session/commit. BYTE-IDENTICAL to [`Self::lm_head_batched`]'s encode (same
    /// dispatches, order, and per-call `[cap, n*vocab]` softcap params — see the
    /// §0.16 root-cause note there). Does NOT finish/readback: the caller appends
    /// this as the final chunk of the body's CB pipeline, then does the single
    /// `commit_and_wait` and reads the returned `(logits_b, normed_b)`. The third
    /// returned buffer (`softcap_params_b`) is handed back so it outlives the GPU
    /// read (kept alive until the caller's finish). Q6_K head only.
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
        let q6k = self.lm_head_q6k.as_ref().ok_or_else(|| {
            anyhow::anyhow!("encode_lm_head_into requires a Q6_K lm_head (production decode path)")
        })?;

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

        s.barrier_between(&[&normed_b, &q6k.buffer], &[&logits_b]);
        dispatch_qmatmul(
            s,
            reg,
            dev,
            &normed_b,
            q6k,
            &logits_b,
            n as u32,
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
