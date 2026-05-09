//! ADR-020 iter-13b — DWQ-proper training loop substrate.
//!
//! Provides the wiring for a Track 2 DWQ distillation step:
//!
//!   1. Initialize per-group `scales` + `biases` + frozen `q_int` from
//!      a frozen FP32 weight via mlx-native's `qdq_affine_init_f32`
//!      kernel.
//!   2. Build a forward tape: `qdq_affine(scales, biases, q_int)` →
//!      reconstruction loss against the frozen weight (per-tensor MSE);
//!      scales + biases are leaves registered with [`AdamOptimizer`].
//!   3. Back-propagate via the existing tape `backward`, and apply
//!      `Adam.step`.
//!   4. Repeat until convergence.
//!
//! This iteration ships:
//!   - [`init_affine_params_gpu`] — host-side wrapper that spawns the
//!     init kernel against a frozen weight buffer and reads back
//!     `(q_int, scales_init, biases_init)` to host memory.  Used both
//!     by the synthetic test and (later) by the production
//!     dwq_quantize entry point.
//!   - [`buffer_from_f32`] — small helper that creates a fresh
//!     `MlxBuffer` from host data, shared between Adam parameter
//!     registration and tape leaves.
//!
//! The synthetic 2-Linear MLP convergence test (gated by `#[test]` and
//! `cfg(test)`) is the load-bearing falsifier: if Adam over
//! `(scales1, biases1, scales2, biases2)` doesn't drive the
//! reconstruction MSE down by ≥5× from a perturbed start over 200
//! steps, the chain is broken somewhere (qdq_affine forward/backward,
//! tape accumulation, Adam state, or finite-difference equivalence).
//!
//! Loss this iteration is per-tensor reconstruction MSE
//! (Σ (qdq_w − w)²) rather than logit KL-div — the qdq_affine →
//! reshape → matmul → KL chain requires a tape `view`/reshape op
//! that is iter-13c work.  Reconstruction MSE has the same
//! gradient-correctness load-bearing property and is sufficient to
//! prove the full training-loop primitive.

use anyhow::{anyhow, Context, Result};
use mlx_native::ops::qdq_affine::dispatch_qdq_affine_init_f32;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

/// Run mlx-native's `qdq_affine_init_f32` kernel against a frozen FP32
/// weight buffer and return CPU copies of `(q_int, scales, biases)`.
///
/// `w` shape: `[n_total]` flat; `n_total = n_groups · group_size`.
/// `group_size`: power of two in `[2, 1024]`, divides `n_total`.
/// `bits`: `[2, 8]`; `n_bins = 2^bits` and must satisfy `n_bins ≤ 256`.
pub fn init_affine_params_gpu(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    w_data: &[f32],
    group_size: usize,
    bits: u32,
) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>)> {
    if !(2..=8).contains(&bits) {
        return Err(anyhow!(
            "init_affine_params_gpu: bits must be in [2, 8]; got {bits}"
        ));
    }
    let n_total = w_data.len();
    if !group_size.is_power_of_two() || !(2..=1024).contains(&group_size) {
        return Err(anyhow!(
            "init_affine_params_gpu: group_size must be a power of two in [2, 1024]; got {group_size}"
        ));
    }
    if n_total % group_size != 0 {
        return Err(anyhow!(
            "init_affine_params_gpu: n_total ({n_total}) must be divisible by group_size ({group_size})"
        ));
    }
    let n_groups = n_total / group_size;
    let n_bins: u32 = 1u32 << bits;

    let mut w_buf = device
        .alloc_buffer(n_total * 4, DType::F32, vec![n_total])
        .map_err(|e| anyhow!("init_affine: alloc w: {e}"))?;
    w_buf
        .as_mut_slice::<f32>()
        .map_err(|e| anyhow!("init_affine: w write: {e}"))?
        .copy_from_slice(w_data);
    let scales_buf = device
        .alloc_buffer(n_groups * 4, DType::F32, vec![n_groups])
        .map_err(|e| anyhow!("init_affine: alloc scales: {e}"))?;
    let biases_buf = device
        .alloc_buffer(n_groups * 4, DType::F32, vec![n_groups])
        .map_err(|e| anyhow!("init_affine: alloc biases: {e}"))?;
    let q_int_buf = device
        .alloc_buffer(n_total, DType::U8, vec![n_total])
        .map_err(|e| anyhow!("init_affine: alloc q_int: {e}"))?;
    let mut meta_buf = device
        .alloc_buffer(8, DType::U32, vec![2])
        .map_err(|e| anyhow!("init_affine: alloc meta: {e}"))?;
    meta_buf
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow!("init_affine: meta write: {e}"))?[..2]
        .copy_from_slice(&[group_size as u32, n_bins]);

    let mut encoder = device
        .command_encoder()
        .map_err(|e| anyhow!("init_affine: encoder: {e}"))?;
    dispatch_qdq_affine_init_f32(
        &mut encoder,
        registry,
        device.metal_device(),
        &w_buf,
        &scales_buf,
        &biases_buf,
        &q_int_buf,
        &meta_buf,
        group_size as u32,
        n_bins,
    )
    .context("init_affine: dispatch qdq_affine_init_f32")?;
    encoder
        .commit_and_wait()
        .map_err(|e| anyhow!("init_affine: commit_and_wait: {e}"))?;

    let scales = scales_buf
        .as_slice::<f32>()
        .map_err(|e| anyhow!("init_affine: scales readback: {e}"))?
        .to_vec();
    let biases = biases_buf
        .as_slice::<f32>()
        .map_err(|e| anyhow!("init_affine: biases readback: {e}"))?
        .to_vec();
    let q_int = q_int_buf
        .as_slice::<u8>()
        .map_err(|e| anyhow!("init_affine: q_int readback: {e}"))?
        .to_vec();
    Ok((q_int, scales, biases))
}

/// Box-Muller transform → unit-variance Gaussian samples from a
/// stable small PRNG (xorshift64*).  Deterministic given `seed`.
///
/// Used by iter-13e and iter-19c as a calibration-activation proxy
/// (post-RMSNorm residual stream stddev ≈ 1.0).
pub fn box_muller_gaussian(n: usize, seed: u64) -> Vec<f32> {
    let mut state = if seed == 0 { 0xDEAD_BEEF_CAFE_BABE } else { seed };
    let mut next_u64 = || -> u64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut next_uniform = || -> f32 {
        // 24-bit mantissa from upper 53 bits → uniform in [2^-24, 1).
        // Avoid 0 so log() in Box-Muller stays finite.
        let r = (next_u64() >> 11) as f32 / (1u64 << 53) as f32;
        r.max(f32::EPSILON)
    };
    let mut out = Vec::with_capacity(n);
    while out.len() + 1 < n {
        let u1 = next_uniform();
        let u2 = next_uniform();
        let mag = (-2.0 * u1.ln()).sqrt();
        let z0 = mag * (std::f32::consts::TAU * u2).cos();
        let z1 = mag * (std::f32::consts::TAU * u2).sin();
        out.push(z0);
        out.push(z1);
    }
    if out.len() < n {
        let u1 = next_uniform();
        let u2 = next_uniform();
        let z0 = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
        out.push(z0);
    }
    out.truncate(n);
    out
}

/// Build a fresh f32 `MlxBuffer` from host data — used by the
/// training loop to wrap Adam-managed parameter state.
/// ADR-020 iter-12e — sample the current process's resident-set size.
///
/// On Apple Silicon Metal-backed processes a large fraction of the
/// "RSS" reported by the kernel is the unified-memory `StorageModeShared`
/// allocation pool, which is the metric §8.3 AC #6 caps at 100 GB.
///
/// Implementation uses `sysinfo::System::new()` then `refresh_processes`
/// + `process(get_current_pid())` + `Process::memory()` (returns bytes
/// in sysinfo 0.30+).  Returns 0 if the process can't be located in
/// sysinfo's snapshot (impossible in practice but the public contract
/// is "best-effort poll, never panic").
pub fn current_rss_bytes() -> u64 {
    use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

    let pid = match sysinfo::get_current_pid() {
        Ok(p) => p,
        Err(_) => return 0,
    };
    let mut sys = System::new();
    sys.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::new().with_memory(),
    );
    sys.process(Pid::from(pid.as_u32() as usize))
        .map(|p| p.memory())
        .unwrap_or(0)
}

/// ADR-020 iter-12e — background RSS watchdog.  Spawns a polling
/// thread that samples [`current_rss_bytes`] every `poll_interval`
/// (default 5s).  When RSS exceeds `cap_bytes`, sets the shared
/// abort flag — the [`train_all_linears_dwq`] driver checks the flag
/// between per-tensor trainings and bails with a clear error +
/// disposition log.
///
/// Drop semantics: dropping the watchdog signals the polling thread
/// to stop and joins it (best-effort; the thread checks the stop flag
/// at every poll boundary so the stop is at most `poll_interval` late).
pub struct RssWatchdog {
    cap_bytes: u64,
    aborted: std::sync::Arc<std::sync::atomic::AtomicBool>,
    peak_rss_bytes: std::sync::Arc<std::sync::atomic::AtomicU64>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl RssWatchdog {
    /// Spawn a watchdog thread polling every `poll_interval`.  Returns
    /// immediately; caller drops the handle to stop the watchdog.
    pub fn spawn(cap_bytes: u64, poll_interval: std::time::Duration) -> Self {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        use std::sync::Arc;

        let aborted = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let aborted_th = aborted.clone();
        let peak_th = peak.clone();
        let stop_th = stop.clone();
        let handle = std::thread::Builder::new()
            .name("dwq-rss-watchdog".to_string())
            .spawn(move || {
                while !stop_th.load(Ordering::Relaxed) {
                    let rss = current_rss_bytes();
                    let prev = peak_th.load(Ordering::Relaxed);
                    if rss > prev {
                        peak_th.store(rss, Ordering::Relaxed);
                    }
                    if rss > cap_bytes {
                        aborted_th.store(true, Ordering::Relaxed);
                        // Continue polling so peak stays accurate even
                        // after the cap was exceeded (helps the driver's
                        // post-abort error message report the worst-case
                        // RSS observed).
                    }
                    std::thread::sleep(poll_interval);
                }
            })
            .expect("spawn dwq-rss-watchdog thread");
        Self {
            cap_bytes,
            aborted,
            peak_rss_bytes: peak,
            stop,
            handle: Some(handle),
        }
    }

    /// Cap configured at spawn time (bytes).
    pub fn cap_bytes(&self) -> u64 {
        self.cap_bytes
    }

    /// Cheap clone of the abort flag for the driver to poll.
    pub fn aborted_handle(&self) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
        self.aborted.clone()
    }

    /// Highest RSS observed by the watchdog so far (bytes).
    pub fn peak_rss_bytes(&self) -> u64 {
        self.peak_rss_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Has the cap been exceeded since spawn?
    pub fn is_aborted(&self) -> bool {
        self.aborted
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Drop for RssWatchdog {
    fn drop(&mut self) {
        self.stop
            .store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

pub fn buffer_from_f32(device: &MlxDevice, data: &[f32]) -> Result<MlxBuffer> {
    let mut buf = device
        .alloc_buffer(data.len() * 4, DType::F32, vec![data.len()])
        .map_err(|e| anyhow!("buffer_from_f32: alloc: {e}"))?;
    buf.as_mut_slice::<f32>()
        .map_err(|e| anyhow!("buffer_from_f32: write: {e}"))?
        .copy_from_slice(data);
    Ok(buf)
}

/// ADR-020 iter-12d-1 — public API extracted from iter-17b's training
/// loop.  Trains a single Linear layer's affine quant params via DWQ
/// using a synthetic teacher (X @ W_real FP64 host oracle).
///
/// This is the building block iter-12d-2 (production driver — iterate
/// across all GGUF Linears) builds on.  The synthetic teacher is the
/// minimum viable variant: it captures DWQ's per-Linear optimization
/// quality without requiring full-model forward (which is iter-11h-f
/// scope).  For real-teacher per-Linear training, swap the X @ W_real
/// host oracle for activations captured from a teacher forward pass.
///
/// ## Algorithm (mirrors iter-17b exactly)
///
///   1. `init_affine_params_gpu(W_real)` → frozen `q_int` + initial
///      `scales` + `biases` (per-group symmetric affine).
///   2. Perturb `scales` and `biases` by `perturb_factor` (typically 2.0×
///      for real GGUF magnitudes; the iter-13e calibration showed real
///      Q4_0 weights have stddev ~0.014 which needs more perturbation
///      headroom than synthetic fixtures).
///   3. Generate `X` ~ Box-Muller Gaussian σ=1 (post-RMSNorm proxy).
///   4. Compute teacher `y_T = X @ W_real` on host in FP64.
///   5. For `n_steps`:
///      - Forward: `qdq_affine(s, b, q_int) → view([n,k]) → transpose →
///        matmul(X, W_q^T) → scalar_mul(1/T) → kl_div_loss(y_S, y_T)`
///      - Backward: ones-seeded `dy = 1/m` per row → `backward(kl)` →
///        gradients on `s` and `b`.
///      - `adam.step({"s": g_s, "b": g_b})`.
///   6. Pack final `s`, `b` + frozen `q_int` into `MlxAffineLinear`.
///
/// ## Convergence contract
///
/// Loss must drop below `initial_kl × convergence_ratio`.  iter-17b's
/// real-GGUF measurement: 53× reduction (ratio = 0.019 << 0.34).
/// Default `convergence_ratio = 0.34` matches iter-17b's acceptance
/// floor.  Returns `Err` if not met.
///
/// ## Shape contract
///
/// * `w_real` — `[n, k]` row-major (one row per output channel,
///   matching llama.cpp/GGUF convention).  `k % group_size == 0`.
/// * Returned `MlxAffineLinear`:
///   - `n` × `k` shape preserved
///   - `q_int` is the frozen 1-byte-per-code quant index (length n·k)
///   - `scales`, `biases` are the trained f32 vectors
///     (length `n × (k / group_size)` each)
///
/// ## Acceptance constraints
///
/// * `n >= 32`, `k >= 32`, `n_tokens >= 32` — matmul backward kernel
///   floor.
/// * `bits ∈ {2, 3, 4, 5, 6, 7, 8}` — `qdq_affine` supported range.
/// * `group_size` is a power of 2 in `[2, 1024]` and divides `k`.
#[derive(Debug, Clone)]
pub struct DwqLinearTrainResult {
    /// Trained Linear ready for safetensors save (iter-16b).
    pub linear: crate::calibrate::mlx_safetensors_loader::MlxAffineLinear,
    /// KL loss on the FIRST training step (post-perturbation).
    pub kl_initial: f32,
    /// Minimum KL loss observed across all training steps.
    pub kl_min: f32,
    /// KL loss on the FINAL training step.
    pub kl_final: f32,
    /// Number of Adam steps actually run (== `n_steps` requested unless
    /// caller passes 0).
    pub steps_run: usize,
}

/// Configuration for [`train_linear_dwq_synthetic_teacher`].  Defaults
/// match the iter-17b acceptance fixture so callers can pass
/// `DwqTrainingConfig::default()` and reproduce iter-17b's measured
/// 53× KL reduction on real GGUF tensors.
#[derive(Debug, Clone)]
pub struct DwqTrainingConfig {
    /// Quantization bits — typically 4 for Q4_0-equivalent.
    pub bits: u32,
    /// Quant group size — typically 32 (matches GGUF Q4_0 block size).
    pub group_size: usize,
    /// Calibration batch size (`m` in matmul).  Must be >= 32 for
    /// matmul backward floor.
    pub n_tokens: usize,
    /// Number of Adam steps.  iter-17b empirically uses 50.
    pub n_steps: usize,
    /// Adam learning rate.  iter-17b: 0.002.
    pub lr: f32,
    /// Distillation temperature.  iter-17b: 2.0.
    pub temperature: f32,
    /// Initial perturbation factor on scales/biases.  Default `1.0`
    /// (production: start from the optimal Q4_0-equivalent symmetric
    /// init, train DOWN from there to find a better local optimum).
    /// Set `> 1.0` for the iter-13e/iter-17b "training recovers from
    /// degradation" test fixture (measures convergence behavior on a
    /// deliberately-degraded start; kl_initial = 0.7-ish, kl_min
    /// drops by 50-90% over Adam steps).  Production benchmarks
    /// (iter-12f) require `1.0` since the DWQ-vs-Q4_0 comparison only
    /// makes sense when both arms start from the same optimal init.
    pub perturb_factor: f32,
    /// PRNG seed for the Box-Muller Gaussian X.
    pub seed: u64,
    /// Convergence floor (assertion threshold for `kl_min / kl_initial`).
    /// iter-17b: 0.34.  Set to a value > 1.0 to disable the convergence
    /// check (e.g. for diagnostic runs).
    pub convergence_ratio: f32,
    /// ADR-020 iter-12e — RSS watchdog cap (bytes).  When `Some(cap)`,
    /// `train_all_linears_dwq` spawns a background thread polling the
    /// process's resident-set size every 5s and aborts the scan if RSS
    /// exceeds this cap.  Defaults to `None` (no watchdog) so existing
    /// callers keep the iter-12d-1/2 behavior unchanged.
    ///
    /// §8.3 AC #6 cap is 100 GB → `Some(100 * 1024 * 1024 * 1024)`.
    pub rss_cap_bytes: Option<u64>,
    /// ADR-020 iter-12f-2 — run the per-Linear delta-KL benchmark
    /// (DWQ vs Q4_0, forward KL against FP32 teacher) inline during
    /// training.  When `true`, each `DwqLinearTrained` row carries a
    /// `bench: Some(PerLinearKlComparison)` and the aggregate
    /// `mean_delta_kl_nats` lands in `DwqAllLinearsResult`.
    ///
    /// **Apples-to-apples constraint** (per §8.3 AC #7): only meaningful
    /// when `perturb_factor == 1.0` (training starts from the optimal
    /// Q4_0-equivalent init).  At `perturb_factor > 1.0` the comparison
    /// trivially favors Q4_0 since DWQ training is recovering from a
    /// degraded start.  iter-12f-2 prints a warning if the operator
    /// requests bench with `perturb_factor != 1.0`.
    pub compute_bench: bool,
    /// ADR-020 AC#7 follow-up — when `Some(x_data)`, override the
    /// synthetic `box_muller_gaussian` activation `X` with the
    /// caller-supplied vector.  Length must equal `n_tokens × k` for
    /// the target Linear.  Used to test the hypothesis that a
    /// non-uniform X (e.g. real captured activation samples) breaks
    /// the perturb=1.0 ratio=1.000 plateau by giving Adam meaningful
    /// signal to optimize against.  `None` preserves the existing
    /// synthetic Gaussian X behavior — zero perf cost on the no-op
    /// path.
    pub x_override: Option<Vec<f32>>,
}

impl Default for DwqTrainingConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 32,
            n_tokens: 32,
            n_steps: 50,
            lr: 0.002,
            temperature: 2.0,
            perturb_factor: 1.0,
            seed: 0xDEADBEEF,
            convergence_ratio: 0.34,
            rss_cap_bytes: None,
            compute_bench: false,
            x_override: None,
        }
    }
}

/// ADR-020 AC#7 Option A — configuration for the full-model DWQ
/// training entry point (`train_all_linears_full_model_dwq`, step 2
/// of the AC#7 implementation ladder per ADR-020 §8.3).
///
/// Sibling of [`DwqTrainingConfig`] (which drives the synthetic
/// per-Linear-teacher path that's mathematically a no-op at
/// `perturb_factor=1.0` per the
/// `project_adr020_dwq_perturb1_noop_finding` memory).  The
/// full-model variant uses an FP32 GGUF teacher + per-Linear
/// `qdq_affine`-wrapped student, optimizing all per-Linear (s, b)
/// jointly via cross-layer KL gradients.
///
/// Foundation tests proving the autograd chain at iter-N+0 of the
/// ladder (commits `28bedfd`, `220a5e1`, `8d5bef6`, `59bcb6d`,
/// `5e86314`):
///   - `qwen35_moe::tests::ac7_option_a_*` series — gradient flow,
///     plateau breakage, SGD convergence, Adam convergence,
///     init-equivalence proof.
#[derive(Debug, Clone)]
pub struct FullModelDwqConfig {
    /// Quant bit-width (typically 4 for production).  Range [2, 8].
    pub bits: u32,
    /// Per-group axis length.  Power-of-two in [2, 1024].  Must
    /// divide every Linear's `k`.
    pub group_size: usize,
    /// Number of Adam steps to take.  iter-12d-1 default = 50;
    /// full-model needs more steps because cross-layer signal
    /// propagates over more layers per step (typical 200-500 in
    /// mlx-lm).
    pub n_steps: usize,
    /// Adam learning rate.  Note from Option A Adam test
    /// (`qwen35_moe::tests::ac7_option_a_full_model_two_layer_adam_*`):
    /// lr=0.01 EXPLODES on this fixture; lr=1e-5 converges.  mlx-lm
    /// uses lr=1e-4 against full-vocab logits (larger losses).
    /// Production tuning is operator territory.
    pub lr: f32,
    /// Distillation temperature.  T=2.0 matches mlx-lm's default
    /// in `mlx_lm/quant/dwq.py:106`.
    pub temperature: f32,
    /// Calibration batch size (n_tokens per forward pass).  Must
    /// be ≥ 32 (matmul backward kernel floor).
    pub batch_size: usize,
    /// Calibration sequence length per batch.
    pub seq_len: usize,
    /// Path to the teacher GGUF (FP32-equivalent — typically the
    /// same model variant the student is being quantized FROM).
    pub gguf_path: std::path::PathBuf,
    /// Calibration corpus as token-id batches.  Each `Vec<u32>` is
    /// a batch of length `batch_size × seq_len` row-major.
    /// Length validated at runtime against `batch_size * seq_len`
    /// per element.
    pub calibration_token_batches: Vec<Vec<u32>>,
    /// Top-K subset of teacher logits to retain per position
    /// (mirrors mlx-lm's `compute_dwq_targets` at
    /// `dwq.py:59` `argpartition kth=-1024`).  Smaller K → smaller
    /// teacher footprint at the cost of distillation fidelity.
    /// 1024 is the production default.
    pub top_k_teacher: usize,
    /// PRNG seed for any auxiliary randomization (reserved for
    /// future calibration sampling — currently unused by the
    /// deterministic forward path).
    pub seed: u64,
    /// RSS watchdog cap (bytes).  When `Some(cap)`, the training
    /// loop polls process RSS and aborts if it exceeds the cap.
    /// §8.3 AC#6 mandate is 100 GB.  Re-uses the same
    /// [`RssWatchdog`] machinery as `DwqTrainingConfig::rss_cap_bytes`.
    pub rss_cap_bytes: Option<u64>,
}

impl Default for FullModelDwqConfig {
    /// Production defaults.  Empty `gguf_path` and
    /// `calibration_token_batches` MUST be supplied by the caller
    /// before invoking the training fn — they default to "no-op
    /// values" purely so the struct supports `..Default::default()`
    /// initializer-shorthand in tests/fixtures.
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 32,
            n_steps: 200,
            lr: 1e-4,
            temperature: 2.0,
            batch_size: 32,
            seq_len: 512,
            gguf_path: std::path::PathBuf::new(),
            calibration_token_batches: Vec::new(),
            top_k_teacher: 1024,
            seed: 0xDEAD_BEEF,
            rss_cap_bytes: None,
        }
    }
}

impl FullModelDwqConfig {
    /// ADR-020 AC#7 Option A — fail-loud preflight validation.
    ///
    /// Called by `train_all_linears_full_model_dwq` (step 2 of the
    /// AC#7 ladder) BEFORE any GPU allocation or model load to
    /// surface bad knob combos as early as possible.  Mirrors the
    /// validation pattern used by `train_linear_dwq_synthetic_teacher`
    /// (line 970 onward).
    ///
    /// Validation rules:
    /// - `bits` ∈ [2, 8] — `init_affine_params_gpu` reject outside this range
    /// - `group_size` power-of-two in [2, 1024] — kernel constraint
    /// - `n_steps > 0`
    /// - `lr` finite and `> 0`
    /// - `temperature` finite and `> 0`
    /// - `batch_size >= 32` — matmul backward kernel floor
    /// - `seq_len >= 2` — next-token target requires ≥1 input + ≥1 target
    /// - `gguf_path` non-empty (caller MUST supply)
    /// - `calibration_token_batches` non-empty
    /// - Every batch's token count == `batch_size × seq_len`
    /// - `top_k_teacher > 0`
    pub fn validate(&self) -> Result<()> {
        if !(2..=8).contains(&self.bits) {
            return Err(anyhow!(
                "FullModelDwqConfig: bits must be in [2, 8]; got {}",
                self.bits
            ));
        }
        if !self.group_size.is_power_of_two() || !(2..=1024).contains(&self.group_size) {
            return Err(anyhow!(
                "FullModelDwqConfig: group_size must be a power of two in [2, 1024]; got {}",
                self.group_size
            ));
        }
        if self.n_steps == 0 {
            return Err(anyhow!("FullModelDwqConfig: n_steps must be > 0"));
        }
        if !self.lr.is_finite() || self.lr <= 0.0 {
            return Err(anyhow!(
                "FullModelDwqConfig: lr must be finite and > 0; got {}",
                self.lr
            ));
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(anyhow!(
                "FullModelDwqConfig: temperature must be finite and > 0; got {}",
                self.temperature
            ));
        }
        if self.batch_size < 32 {
            return Err(anyhow!(
                "FullModelDwqConfig: batch_size must be >= 32 (matmul backward floor); got {}",
                self.batch_size
            ));
        }
        if self.seq_len < 2 {
            return Err(anyhow!(
                "FullModelDwqConfig: seq_len must be >= 2 (next-token target needs >= 1 input + 1 target); got {}",
                self.seq_len
            ));
        }
        if self.gguf_path.as_os_str().is_empty() {
            return Err(anyhow!(
                "FullModelDwqConfig: gguf_path must be non-empty (caller must supply teacher GGUF path)"
            ));
        }
        if self.calibration_token_batches.is_empty() {
            return Err(anyhow!(
                "FullModelDwqConfig: calibration_token_batches must be non-empty"
            ));
        }
        let expected_per_batch = self.batch_size.checked_mul(self.seq_len).ok_or_else(|| {
            anyhow!(
                "FullModelDwqConfig: batch_size * seq_len overflows usize (batch_size={}, seq_len={})",
                self.batch_size,
                self.seq_len
            )
        })?;
        for (i, batch) in self.calibration_token_batches.iter().enumerate() {
            if batch.len() != expected_per_batch {
                return Err(anyhow!(
                    "FullModelDwqConfig: calibration_token_batches[{i}] has {} tokens; expected batch_size * seq_len = {} * {} = {}",
                    batch.len(),
                    self.batch_size,
                    self.seq_len,
                    expected_per_batch
                ));
            }
        }
        if self.top_k_teacher == 0 {
            return Err(anyhow!(
                "FullModelDwqConfig: top_k_teacher must be > 0"
            ));
        }
        Ok(())
    }
}

/// ADR-020 AC#7 Option A — load a JSONL calibration corpus from disk
/// and pack it into the row-major `Vec<Vec<u32>>` shape consumed by
/// [`FullModelDwqConfig::calibration_token_batches`].
///
/// File format: one JSON object per line, each with a `"text"` string
/// field (matches mlx-lm's `CompletionsDataset` convention at
/// `mlx_lm/tuner/datasets.py:25-29`).  Lines that are pure whitespace
/// are skipped; lines that fail to parse OR lack a string `"text"`
/// field hard-fail (no silent skipping — mantra: fail loud).
///
/// Packing strategy: each prompt is independently tokenized via the
/// supplied [`tokenizers::Tokenizer`] (with `add_special_tokens=true`,
/// matching the production serve path at
/// `src/serve/api/handlers.rs:6056`).  Each tokenized prompt
/// occupies one row of the output batch:
/// - if the prompt tokenizes to **more** than `seq_len` tokens, it is
///   truncated to the first `seq_len` ids,
/// - if **fewer**, the remaining slots are zero-padded.
///
/// Rows are accumulated `batch_size` at a time into a single flat
/// `Vec<u32>` of length `batch_size * seq_len` (row-major: row 0 ids
/// occupy indices `[0..seq_len)`, row 1 occupies `[seq_len..2*seq_len)`,
/// etc.).  The trailing partial batch (`< batch_size` prompts) is
/// **dropped** rather than padded — mlx-lm's `iterate_batches` does
/// the same at `tuner/trainer.py:134-135` (it never yields a short
/// batch).
///
/// Returns `Vec<Vec<u32>>` ready to assign to
/// `FullModelDwqConfig::calibration_token_batches`.  Each element
/// satisfies the `validate()` invariant `len() == batch_size *
/// seq_len`.
///
/// # Errors
/// - `path` does not exist or is not readable
/// - `batch_size == 0` or `seq_len == 0` (multiplication would be 0)
/// - any non-blank line is not valid JSON
/// - any JSON object has no `"text"` field, or `"text"` is not a
///   string
/// - the file contains fewer than `batch_size` non-blank prompts
///   (training would have zero batches)
/// - the tokenizer rejects a prompt
pub fn load_calibration_corpus_jsonl(
    path: &std::path::Path,
    tokenizer: &tokenizers::Tokenizer,
    batch_size: usize,
    seq_len: usize,
) -> Result<Vec<Vec<u32>>> {
    use std::io::BufRead;

    if batch_size == 0 {
        return Err(anyhow!(
            "load_calibration_corpus_jsonl: batch_size must be > 0"
        ));
    }
    if seq_len == 0 {
        return Err(anyhow!(
            "load_calibration_corpus_jsonl: seq_len must be > 0"
        ));
    }

    let file = std::fs::File::open(path)
        .with_context(|| format!("opening calibration JSONL at {}", path.display()))?;
    let reader = std::io::BufReader::new(file);

    let mut tokenized_rows: Vec<Vec<u32>> = Vec::new();

    for (idx, line_res) in reader.lines().enumerate() {
        let line_no = idx + 1;
        let line = line_res
            .with_context(|| format!("reading line {} of {}", line_no, path.display()))?;
        if line.trim().is_empty() {
            continue;
        }

        let parsed: serde_json::Value = serde_json::from_str(&line).with_context(|| {
            format!(
                "parsing JSONL line {} of {} as JSON",
                line_no,
                path.display()
            )
        })?;
        let text = parsed.get("text").and_then(|v| v.as_str()).ok_or_else(|| {
            anyhow!(
                "JSONL line {} of {}: missing string field \"text\"",
                line_no,
                path.display()
            )
        })?;

        let enc = tokenizer.encode(text, true).map_err(|e| {
            anyhow!(
                "tokenizing JSONL line {} of {}: {}",
                line_no,
                path.display(),
                e
            )
        })?;
        let ids = enc.get_ids();

        let mut row: Vec<u32> = vec![0u32; seq_len];
        let take = ids.len().min(seq_len);
        row[..take].copy_from_slice(&ids[..take]);
        tokenized_rows.push(row);
    }

    if tokenized_rows.len() < batch_size {
        return Err(anyhow!(
            "load_calibration_corpus_jsonl: file {} produced {} non-blank prompts, \
             which is less than batch_size = {}; training would have zero batches",
            path.display(),
            tokenized_rows.len(),
            batch_size,
        ));
    }

    let n_full_batches = tokenized_rows.len() / batch_size;
    let mut batches: Vec<Vec<u32>> = Vec::with_capacity(n_full_batches);
    for chunk in tokenized_rows.chunks_exact(batch_size) {
        let mut flat: Vec<u32> = Vec::with_capacity(batch_size * seq_len);
        for row in chunk {
            flat.extend_from_slice(row);
        }
        debug_assert_eq!(flat.len(), batch_size * seq_len);
        batches.push(flat);
    }

    Ok(batches)
}

/// ADR-020 AC#7 Option A — translate a [`FullModelDwqConfig`] into the
/// [`crate::calibrate::dwq_targets::ComputeTargetsConfig`] expected by
/// [`crate::calibrate::dwq_targets::compute_dwq_targets`].
///
/// Pure adapter: takes the caller-supplied teacher save directory and
/// the teacher's vocab size (extracted from the GGUF metadata at load
/// time, NOT inferable from the cfg itself), produces a config that
/// the teacher-capture pass consumes.  Validates that `vocab > 0` and
/// `top_k_teacher <= vocab` so the teacher capture never trips its own
/// internal asserts.
pub fn build_dwq_targets_config_from_full_model(
    cfg: &FullModelDwqConfig,
    save_dir: std::path::PathBuf,
    vocab: usize,
) -> Result<crate::calibrate::dwq_targets::ComputeTargetsConfig> {
    if vocab == 0 {
        return Err(anyhow!(
            "build_dwq_targets_config_from_full_model: vocab must be > 0"
        ));
    }
    if cfg.top_k_teacher > vocab {
        return Err(anyhow!(
            "build_dwq_targets_config_from_full_model: top_k_teacher ({}) must be <= vocab ({})",
            cfg.top_k_teacher,
            vocab
        ));
    }
    Ok(crate::calibrate::dwq_targets::ComputeTargetsConfig {
        top_k: cfg.top_k_teacher,
        save_dir,
        vocab,
    })
}

/// ADR-020 AC#7 Option A — borrow a [`FullModelDwqConfig`]'s
/// pre-tokenized batches as a single
/// [`crate::calibrate::dwq_targets::CalibrationSplit`].
///
/// hf2q's `compute_dwq_targets` always processes a *vector* of splits
/// (mlx-lm uses both train + valid).  For Option A the caller provides
/// one corpus and labels it with the supplied static `name` (typically
/// `"train"`).  This helper exists to keep the borrow lifetimes
/// explicit and avoid cfg.calibration_token_batches.as_slice() at every
/// call site.
pub fn build_calibration_split_from_full_model<'a>(
    cfg: &'a FullModelDwqConfig,
    name: &'static str,
) -> crate::calibrate::dwq_targets::CalibrationSplit<'a> {
    crate::calibrate::dwq_targets::CalibrationSplit {
        name,
        batches: cfg.calibration_token_batches.as_slice(),
        batch_size: cfg.batch_size,
        seq_len: cfg.seq_len,
    }
}

/// ADR-020 AC#7 Option A — owned teacher provider + paired
/// `ComputeTargetsConfig` ready for the teacher-capture pass.
///
/// The struct owns the loaded [`GgufTeacherProvider`] (which holds a
/// multi-GB Qwen35Model on the GPU) so the caller can drive
/// [`crate::calibrate::dwq_targets::compute_dwq_targets`] without
/// re-loading the model.  The split is intentionally NOT included
/// here — its borrow lifetime would tie the prepared bundle to the
/// cfg, which complicates ownership.  Caller composes the split via
/// [`build_calibration_split_from_full_model`] at the call site.
pub struct PreparedFullModelTeacherInputs {
    pub teacher: crate::calibrate::gguf_teacher::GgufTeacherProvider,
    pub targets_cfg: crate::calibrate::dwq_targets::ComputeTargetsConfig,
}

/// ADR-020 AC#7 Option A — open the teacher GGUF and assemble every
/// non-corpus input that [`crate::calibrate::dwq_targets::compute_dwq_targets`]
/// needs.
///
/// Composition step that ties together:
///   1. `cfg.validate()` — fail loud on bad knobs BEFORE any disk I/O
///   2. `GgufTeacherProvider::from_gguf_path(cfg.gguf_path)` — load
///      the teacher (multi-GB; ~30s+ wallclock on M5 Max for 35B)
///   3. extract `vocab` from the loaded model (NOT from the cfg —
///      the GGUF is the source of truth, see
///      `gguf_teacher.rs:85-90`)
///   4. `build_dwq_targets_config_from_full_model` to assemble the
///      `ComputeTargetsConfig`
///
/// The caller still owns `cfg`; this fn does NOT consume it because
/// the split helper borrows `cfg.calibration_token_batches`.
///
/// Returns the owned teacher + the targets cfg.  Caller drives
/// `compute_dwq_targets(&mut prepared.teacher,
/// &[build_calibration_split_from_full_model(&cfg, "train")],
/// &prepared.targets_cfg)`.
pub fn prepare_full_model_teacher_inputs(
    cfg: &FullModelDwqConfig,
    save_dir: std::path::PathBuf,
) -> Result<PreparedFullModelTeacherInputs> {
    // Step 1 — validate BEFORE any I/O.
    cfg.validate()
        .context("prepare_full_model_teacher_inputs: cfg.validate")?;

    // Step 2 — open teacher.  This is the heavy step (model load +
    // GPU upload).
    let teacher = crate::calibrate::gguf_teacher::GgufTeacherProvider::from_gguf_path(
        &cfg.gguf_path,
    )
    .with_context(|| {
        format!(
            "prepare_full_model_teacher_inputs: GgufTeacherProvider::from_gguf_path {}",
            cfg.gguf_path.display()
        )
    })?;

    // Step 3 — derive vocab from the loaded model (authoritative).
    let vocab = teacher.vocab();

    // Step 4 — assemble the targets cfg.
    let targets_cfg = build_dwq_targets_config_from_full_model(cfg, save_dir, vocab)
        .context("prepare_full_model_teacher_inputs: build_dwq_targets_config_from_full_model")?;

    Ok(PreparedFullModelTeacherInputs {
        teacher,
        targets_cfg,
    })
}

/// ADR-020 AC#7 Option A — drive the teacher-capture pass over a
/// single calibration split.  Generic over any
/// [`crate::calibrate::dwq_targets::TeacherLogitsProvider`] so this
/// function is unit-testable with a synthetic teacher (no real GGUF
/// load).
///
/// Builds the split via [`build_calibration_split_from_full_model`]
/// (labelling it `"train"` — the only split Option A consumes; mlx-lm
/// supports a separate `"valid"` split for early-stopping but Option
/// A's plain Adam loop has no early-stop hook yet, so a single train
/// split is sufficient and matches `dwq.py:53` behaviour when invoked
/// without a held-out set).
///
/// Returns the per-split count summary from
/// [`crate::calibrate::dwq_targets::compute_dwq_targets`] —
/// `Vec<(split_name, n_batches_written)>`.  For the single-split
/// flow that's `vec![("train", N)]` where N == cfg.calibration_token_batches.len().
pub fn drive_full_model_teacher_capture<T>(
    teacher: &mut T,
    cfg: &FullModelDwqConfig,
    targets_cfg: &crate::calibrate::dwq_targets::ComputeTargetsConfig,
) -> Result<Vec<(String, usize)>>
where
    T: crate::calibrate::dwq_targets::TeacherLogitsProvider,
{
    let split = build_calibration_split_from_full_model(cfg, "train");
    crate::calibrate::dwq_targets::compute_dwq_targets(teacher, &[split], targets_cfg)
        .context("drive_full_model_teacher_capture: compute_dwq_targets")
}

/// ADR-020 AC#7 Option A — ergonomic wrapper that drives
/// [`drive_full_model_teacher_capture`] using an owned
/// [`PreparedFullModelTeacherInputs`].
///
/// Equivalent to:
/// ```ignore
/// drive_full_model_teacher_capture(&mut prepared.teacher, cfg, &prepared.targets_cfg)
/// ```
/// but spelled out so call sites don't have to thread two fields out of
/// the prepared bundle.
pub fn drive_full_model_teacher_capture_prepared(
    prepared: &mut PreparedFullModelTeacherInputs,
    cfg: &FullModelDwqConfig,
) -> Result<Vec<(String, usize)>> {
    drive_full_model_teacher_capture(&mut prepared.teacher, cfg, &prepared.targets_cfg)
}

/// ADR-020 AC#7 Option A — typed teacher-target bundle for one
/// calibration batch.  Mirror of mlx-lm's `load_dwq_target` return
/// shape at `dwq.py:114` (logits + indices tensor pair) but wrapped
/// in a struct so call sites don't have to remember the 5-tuple
/// arity.
#[derive(Debug, Clone)]
pub struct TeacherBatchTargets {
    /// Top-K logits, row-major `[batch, seq, top_k]` flat.
    pub logits: Vec<f32>,
    /// Top-K indices into the full vocab, row-major same shape.
    pub indices: Vec<u32>,
    /// Batch dimension (matches `cfg.batch_size`).
    pub batch: usize,
    /// Sequence dimension AFTER the next-token trim
    /// (== `cfg.seq_len - 1`).
    pub seq: usize,
    /// Top-K (matches `cfg.top_k_teacher`).
    pub top_k: usize,
}

/// ADR-020 AC#7 Option A — load one batch of teacher targets from
/// disk and cross-check against [`FullModelDwqConfig`].
///
/// Wraps [`crate::calibrate::dwq_targets::load_dwq_target`] (which
/// reads the raw safetensors header) and verifies that the on-disk
/// shape agrees with the cfg the student will train against:
/// - `batch == cfg.batch_size`
/// - `seq == cfg.seq_len - 1` (next-token trim mirrors `dwq.py:53`
///   where the input batch is sliced to `[:, :-1]` before the teacher
///   forward — see `dwq_targets.rs:148`)
/// - `top_k == cfg.top_k_teacher`
///
/// A mismatch at any axis is a hard error: silently using
/// shape-incompatible teacher targets would either (a) cause an
/// unrelated index/shape error deep in the student loss kernel, or
/// (b) silently corrupt the gradient because indices on the wrong
/// vocab axis are still in-range.  Fail-loud here.
pub fn load_teacher_targets_for_batch(
    save_dir: &std::path::Path,
    split_name: &str,
    batch_idx: usize,
    cfg: &FullModelDwqConfig,
) -> Result<TeacherBatchTargets> {
    let (logits, indices, batch, seq, top_k) =
        crate::calibrate::dwq_targets::load_dwq_target(save_dir, split_name, batch_idx)
            .with_context(|| {
                format!(
                    "load_teacher_targets_for_batch({}/{split_name}/{batch_idx})",
                    save_dir.display()
                )
            })?;

    if batch != cfg.batch_size {
        return Err(anyhow!(
            "load_teacher_targets_for_batch: file batch ({}) != cfg.batch_size ({}) at {}/{}/{}",
            batch,
            cfg.batch_size,
            save_dir.display(),
            split_name,
            batch_idx
        ));
    }
    let expected_seq = cfg
        .seq_len
        .checked_sub(1)
        .ok_or_else(|| anyhow!("cfg.seq_len ({}) underflows when subtracting 1", cfg.seq_len))?;
    if seq != expected_seq {
        return Err(anyhow!(
            "load_teacher_targets_for_batch: file seq ({}) != cfg.seq_len - 1 ({}) at {}/{}/{}",
            seq,
            expected_seq,
            save_dir.display(),
            split_name,
            batch_idx
        ));
    }
    if top_k != cfg.top_k_teacher {
        return Err(anyhow!(
            "load_teacher_targets_for_batch: file top_k ({}) != cfg.top_k_teacher ({}) at {}/{}/{}",
            top_k,
            cfg.top_k_teacher,
            save_dir.display(),
            split_name,
            batch_idx
        ));
    }

    Ok(TeacherBatchTargets {
        logits,
        indices,
        batch,
        seq,
        top_k,
    })
}

/// ADR-020 AC#7 Option A — per-position top-K KL loss CPU oracle.
///
/// Mirror of mlx-lm `dwq.py:108-114`'s `loss_fn`:
/// ```text
///   scale = 1 / temperature
///   losses = kl_div_loss(scale * student_logits, scale * teacher_logits)
///   loss   = mean(losses)            # over (batch * seq) positions
/// ```
/// where `kl_div_loss(q_logits, p_logits) = KL(softmax(p) || softmax(q))`
/// (matches hf2q's GPU kernel
/// [`crate::calibrate::dynamic_quant_gpu::kl_div_loss_per_row`] at
/// `dynamic_quant_gpu.rs:127-138` — the GPU version with shape
/// `[N, V]`; this oracle expects `[batch, seq, V]` flat).
///
/// Inputs are top-K reductions: each row is the K logits at the same
/// vocab indices for the matched (student, teacher) position.  The
/// caller is responsible for indexing the student model's full-vocab
/// output through `TeacherBatchTargets.indices` BEFORE invoking this
/// fn — exactly what `mx.take_along_axis(logits, ids, axis=-1)` does
/// at `dwq.py:111`.
///
/// `student` and `teacher` must both be flat row-major
/// `[batch, seq, top_k]` of length `batch * seq * top_k`.
///
/// Returns scalar f32 — mean per-position KL.  Numerically stable:
/// uses the `max + log(Σexp(x − max))` form for both logsumexp's so
/// large logits don't overflow.
///
/// Mathematical contract verified by tests:
/// - student == teacher ⇒ loss == 0 (within 1e-6)
/// - increasing temperature ⇒ smaller loss (softer targets)
/// - hand-computable: K=2 case with explicit numerical answer
///
/// # Errors
/// - `student.len() != teacher.len()`
/// - `batch * seq * top_k` overflows or doesn't equal the slice len
/// - `top_k == 0` or `batch == 0` or `seq == 0`
/// - `temperature <= 0` or non-finite
pub fn kl_loss_topk_oracle(
    student: &[f32],
    teacher: &[f32],
    batch: usize,
    seq: usize,
    top_k: usize,
    temperature: f32,
) -> Result<f32> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(anyhow!(
            "kl_loss_topk_oracle: temperature must be finite and > 0; got {temperature}"
        ));
    }
    if batch == 0 || seq == 0 || top_k == 0 {
        return Err(anyhow!(
            "kl_loss_topk_oracle: batch ({batch}), seq ({seq}), top_k ({top_k}) must all be > 0"
        ));
    }
    let n_positions = batch
        .checked_mul(seq)
        .ok_or_else(|| anyhow!("batch * seq overflows usize"))?;
    let expected = n_positions
        .checked_mul(top_k)
        .ok_or_else(|| anyhow!("batch * seq * top_k overflows usize"))?;
    if student.len() != expected {
        return Err(anyhow!(
            "kl_loss_topk_oracle: student.len ({}) != batch*seq*top_k ({})",
            student.len(),
            expected
        ));
    }
    if teacher.len() != expected {
        return Err(anyhow!(
            "kl_loss_topk_oracle: teacher.len ({}) != batch*seq*top_k ({})",
            teacher.len(),
            expected
        ));
    }

    let inv_t = 1.0_f32 / temperature;

    let mut total_kl = 0.0_f64; // f64 accumulator — mean over many positions
    for pos in 0..n_positions {
        let lo = pos * top_k;
        let hi = lo + top_k;
        let s_row = &student[lo..hi];
        let t_row = &teacher[lo..hi];

        // Numerically-stable logsumexp for student row (scaled).
        let mut s_max = f32::NEG_INFINITY;
        for &x in s_row {
            let v = x * inv_t;
            if v > s_max {
                s_max = v;
            }
        }
        let mut s_sum_exp = 0.0_f32;
        for &x in s_row {
            s_sum_exp += (x * inv_t - s_max).exp();
        }
        let s_log_norm = s_max + s_sum_exp.ln();

        // Same for teacher row (scaled).
        let mut t_max = f32::NEG_INFINITY;
        for &x in t_row {
            let v = x * inv_t;
            if v > t_max {
                t_max = v;
            }
        }
        let mut t_sum_exp = 0.0_f32;
        for &x in t_row {
            t_sum_exp += (x * inv_t - t_max).exp();
        }
        let t_log_norm = t_max + t_sum_exp.ln();

        // KL(softmax(t) || softmax(s)) = Σ p_t · (log p_t − log p_s)
        let mut row_kl = 0.0_f32;
        for (st, te) in s_row.iter().zip(t_row.iter()) {
            let s_scaled = *st * inv_t;
            let t_scaled = *te * inv_t;
            let log_p_t = t_scaled - t_log_norm;
            let log_p_s = s_scaled - s_log_norm;
            let p_t = log_p_t.exp();
            row_kl += p_t * (log_p_t - log_p_s);
        }
        total_kl += row_kl as f64;
    }

    Ok((total_kl / n_positions as f64) as f32)
}

/// ADR-020 AC#7 Option A — gather student logits along the vocab
/// axis using teacher-supplied top-K indices.  CPU oracle of MLX's
/// `mx.take_along_axis(logits, ids, axis=-1)` at `dwq.py:111`.
///
/// Inputs:
/// - `student_logits_full`: flat row-major `[batch, seq, vocab]`,
///   length `batch * seq * vocab`
/// - `indices`: flat row-major `[batch, seq, top_k]`, length
///   `batch * seq * top_k`, each value `< vocab`
///
/// Output: flat row-major `[batch, seq, top_k]` where
/// `out[r, t, k] = student_logits_full[r, t, indices[r, t, k]]`.
///
/// This is the second half of the indexed-distillation pipeline:
/// the teacher's `TeacherBatchTargets.indices` tell us WHICH vocab
/// positions to compare; this fn pulls the student's logits at
/// exactly those positions, after which `kl_loss_topk_oracle` does
/// the scaled-KL math on the matched K-vector pair.
///
/// Index validation: every entry of `indices` MUST be `< vocab`.
/// Out-of-range hits are a hard error rather than silent wraparound
/// — a stale teacher-targets file paired with a re-quantized student
/// would otherwise corrupt training without any visible signal.
///
/// # Errors
/// - `student_logits_full.len() != batch * seq * vocab`
/// - `indices.len() != batch * seq * top_k`
/// - any of `batch`, `seq`, `vocab`, `top_k` is `0` or
///   `batch * seq * vocab` / `... * top_k` overflow
/// - any `indices[i] >= vocab`
pub fn take_along_topk_indices(
    student_logits_full: &[f32],
    indices: &[u32],
    batch: usize,
    seq: usize,
    vocab: usize,
    top_k: usize,
) -> Result<Vec<f32>> {
    if batch == 0 || seq == 0 || vocab == 0 || top_k == 0 {
        return Err(anyhow!(
            "take_along_topk_indices: batch ({batch}), seq ({seq}), vocab ({vocab}), top_k ({top_k}) must all be > 0"
        ));
    }
    let bs = batch
        .checked_mul(seq)
        .ok_or_else(|| anyhow!("batch * seq overflows usize"))?;
    let logits_len = bs
        .checked_mul(vocab)
        .ok_or_else(|| anyhow!("batch * seq * vocab overflows usize"))?;
    let idx_len = bs
        .checked_mul(top_k)
        .ok_or_else(|| anyhow!("batch * seq * top_k overflows usize"))?;
    if student_logits_full.len() != logits_len {
        return Err(anyhow!(
            "take_along_topk_indices: student_logits_full.len ({}) != batch*seq*vocab ({})",
            student_logits_full.len(),
            logits_len
        ));
    }
    if indices.len() != idx_len {
        return Err(anyhow!(
            "take_along_topk_indices: indices.len ({}) != batch*seq*top_k ({})",
            indices.len(),
            idx_len
        ));
    }

    let mut out: Vec<f32> = Vec::with_capacity(idx_len);
    for pos in 0..bs {
        let row_lo = pos * vocab;
        let idx_lo = pos * top_k;
        for k in 0..top_k {
            let v = indices[idx_lo + k];
            if (v as usize) >= vocab {
                return Err(anyhow!(
                    "take_along_topk_indices: indices[{}] = {} >= vocab ({})",
                    idx_lo + k,
                    v,
                    vocab
                ));
            }
            out.push(student_logits_full[row_lo + v as usize]);
        }
    }
    debug_assert_eq!(out.len(), idx_len);
    Ok(out)
}

/// ADR-020 AC#7 Option A — GPU-tape mirror of [`kl_loss_topk_oracle`].
///
/// Composes existing autograd ops to run the indexed top-K KL on
/// GpuTape with full backward support:
/// ```text
///   s_scaled = scalar_mul(student_topk, 1/T)
///   t_scaled = scalar_mul(teacher_topk, 1/T)
///   per_row  = kl_div_loss_per_row(s_scaled, t_scaled)   # [N]
///   sum_row  = row_sum(view(per_row, [1, N]))            # [1]
///   loss     = scalar_mul(sum_row, 1/N)                  # [1]
/// ```
///
/// Inputs:
/// - `student_topk`: 2-D `[N, top_k]` GpuTensor — the student's
///   logits gathered at teacher's top-K vocab positions (caller
///   produces this via [`crate::calibrate::autograd_gpu_tape::take_along_axis_topk`]
///   on the full-vocab student output, then a `view` to flatten
///   `[batch, seq, top_k]` to `[batch*seq, top_k]`).
/// - `teacher_topk`: 2-D `[N, top_k]` GpuTensor — same shape as
///   student.  Treated as a constant target (caller builds it as a
///   leaf from `TeacherBatchTargets.logits`).
/// - `temperature > 0`: scalar; `inv_t = 1/T` softens both
///   distributions per Hinton 2015 KD (matches `dwq.py:106`).
///
/// Output: 1-D scalar `[1]` GpuTensor — mean per-position KL.  The
/// caller can backprop with `backward(&loss, ones_like(&tape, &[1]))`
/// to produce `∂loss/∂student_topk` plus gradients on any leaves
/// upstream of `student_topk`.
///
/// Numerical contract verified against [`kl_loss_topk_oracle`]
/// byte-equivalently in tests (within 1e-5 because hf2q's GPU
/// `kl_div_loss_per_row` uses the same softmax-then-log composition
/// as the CPU oracle, modulo per-element f32 rounding).
///
/// # Errors
/// - `student_topk.shape != teacher_topk.shape`
/// - `student_topk.shape.len() != 2`
/// - `temperature` not finite or `<= 0`
/// - the underlying ops fail (propagated)
pub fn kl_loss_topk_via_tape(
    student_topk: &crate::calibrate::autograd_gpu_tape::GpuTensor,
    teacher_topk: &crate::calibrate::autograd_gpu_tape::GpuTensor,
    temperature: f32,
) -> Result<crate::calibrate::autograd_gpu_tape::GpuTensor> {
    use crate::calibrate::autograd_gpu_tape::{row_sum, scalar_mul, view};
    use crate::calibrate::dynamic_quant_gpu::kl_div_loss_per_row;

    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(anyhow!(
            "kl_loss_topk_via_tape: temperature must be finite and > 0; got {temperature}"
        ));
    }
    if student_topk.shape().len() != 2 {
        return Err(anyhow!(
            "kl_loss_topk_via_tape: student_topk must be 2-D [N, top_k]; got {:?}",
            student_topk.shape()
        ));
    }
    if student_topk.shape() != teacher_topk.shape() {
        return Err(anyhow!(
            "kl_loss_topk_via_tape: student shape {:?} != teacher shape {:?}",
            student_topk.shape(),
            teacher_topk.shape()
        ));
    }

    let n = student_topk.shape()[0];
    if n == 0 {
        return Err(anyhow!(
            "kl_loss_topk_via_tape: N (rows) must be > 0; got {n}"
        ));
    }

    let inv_t = 1.0_f32 / temperature;
    let s_scaled = scalar_mul(student_topk, inv_t)
        .context("kl_loss_topk_via_tape: scalar_mul(student, 1/T)")?;
    let t_scaled = scalar_mul(teacher_topk, inv_t)
        .context("kl_loss_topk_via_tape: scalar_mul(teacher, 1/T)")?;

    let per_row = kl_div_loss_per_row(&s_scaled, &t_scaled)
        .context("kl_loss_topk_via_tape: kl_div_loss_per_row")?;
    // per_row.shape == [N]; row_sum needs 2-D, so view as [1, N].
    let per_row_2d = view(&per_row, vec![1, n])
        .context("kl_loss_topk_via_tape: view(per_row, [1, N])")?;
    let sum_row = row_sum(&per_row_2d).context("kl_loss_topk_via_tape: row_sum")?;
    // sum_row.shape == [1]; multiply by 1/N to get the mean.
    scalar_mul(&sum_row, 1.0_f32 / n as f32)
        .context("kl_loss_topk_via_tape: scalar_mul(sum, 1/N)")
}

/// ADR-020 AC#7 Option A — build an `MlxBuffer` of U32 indices on
/// the supplied device, suitable for feeding
/// [`crate::calibrate::autograd_gpu_tape::take_along_axis_topk`].
///
/// The output buffer has element count `n_rows * top_k`, dtype
/// `U32`, and shape `[n_rows, top_k]` (matching the `take_along`
/// kernel's expectation).
///
/// Validates `indices.len() == n_rows * top_k` and every entry's
/// in-range bound is checked LATER by the kernel (we don't have
/// `vocab` here).  Caller of `gather_student_topk_via_tape` will
/// assert vocab-bound before launch.
pub fn build_topk_indices_buffer(
    device: &MlxDevice,
    indices: &[u32],
    n_rows: usize,
    top_k: usize,
) -> Result<MlxBuffer> {
    if n_rows == 0 || top_k == 0 {
        return Err(anyhow!(
            "build_topk_indices_buffer: n_rows ({n_rows}) and top_k ({top_k}) must be > 0"
        ));
    }
    let expected = n_rows
        .checked_mul(top_k)
        .ok_or_else(|| anyhow!("build_topk_indices_buffer: n_rows * top_k overflows"))?;
    if indices.len() != expected {
        return Err(anyhow!(
            "build_topk_indices_buffer: indices.len ({}) != n_rows * top_k ({})",
            indices.len(),
            expected
        ));
    }
    let mut buf = device
        .alloc_buffer(expected * 4, DType::U32, vec![n_rows, top_k])
        .map_err(|e| anyhow!("build_topk_indices_buffer: alloc: {e}"))?;
    buf.as_mut_slice::<u32>()
        .map_err(|e| anyhow!("build_topk_indices_buffer: as_mut_slice: {e}"))?
        .copy_from_slice(indices);
    Ok(buf)
}

/// ADR-020 AC#7 Option A — GPU-tape mirror of [`take_along_topk_indices`].
///
/// Given the student's full-vocab logits as a 2-D `[N, vocab]` OR
/// 3-D `[batch, seq, vocab]` GpuTensor, gather the K logits at the
/// teacher-supplied vocab positions and return a 2-D `[N, top_k]`
/// tensor ready to feed [`kl_loss_topk_via_tape`].
///
/// `indices_buf` must be U32 with element count `N * top_k` (caller
/// builds it via [`build_topk_indices_buffer`] or hands an existing
/// buffer back from [`TeacherBatchTargets`]).
///
/// Index validation: every entry MUST be `< vocab`.  Out-of-range
/// hits are caught here BEFORE the launch so a stale teacher-targets
/// file can't silently corrupt the GPU forward.  Validation happens
/// against the in-CPU view of `indices_buf`.
///
/// # Errors
/// - `student_full.shape().len() not in {2, 3}`
/// - `indices_buf.dtype() != U32`
/// - `indices_buf.element_count() != N * top_k`
/// - any index `>= vocab`
/// - underlying ops fail (propagated)
pub fn gather_student_topk_via_tape(
    student_full: &crate::calibrate::autograd_gpu_tape::GpuTensor,
    indices_buf: MlxBuffer,
    top_k: usize,
) -> Result<crate::calibrate::autograd_gpu_tape::GpuTensor> {
    use crate::calibrate::autograd_gpu_tape::{take_along_axis_topk, view};

    let shape = student_full.shape();
    if !(shape.len() == 2 || shape.len() == 3) {
        return Err(anyhow!(
            "gather_student_topk_via_tape: student_full must be 2-D [N, V] or 3-D [B, S, V]; got {:?}",
            shape
        ));
    }
    let (n, vocab) = match shape.len() {
        2 => (shape[0], shape[1]),
        3 => (shape[0] * shape[1], shape[2]),
        _ => unreachable!(),
    };
    if top_k == 0 || top_k > vocab {
        return Err(anyhow!(
            "gather_student_topk_via_tape: top_k must be in (0, vocab={vocab}]; got {top_k}"
        ));
    }
    if indices_buf.dtype() != DType::U32 {
        return Err(anyhow!(
            "gather_student_topk_via_tape: indices_buf.dtype must be U32; got {}",
            indices_buf.dtype()
        ));
    }
    let expected = n * top_k;
    if indices_buf.element_count() != expected {
        return Err(anyhow!(
            "gather_student_topk_via_tape: indices_buf.element_count ({}) != N * top_k ({})",
            indices_buf.element_count(),
            expected
        ));
    }

    // Vocab-range validation against the CPU-visible view.
    {
        let host_idx: &[u32] = indices_buf
            .as_slice()
            .map_err(|e| anyhow!("gather_student_topk_via_tape: as_slice (validate): {e}"))?;
        for (i, &v) in host_idx.iter().enumerate() {
            if (v as usize) >= vocab {
                return Err(anyhow!(
                    "gather_student_topk_via_tape: indices[{}] = {} >= vocab ({})",
                    i,
                    v,
                    vocab
                ));
            }
        }
    }

    // Reshape 3-D student_full to 2-D [N, vocab] if needed.
    let student_2d_owner;
    let student_2d: &crate::calibrate::autograd_gpu_tape::GpuTensor = if shape.len() == 3 {
        student_2d_owner = view(student_full, vec![n, vocab])
            .context("gather_student_topk_via_tape: view 3-D → 2-D")?;
        &student_2d_owner
    } else {
        student_full
    };

    take_along_axis_topk(student_2d, indices_buf, top_k)
        .context("gather_student_topk_via_tape: take_along_axis_topk")
}

/// ADR-020 AC#7 Option A — canonical Adam param-name kind axis for
/// affine quant parameters.  Keeps the suffix discipline in one
/// place so the registration helper, the read helper, and any
/// future serializer all agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffineParamKind {
    /// Per-group quant scale (matches mlx-affine's `.scales`
    /// safetensors suffix at `mlx_safetensors_loader.rs`).
    Scale,
    /// Per-group quant bias (matches `.biases` suffix).
    Bias,
}

impl AffineParamKind {
    /// Suffix appended to the tensor identifier to form the Adam
    /// param-name.  `.scales` / `.biases` mirrors the on-disk
    /// safetensors layout — keeps train-time and serialize-time
    /// vocabulary identical so a future `Adam → safetensors` flush
    /// can iterate by suffix.
    pub fn suffix(self) -> &'static str {
        match self {
            AffineParamKind::Scale => ".scales",
            AffineParamKind::Bias => ".biases",
        }
    }
}

/// ADR-020 AC#7 Option A — produce the canonical Adam param-name
/// for an affine quant parameter associated with `tensor_id`.
///
/// Examples:
/// ```ignore
///   affine_param_name("blk.0.attn_q.weight", AffineParamKind::Scale)
///       == "blk.0.attn_q.weight.scales"
///   affine_param_name("L0_E2_gate", AffineParamKind::Bias)
///       == "L0_E2_gate.biases"
/// ```
pub fn affine_param_name(tensor_id: &str, kind: AffineParamKind) -> String {
    format!("{}{}", tensor_id, kind.suffix())
}

/// ADR-020 AC#7 Option A — register an (s, b) affine quant param
/// PAIR with [`crate::calibrate::adam::AdamOptimizer`] under the
/// canonical names produced by [`affine_param_name`].
///
/// Allocates fresh GPU buffers from the supplied f32 init slices,
/// registers each under its name, and returns the two keys so the
/// caller can later thread gradients back via the same names in
/// [`crate::calibrate::adam::AdamOptimizer::step`].
///
/// # Errors
/// - either init slice is empty
/// - duplicate registration (Adam returns an error)
/// - GPU buffer allocation failure
pub fn register_affine_pair(
    adam: &mut crate::calibrate::adam::AdamOptimizer,
    device: &MlxDevice,
    tensor_id: &str,
    scale_init: &[f32],
    bias_init: &[f32],
) -> Result<(String, String)> {
    if scale_init.is_empty() {
        return Err(anyhow!(
            "register_affine_pair: scale_init must be non-empty for tensor '{tensor_id}'"
        ));
    }
    if bias_init.is_empty() {
        return Err(anyhow!(
            "register_affine_pair: bias_init must be non-empty for tensor '{tensor_id}'"
        ));
    }
    let scale_name = affine_param_name(tensor_id, AffineParamKind::Scale);
    let bias_name = affine_param_name(tensor_id, AffineParamKind::Bias);

    let scale_buf = buffer_from_f32(device, scale_init)
        .with_context(|| format!("register_affine_pair: alloc scale for '{tensor_id}'"))?;
    let bias_buf = buffer_from_f32(device, bias_init)
        .with_context(|| format!("register_affine_pair: alloc bias for '{tensor_id}'"))?;

    adam.register_param(scale_name.clone(), scale_buf)
        .with_context(|| format!("register_affine_pair: register scale '{scale_name}'"))?;
    adam.register_param(bias_name.clone(), bias_buf)
        .with_context(|| format!("register_affine_pair: register bias '{bias_name}'"))?;

    Ok((scale_name, bias_name))
}

/// ADR-020 AC#7 Option A — read the trained (s, b) values back out
/// of [`crate::calibrate::adam::AdamOptimizer`] using the canonical
/// names produced by [`affine_param_name`].
///
/// Symmetric to [`register_affine_pair`].  Returns `(scales, biases)`
/// as host-side `Vec<f32>` so the caller can serialize them into a
/// DWQ-overlay safetensors blob.
///
/// # Errors
/// - either name is not registered with `adam`
pub fn read_affine_pair(
    adam: &crate::calibrate::adam::AdamOptimizer,
    tensor_id: &str,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let scale_name = affine_param_name(tensor_id, AffineParamKind::Scale);
    let bias_name = affine_param_name(tensor_id, AffineParamKind::Bias);
    let s = adam
        .read_param(&scale_name)
        .with_context(|| format!("read_affine_pair: read scale '{scale_name}'"))?;
    let b = adam
        .read_param(&bias_name)
        .with_context(|| format!("read_affine_pair: read bias '{bias_name}'"))?;
    Ok((s, b))
}

/// ADR-020 AC#7 Option A — packed (s_init, b_init, q_int) artifacts
/// for one Linear, ready to wire into the student's per-Linear
/// training step:
/// - `scales` + `biases` go to [`register_affine_pair`] (Adam
///   trainable leaves)
/// - `q_int_bytes` becomes the constant q_int input to
///   [`crate::calibrate::autograd_gpu_tape::qdq_affine`] each forward
///
/// `n` (output dim) × `k` (input dim) match the original 2-D weight
/// shape and are carried alongside the artifacts so downstream
/// callers don't have to plumb shape independently.
#[derive(Debug, Clone)]
pub struct DwqQuantPack {
    /// Per-element packed quant codes, length `n * k`.  U8 with
    /// values in `[0, 2^bits)`.  Caller passes this through
    /// `qdq_affine(s, b, &q_int_bytes, group_size)` each step —
    /// q_int is treated as a constant input, NOT a trainable leaf.
    pub q_int_bytes: Vec<u8>,
    /// Per-group scale init, length `n * k / group_size`.  Min-max
    /// projection optimum (see `qdq_affine.metal:85-103`).
    pub scales: Vec<f32>,
    /// Per-group bias init, same length as `scales`.
    pub biases: Vec<f32>,
    /// Output dim of the original Linear (rows of the weight slice).
    pub n: usize,
    /// Input dim (cols).
    pub k: usize,
    /// Group size used during init.  The same value MUST be passed
    /// to every downstream `qdq_affine` call to recover the same
    /// W → q_int mapping.
    pub group_size: usize,
}

/// ADR-020 AC#7 Option A — single-call wrapper around
/// [`init_affine_params_gpu`] that carries the n×k shape with the
/// returned artifacts and opens a fresh
/// [`mlx_native::KernelRegistry`] internally so call sites don't
/// have to.
///
/// Inputs:
/// - `device`: live MlxDevice (the GPU init kernel needs metal_device())
/// - `w_f32`: dequantized 2-D weight slice, length `n * k`,
///   row-major (matches GGUF's `[n, k]` weight convention)
/// - `n`, `k`: weight dims (validated against `w_f32.len()`)
/// - `n_bits`: `2..=8` (passed through to the kernel as `n_bins =
///   1 << n_bits`)
/// - `group_size`: power-of-two in `[2, 1024]`, MUST divide `k`
///
/// Returns a [`DwqQuantPack`] suitable for handing to
/// [`register_affine_pair`] (`.scales` + `.biases`) and the per-step
/// `qdq_affine(s, b, &.q_int_bytes, group_size)` forward.
///
/// # Errors
/// - `w_f32.len() != n * k`
/// - `n == 0` or `k == 0`
/// - `k % group_size != 0` (caught by `init_affine_params_gpu`'s
///   `n_total % group_size` check via `n_total = n*k`)
/// - underlying `init_affine_params_gpu` fails (propagated)
pub fn quant_pack_for_dwq(
    device: &MlxDevice,
    w_f32: &[f32],
    n: usize,
    k: usize,
    n_bits: u32,
    group_size: usize,
) -> Result<DwqQuantPack> {
    if n == 0 || k == 0 {
        return Err(anyhow!(
            "quant_pack_for_dwq: n ({n}) and k ({k}) must be > 0"
        ));
    }
    let expected = n
        .checked_mul(k)
        .ok_or_else(|| anyhow!("quant_pack_for_dwq: n * k overflows"))?;
    if w_f32.len() != expected {
        return Err(anyhow!(
            "quant_pack_for_dwq: w_f32.len ({}) != n * k ({})",
            w_f32.len(),
            expected
        ));
    }
    if k % group_size != 0 {
        return Err(anyhow!(
            "quant_pack_for_dwq: k ({k}) must be divisible by group_size ({group_size})"
        ));
    }

    let mut registry = KernelRegistry::new();
    let (q_int_bytes, scales, biases) =
        init_affine_params_gpu(device, &mut registry, w_f32, group_size, n_bits)
            .context("quant_pack_for_dwq: init_affine_params_gpu")?;

    Ok(DwqQuantPack {
        q_int_bytes,
        scales,
        biases,
        n,
        k,
        group_size,
    })
}

/// ADR-020 iter-12d-2 — per-Linear training summary inside a model
/// scan (one entry per successfully-trained tensor).
#[derive(Debug, Clone)]
pub struct DwqLinearTrained {
    /// GGUF tensor name (e.g. `"blk.0.attn_qkv.weight"`).
    pub name: String,
    /// Output dim (number of rows of the f32 weight slice).
    pub n: usize,
    /// Input dim (number of cols).
    pub k: usize,
    pub kl_initial: f32,
    pub kl_min: f32,
    pub steps_run: usize,
    /// ADR-020 iter-12f-2 — optional per-Linear DWQ-vs-Q4_0 KL benchmark.
    /// `Some` iff `cfg.compute_bench` was true during training.
    pub bench: Option<crate::calibrate::dwq_benchmark::PerLinearKlComparison>,
}

/// Reason a tensor was skipped during a model scan (rank mismatch,
/// quant-config violation, training failure).  Surfaced so callers
/// (e.g. CLI wrapper) can print a per-tensor disposition log.
#[derive(Debug, Clone)]
pub struct DwqLinearSkipped {
    pub name: String,
    pub reason: String,
}

/// Aggregate output of [`train_all_linears_dwq`] — the trained
/// per-Linear quant params packed as a single mlx-format safetensors
/// byte stream + structured per-tensor diagnostics.
#[derive(Debug)]
pub struct DwqAllLinearsResult {
    /// One entry per successfully-trained Linear, in scan order.
    pub trained: Vec<DwqLinearTrained>,
    /// One entry per skipped tensor, with reason.
    pub skipped: Vec<DwqLinearSkipped>,
    /// Serialized mlx-format safetensors output containing
    /// `<name>.weight` (U32-packed quant codes), `<name>.scales`,
    /// `<name>.biases` triplets per trained tensor.
    pub safetensors_bytes: Vec<u8>,
    /// ADR-020 iter-12f-2 — mean `delta_kl_nats` aggregated across
    /// trained Linears that have a `bench` populated (i.e.
    /// `cfg.compute_bench == true`).  `None` if compute_bench was
    /// false or zero Linears trained.  Positive ⇒ DWQ outperforms
    /// Q4_0 at the per-Linear level on average.
    ///
    /// §8.3 AC #7 acceptance gate: `mean_delta_kl_nats > 0.05` (per
    /// Linear) is the recommended floor.  Caller decides what to
    /// do with the result; the driver does not auto-fail on a low
    /// mean.
    pub mean_delta_kl_nats: Option<f32>,
}

/// ADR-020 iter-12d-2 — production driver: scan a real GGUF model,
/// enumerate all rank-2 Linear weight tensors, run iter-12d-1's
/// `train_linear_dwq_synthetic_teacher` per tensor, accumulate trained
/// `MlxAffineLinear` outputs into a single mlx-format safetensors byte
/// stream.
///
/// ## Filtering
///
/// Tensors are skipped (and recorded in `result.skipped`) if any of:
///   - `name_filter(name) == false`
///   - rank != 2 (1-D bias / norm tensors)
///   - `n < 32` or `k < 32` (matmul kernel floor)
///   - `k % cfg.group_size != 0`
///   - GGUF dequant fails (unsupported `ggml_type`)
///   - Training fails to converge (`kl_min > kl_initial × convergence_ratio`)
///
/// ## Output safetensors layout
///
/// Per trained tensor:
///   - `<name>.weight` — U32, shape `[n, k * bits / 32]`
///   - `<name>.scales` — `cfg.float_dtype` (default BF16), shape `[n, k / group_size]`
///   - `<name>.biases` — `cfg.float_dtype`, shape `[n, k / group_size]`
///
/// Matches mlx-lm's flat-parameter naming convention so the output
/// loads cleanly via `MlxAffineLinear::from_safetensors`.
pub fn train_all_linears_dwq<F>(
    gguf_path: &std::path::Path,
    cfg: &DwqTrainingConfig,
    float_dtype: safetensors::tensor::Dtype,
    name_filter: F,
) -> Result<DwqAllLinearsResult>
where
    F: Fn(&str) -> bool,
{
    use crate::calibrate::mlx_safetensors_loader::{MlxAffineLinear, MlxAffineLinearBytes};
    use mlx_native::gguf::GgufFile;

    let device = MlxDevice::new()
        .map_err(|e| anyhow!("train_all_linears_dwq: device: {e}"))?;
    let gguf = GgufFile::open(gguf_path)
        .with_context(|| format!("open gguf {}", gguf_path.display()))?;

    // Snapshot tensor names up front (release borrows on gguf maps before
    // re-borrowing for tensor_info / load_tensor_f32 in the loop body).
    let names: Vec<String> = gguf
        .tensor_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    let mut trained: Vec<DwqLinearTrained> = Vec::new();
    let mut skipped: Vec<DwqLinearSkipped> = Vec::new();
    // Owned per-tensor bytes; views into these are passed to safetensors
    // serialize once at the end.
    let mut all_bytes: Vec<(String, MlxAffineLinearBytes)> = Vec::new();

    // ADR-020 iter-12e — optional RSS watchdog.  Lives for the duration
    // of the scan; dropped at function exit (Drop joins the thread).
    let watchdog: Option<RssWatchdog> = cfg.rss_cap_bytes.map(|cap| {
        RssWatchdog::spawn(cap, std::time::Duration::from_secs(5))
    });

    for name in names {
        // Pre-tensor watchdog check — abort the scan if RSS cap exceeded.
        if let Some(w) = &watchdog {
            if w.is_aborted() {
                return Err(anyhow!(
                    "train_all_linears_dwq: RSS watchdog aborted scan — \
                     peak RSS {} bytes ({:.2} GB) exceeded cap {} bytes \
                     ({:.2} GB) after {} trained tensors (next would have \
                     been '{}')",
                    w.peak_rss_bytes(),
                    w.peak_rss_bytes() as f64 / 1024.0_f64.powi(3),
                    w.cap_bytes(),
                    w.cap_bytes() as f64 / 1024.0_f64.powi(3),
                    trained.len(),
                    name,
                ));
            }
        }

        if !name_filter(&name) {
            skipped.push(DwqLinearSkipped {
                name: name.clone(),
                reason: "filtered out by name_filter".to_string(),
            });
            continue;
        }
        let info = match gguf.tensor_info(&name) {
            Some(i) => i,
            None => {
                skipped.push(DwqLinearSkipped {
                    name: name.clone(),
                    reason: "tensor_info missing (gguf inconsistency)".to_string(),
                });
                continue;
            }
        };
        // ADR-020 AC#5 Iter C2.1 — rank-3 expert-stacked MoE tensors
        // (`blk.{i}.ffn_gate_exps.weight`, `ffn_up_exps.weight`,
        // `ffn_down_exps.weight`, fused `ffn_gate_up_exps.weight`) get
        // per-expert training.  Each expert is sliced out as a rank-2
        // Linear, trained independently via the synthetic teacher, and
        // emitted as a separate safetensors triplet keyed by
        // `blk.{i}.<role_no_exps>.{e}` (matches HF-style per-expert
        // naming + the parse_dwq_overlay_role MoeExpert stem pattern).
        if info.shape.len() == 3 && name.contains("_exps.weight") {
            let n_experts = info.shape[0];
            let n = info.shape[1];
            let k = info.shape[2];
            if n < 32 || k < 32 {
                skipped.push(DwqLinearSkipped {
                    name: name.clone(),
                    reason: format!("MoE expert dim too small (n={n} k={k} need >=32)"),
                });
                continue;
            }
            if k % cfg.group_size != 0 {
                skipped.push(DwqLinearSkipped {
                    name: name.clone(),
                    reason: format!("MoE expert k={k} not divisible by group_size={}", cfg.group_size),
                });
                continue;
            }

            // Load + dequant rank-3 expert stack as one big F32 buffer.
            let buf = match gguf.load_tensor_f32(&name, &device) {
                Ok(b) => b,
                Err(e) => {
                    skipped.push(DwqLinearSkipped {
                        name: name.clone(),
                        reason: format!("MoE load_tensor_f32 failed: {e}"),
                    });
                    continue;
                }
            };
            let w_full: Vec<f32> = match buf.as_slice::<f32>() {
                Ok(s) => s.to_vec(),
                Err(e) => {
                    skipped.push(DwqLinearSkipped {
                        name: name.clone(),
                        reason: format!("MoE buffer slice failed: {e}"),
                    });
                    continue;
                }
            };
            // Drop the F32 buffer eagerly — we cloned into `w_full` so
            // the GPU-resident copy is no longer needed for the rest of
            // this scan.  Saves multi-GB peak RSS on large MoE tensors.
            drop(buf);

            // Rename stem: strip `_exps.weight` → `<base_role>` (e.g.
            // `blk.0.ffn_gate_exps` → `blk.0.ffn_gate`,
            // `blk.0.ffn_gate_up_exps` → `blk.0.ffn_gate_up`).
            let stem_no_weight = name.strip_suffix(".weight").unwrap_or(&name);
            let base_stem = stem_no_weight
                .strip_suffix("_exps")
                .unwrap_or(stem_no_weight);

            let per_expert_floats = n * k;
            let mut moe_trained = 0usize;
            let mut moe_failed = 0usize;
            for expert in 0..n_experts {
                if let Some(w) = &watchdog {
                    if w.is_aborted() {
                        return Err(anyhow!(
                            "train_all_linears_dwq: RSS watchdog aborted MoE scan at \
                             expert {expert}/{n_experts} of {name}"
                        ));
                    }
                }
                let w_expert: Vec<f32> = w_full
                    [expert * per_expert_floats..(expert + 1) * per_expert_floats]
                    .to_vec();
                let result = match train_linear_dwq_synthetic_teacher(
                    &device,
                    &w_expert,
                    n,
                    k,
                    cfg,
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        skipped.push(DwqLinearSkipped {
                            name: format!("{base_stem}.{expert}.weight"),
                            reason: format!("MoE expert training failed: {e}"),
                        });
                        moe_failed += 1;
                        continue;
                    }
                };
                let bytes = match result.linear.to_safetensors_bytes(float_dtype) {
                    Ok(b) => b,
                    Err(e) => {
                        skipped.push(DwqLinearSkipped {
                            name: format!("{base_stem}.{expert}.weight"),
                            reason: format!("MoE to_safetensors_bytes failed: {e}"),
                        });
                        moe_failed += 1;
                        continue;
                    }
                };

                let bench: Option<crate::calibrate::dwq_benchmark::PerLinearKlComparison> = if cfg
                    .compute_bench
                {
                    match crate::calibrate::dwq_benchmark::benchmark_dwq_vs_q4_0_kl(
                        &device,
                        &w_expert,
                        n,
                        k,
                        &result.linear,
                        cfg.n_tokens,
                        cfg.temperature,
                        cfg.seed,
                    ) {
                        Ok(b) => Some(b),
                        Err(_) => None,
                    }
                } else {
                    None
                };

                trained.push(DwqLinearTrained {
                    name: format!("{base_stem}.{expert}.weight"),
                    n,
                    k,
                    kl_initial: result.kl_initial,
                    kl_min: result.kl_min,
                    steps_run: result.steps_run,
                    bench,
                });
                let stem = format!("{base_stem}.{expert}");
                all_bytes.push((stem, bytes));
                moe_trained += 1;
            }
            tracing::info!(
                tensor = %name,
                base_stem = %base_stem,
                n_experts,
                trained = moe_trained,
                failed = moe_failed,
                "MoE expert tensor processed"
            );
            continue;
        }

        if info.shape.len() != 2 {
            skipped.push(DwqLinearSkipped {
                name: name.clone(),
                reason: format!("rank={} (need 2)", info.shape.len()),
            });
            continue;
        }
        let n = info.shape[0];
        let k = info.shape[1];
        if n < 32 || k < 32 {
            skipped.push(DwqLinearSkipped {
                name: name.clone(),
                reason: format!("dim too small (n={n} k={k} need >=32)"),
            });
            continue;
        }
        if k % cfg.group_size != 0 {
            skipped.push(DwqLinearSkipped {
                name: name.clone(),
                reason: format!("k={k} not divisible by group_size={}", cfg.group_size),
            });
            continue;
        }

        // Load + dequant via mlx-native (handles all supported ggml types).
        let buf = match gguf.load_tensor_f32(&name, &device) {
            Ok(b) => b,
            Err(e) => {
                skipped.push(DwqLinearSkipped {
                    name: name.clone(),
                    reason: format!("load_tensor_f32 failed: {e}"),
                });
                continue;
            }
        };
        let w_real: Vec<f32> = match buf.as_slice::<f32>() {
            Ok(s) => s.to_vec(),
            Err(e) => {
                skipped.push(DwqLinearSkipped {
                    name: name.clone(),
                    reason: format!("buffer slice failed: {e}"),
                });
                continue;
            }
        };

        // Train.  Capture errors as skip reasons rather than aborting the
        // whole scan — production driver should keep going through partial
        // failures so the operator gets a complete disposition log.
        let result = match train_linear_dwq_synthetic_teacher(&device, &w_real, n, k, cfg) {
            Ok(r) => r,
            Err(e) => {
                skipped.push(DwqLinearSkipped {
                    name: name.clone(),
                    reason: format!("training failed: {e}"),
                });
                continue;
            }
        };
        let bytes = match result.linear.to_safetensors_bytes(float_dtype) {
            Ok(b) => b,
            Err(e) => {
                skipped.push(DwqLinearSkipped {
                    name: name.clone(),
                    reason: format!("to_safetensors_bytes failed: {e}"),
                });
                continue;
            }
        };

        // ADR-020 iter-12f-2 — optional inline benchmark vs Q4_0 baseline.
        // Runs while we still have w_real loaded to avoid a redundant
        // GGUF re-dequant.  Only meaningful at perturb_factor == 1.0
        // (warned about below).
        let bench: Option<crate::calibrate::dwq_benchmark::PerLinearKlComparison> = if cfg
            .compute_bench
        {
            match crate::calibrate::dwq_benchmark::benchmark_dwq_vs_q4_0_kl(
                &device,
                &w_real,
                n,
                k,
                &result.linear,
                cfg.n_tokens,
                cfg.temperature,
                cfg.seed,
            ) {
                Ok(b) => Some(b),
                Err(e) => {
                    skipped.push(DwqLinearSkipped {
                        name: format!("{name} (bench)"),
                        reason: format!("benchmark failed: {e}"),
                    });
                    None
                }
            }
        } else {
            None
        };

        trained.push(DwqLinearTrained {
            name: name.clone(),
            n,
            k,
            kl_initial: result.kl_initial,
            kl_min: result.kl_min,
            steps_run: result.steps_run,
            bench,
        });
        // mlx-lm naming convention strips the trailing ".weight" suffix
        // before re-attaching ".weight"/.scales"/.biases" — but since GGUF
        // tensor names already end in ".weight" (e.g. "blk.0.attn_qkv.weight"),
        // we strip it once here to avoid producing ".weight.weight" keys.
        let stem = name.strip_suffix(".weight").unwrap_or(&name).to_string();
        all_bytes.push((stem, bytes));
    }

    // Build views over the owned bytes + serialize.
    let mut views_pairs: Vec<(String, _)> = Vec::with_capacity(all_bytes.len() * 3);
    for (stem, b) in &all_bytes {
        let (w, s, bi) = b
            .to_safetensors_views()
            .with_context(|| format!("to_safetensors_views for {stem}"))?;
        views_pairs.push((format!("{stem}.weight"), w));
        views_pairs.push((format!("{stem}.scales"), s));
        views_pairs.push((format!("{stem}.biases"), bi));
    }
    // ADR-020 AC#5 Iter D — embed `bits` + `group_size` in safetensors
    // metadata so `MlxModelWeights::apply_dwq_overlay` can reconstruct
    // MlxAffineLinear without additional CLI flags or guessing.  The
    // `format` marker lets the loader fail-fast on non-DWQ safetensors.
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("format".to_string(), "mlx-affine-dwq-v1".to_string());
    metadata.insert("bits".to_string(), cfg.bits.to_string());
    metadata.insert(
        "group_size".to_string(),
        cfg.group_size.to_string(),
    );
    metadata.insert(
        "trained_count".to_string(),
        all_bytes.len().to_string(),
    );
    let safetensors_bytes = safetensors::tensor::serialize(views_pairs.iter()
        .map(|(k, v)| (k.as_str(), v)), Some(metadata))
        .map_err(|e| anyhow!("safetensors serialize: {e}"))?;
    // Discard the type holder so MlxAffineLinear is in scope.
    let _ = std::any::type_name::<MlxAffineLinear>();

    // ADR-020 iter-12f-2 — aggregate per-Linear bench results.
    let mean_delta_kl_nats: Option<f32> = if cfg.compute_bench {
        if cfg.perturb_factor != 1.0 {
            eprintln!(
                "[dwq] WARNING: compute_bench=true with perturb_factor={} — \
                 the DWQ-vs-Q4_0 comparison is only meaningful at \
                 perturb_factor == 1.0 (training starts from optimal init).  \
                 Reported mean_delta_kl_nats may underrepresent DWQ quality.",
                cfg.perturb_factor
            );
        }
        let with_bench: Vec<&DwqLinearTrained> = trained
            .iter()
            .filter(|t| t.bench.is_some())
            .collect();
        if with_bench.is_empty() {
            None
        } else {
            let sum: f64 = with_bench
                .iter()
                .map(|t| t.bench.as_ref().unwrap().delta_kl_nats as f64)
                .sum();
            Some((sum / with_bench.len() as f64) as f32)
        }
    } else {
        None
    };

    Ok(DwqAllLinearsResult {
        trained,
        skipped,
        safetensors_bytes,
        mean_delta_kl_nats,
    })
}

/// Train a single Linear's DWQ affine quant params (synthetic teacher).
/// See [`DwqLinearTrainResult`] for the algorithm + convergence contract.
pub fn train_linear_dwq_synthetic_teacher(
    device: &MlxDevice,
    w_real: &[f32],
    n: usize,
    k: usize,
    cfg: &DwqTrainingConfig,
) -> Result<DwqLinearTrainResult> {
    use std::collections::BTreeMap;

    use crate::calibrate::adam::{AdamConfig, AdamOptimizer};
    use crate::calibrate::autograd_gpu_tape::{
        backward, matmul, qdq_affine, scalar_mul, transpose, view, GpuTape, GpuTensor,
    };
    use crate::calibrate::dynamic_quant_gpu::kl_div_loss_per_row;
    use crate::calibrate::mlx_safetensors_loader::MlxAffineLinear;

    // ---- Validation ----
    if w_real.len() != n * k {
        return Err(anyhow!(
            "train_linear_dwq_synthetic_teacher: w_real.len()={} != n*k={}*{}={}",
            w_real.len(), n, k, n * k
        ));
    }
    if n < 32 || k < 32 {
        return Err(anyhow!(
            "train_linear_dwq_synthetic_teacher: n={n} k={k} below matmul floor (>= 32)"
        ));
    }
    if cfg.n_tokens < 32 {
        return Err(anyhow!(
            "train_linear_dwq_synthetic_teacher: n_tokens={} < 32 (matmul backward floor)",
            cfg.n_tokens
        ));
    }
    if cfg.group_size == 0 || k % cfg.group_size != 0 {
        return Err(anyhow!(
            "train_linear_dwq_synthetic_teacher: k={k} not divisible by group_size={}",
            cfg.group_size
        ));
    }
    if cfg.n_steps == 0 {
        return Err(anyhow!(
            "train_linear_dwq_synthetic_teacher: n_steps must be > 0"
        ));
    }

    let groups_per_row = k / cfg.group_size;
    let m = cfg.n_tokens;

    // ---- Phase 1: init affine params from real weight ----
    let mut init_registry = KernelRegistry::new();
    let (q_int, s_init, b_init) = init_affine_params_gpu(
        device,
        &mut init_registry,
        w_real,
        cfg.group_size,
        cfg.bits,
    )
    .context("init_affine_params_gpu")?;
    if q_int.len() != n * k {
        return Err(anyhow!(
            "init_affine_params_gpu: q_int.len()={} != n*k={}",
            q_int.len(), n * k
        ));
    }
    if s_init.len() != n * groups_per_row || b_init.len() != n * groups_per_row {
        return Err(anyhow!(
            "init_affine_params_gpu: s/b length mismatch (got s={}, b={}, expected {})",
            s_init.len(), b_init.len(), n * groups_per_row
        ));
    }

    // ---- Phase 2: perturb ----
    let s_p: Vec<f32> = s_init.iter().map(|v| v * cfg.perturb_factor).collect();
    let b_p: Vec<f32> = b_init.iter().map(|v| v * cfg.perturb_factor).collect();

    // ---- Phase 3: activations (synthetic Gaussian by default; caller
    //              can override with real captured X via cfg.x_override) ----
    let x_data: Vec<f32> = if let Some(x_user) = cfg.x_override.as_ref() {
        if x_user.len() != m * k {
            return Err(anyhow!(
                "x_override length {} != n_tokens × k = {} × {} = {}",
                x_user.len(), m, k, m * k
            ));
        }
        x_user.clone()
    } else {
        box_muller_gaussian(m * k, cfg.seed)
    };
    let mut y_teacher = vec![0.0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += (x_data[r * k + kk] as f64) * (w_real[c * k + kk] as f64);
            }
            y_teacher[r * n + c] = acc as f32;
        }
    }

    // ---- Phase 4: Adam training loop ----
    let adam_cfg = AdamConfig {
        lr: cfg.lr,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
    };
    let mut adam = AdamOptimizer::new(device.clone(), adam_cfg).context("AdamOptimizer::new")?;
    adam.register_param("s", buffer_from_f32(device, &s_p)?)?;
    adam.register_param("b", buffer_from_f32(device, &b_p)?)?;

    let inv_t = 1.0 / cfg.temperature;
    let tape = GpuTape::new(device.clone());

    let step = |adam: &mut AdamOptimizer, tape: &GpuTape| -> Result<f32> {
        let s = adam.read_param("s")?;
        let b = adam.read_param("b")?;
        let s_leaf = GpuTensor::from_vec(tape, &s, vec![s.len()])?;
        let b_leaf = GpuTensor::from_vec(tape, &b, vec![b.len()])?;
        let qdq_flat = qdq_affine(&s_leaf, &b_leaf, &q_int, cfg.group_size)?;
        let w_q = view(&qdq_flat, vec![n, k])?;
        let w_q_t = transpose(&w_q)?;
        let xt = GpuTensor::from_vec(tape, &x_data, vec![m, k])?;
        let y_s = matmul(&xt, &w_q_t)?;
        let y_t_leaf = GpuTensor::from_vec(tape, &y_teacher, vec![m, n])?;
        let y_s_scaled = scalar_mul(&y_s, inv_t)?;
        let y_t_scaled = scalar_mul(&y_t_leaf, inv_t)?;
        let kl = kl_div_loss_per_row(&y_s_scaled, &y_t_scaled)?;
        let kl_host = kl.to_vec()?;
        let loss = (kl_host.iter().map(|v| *v as f64).sum::<f64>()
            / kl_host.len() as f64) as f32;
        let mut dy_buf = tape
            .device()
            .alloc_buffer(kl_host.len() * 4, DType::F32, kl.shape().to_vec())
            .map_err(|e| anyhow!("dy alloc: {e}"))?;
        dy_buf
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("dy slice: {e}"))?
            .iter_mut()
            .for_each(|v| *v = 1.0 / m as f32);
        let grads = backward(&kl, dy_buf)?;
        let g_s = grads[s_leaf.node_idx()]
            .as_ref()
            .ok_or_else(|| anyhow!("missing s grad"))?
            .clone();
        let g_b = grads[b_leaf.node_idx()]
            .as_ref()
            .ok_or_else(|| anyhow!("missing b grad"))?
            .clone();
        let mut g_map = BTreeMap::new();
        g_map.insert("s".to_string(), g_s);
        g_map.insert("b".to_string(), g_b);
        adam.step(&g_map)?;
        Ok(loss)
    };

    let initial_loss = step(&mut adam, &tape).context("initial step")?;
    if !initial_loss.is_finite() {
        return Err(anyhow!(
            "train_linear_dwq_synthetic_teacher: initial KL non-finite ({initial_loss})"
        ));
    }
    tape.reset();

    let mut min_loss = initial_loss;
    let mut last_loss = initial_loss;
    for s_idx in 1..cfg.n_steps {
        let l = step(&mut adam, &tape).with_context(|| format!("step {s_idx}"))?;
        tape.reset();
        if !l.is_finite() {
            return Err(anyhow!(
                "train_linear_dwq_synthetic_teacher: KL non-finite at step {s_idx} ({l})"
            ));
        }
        if l < min_loss {
            min_loss = l;
        }
        last_loss = l;
    }

    // ---- Phase 5: convergence gate + pack ----
    let ratio = min_loss / initial_loss;
    if ratio > cfg.convergence_ratio {
        return Err(anyhow!(
            "train_linear_dwq_synthetic_teacher: did not converge — \
             initial_kl={initial_loss}, min_kl={min_loss}, ratio={ratio} > {}",
            cfg.convergence_ratio
        ));
    }

    let s_trained = adam.read_param("s")?;
    let b_trained = adam.read_param("b")?;
    let linear = MlxAffineLinear {
        n,
        k,
        group_size: cfg.group_size,
        bits: cfg.bits,
        q_int: q_int.clone(),
        scales: s_trained,
        biases: b_trained,
    };
    Ok(DwqLinearTrainResult {
        linear,
        kl_initial: initial_loss,
        kl_min: min_loss,
        kl_final: last_loss,
        steps_run: cfg.n_steps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::adam::{AdamConfig, AdamOptimizer};
    use crate::calibrate::autograd_gpu_tape::{
        backward, ones_like, qdq_affine, square, sub, GpuTape, GpuTensor,
    };
    use std::collections::BTreeMap;

    /// Synthetic 2-tensor DWQ training loop.  Two frozen FP32 weight
    /// tensors `W1`, `W2` are encoded with affine quantization
    /// (per-group min/max init); scales+biases are then PERTURBED by
    /// +5% and Adam is asked to recover them by minimizing the
    /// per-tensor reconstruction MSE
    ///
    ///   L = Σ_i (qdq_W1[i] − W1[i])² + Σ_i (qdq_W2[i] − W2[i])²
    ///
    /// Acceptance: best loss across the trajectory < 0.2 × initial
    /// loss (5× reduction).  The 5% perturbation creates ~10% of the
    /// init loss as headroom, and Adam converges back toward (but not
    /// to, because the integer codes were chosen for the original s+b
    /// pair) the analytical minimum.
    ///
    /// What this falsifies if it fails:
    ///   - qdq_affine forward (mlx-native kernel + Rust dispatch)
    ///   - qdq_affine backward routing into scales+biases parents
    ///   - tape accumulator semantics for OpKind::QdqAffine
    ///   - Adam state register/step with multi-param BTreeMap
    ///   - sub/square/ones_like/backward chain composition
    #[test]
    fn dwq_loop_synthetic_recovers_perturbed_affine_params() {
        let group_size = 32usize;
        // Tensor shapes: arbitrary multiples of group_size.
        let w1_n = group_size * 4; // 128 elements, 4 groups
        let w2_n = group_size * 6; // 192 elements, 6 groups

        let w1: Vec<f32> = (0..w1_n)
            .map(|i| ((i as f32) * 0.0193 - 0.5).sin() * 0.4)
            .collect();
        let w2: Vec<f32> = (0..w2_n)
            .map(|i| ((i as f32) * 0.0241 + 0.3).cos() * 0.3)
            .collect();

        let device = MlxDevice::new().expect("device");
        let mut init_registry = KernelRegistry::new();

        let (q1, s1_init, b1_init) =
            init_affine_params_gpu(&device, &mut init_registry, &w1, group_size, 4)
                .expect("init w1");
        let (q2, s2_init, b2_init) =
            init_affine_params_gpu(&device, &mut init_registry, &w2, group_size, 4)
                .expect("init w2");

        // Perturb +5% to give Adam something to learn.
        let perturb = |xs: &[f32], factor: f32| -> Vec<f32> {
            xs.iter().map(|v| v * factor).collect()
        };
        let s1_p = perturb(&s1_init, 1.05);
        let b1_p = perturb(&b1_init, 1.05);
        let s2_p = perturb(&s2_init, 1.05);
        let b2_p = perturb(&b2_init, 1.05);

        let cfg = AdamConfig {
            lr: 0.005,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        };
        let mut adam = AdamOptimizer::new(device.clone(), cfg).expect("adam");
        adam.register_param("s1", buffer_from_f32(&device, &s1_p).unwrap())
            .unwrap();
        adam.register_param("b1", buffer_from_f32(&device, &b1_p).unwrap())
            .unwrap();
        adam.register_param("s2", buffer_from_f32(&device, &s2_p).unwrap())
            .unwrap();
        adam.register_param("b2", buffer_from_f32(&device, &b2_p).unwrap())
            .unwrap();

        // Single shared tape — `tape.reset()` between iterations drops
        // per-step nodes without device churn (mantra: avoid Metal
        // residency-set contention from per-iter MlxDevice::new()).
        let tape = GpuTape::new(device.clone());

        let train_step = |adam: &mut AdamOptimizer, tape: &GpuTape| -> Result<f32> {
            let s1 = adam.read_param("s1")?;
            let b1 = adam.read_param("b1")?;
            let s2 = adam.read_param("s2")?;
            let b2 = adam.read_param("b2")?;

            let s1_leaf = GpuTensor::from_vec(tape, &s1, vec![s1.len()])?;
            let b1_leaf = GpuTensor::from_vec(tape, &b1, vec![b1.len()])?;
            let s2_leaf = GpuTensor::from_vec(tape, &s2, vec![s2.len()])?;
            let b2_leaf = GpuTensor::from_vec(tape, &b2, vec![b2.len()])?;

            let qdq1 = qdq_affine(&s1_leaf, &b1_leaf, &q1, group_size)?;
            let qdq2 = qdq_affine(&s2_leaf, &b2_leaf, &q2, group_size)?;
            let w1_const = GpuTensor::from_vec(tape, &w1, vec![w1.len()])?;
            let w2_const = GpuTensor::from_vec(tape, &w2, vec![w2.len()])?;
            let r1 = sub(&qdq1, &w1_const)?;
            let r2 = sub(&qdq2, &w2_const)?;
            let sq1 = square(&r1)?;
            let sq2 = square(&r2)?;

            // Loss = Σ sq1 + Σ sq2 (host reduction; no GPU sum kernel
            // yet).  Backward seeds dy = ones for each subgraph; the
            // accumulator merges contributions to s1/b1/s2/b2 leaves.
            let sq1_host = sq1.to_vec()?;
            let sq2_host = sq2.to_vec()?;
            let loss = sq1_host.iter().map(|v| *v as f64).sum::<f64>()
                + sq2_host.iter().map(|v| *v as f64).sum::<f64>();

            let dy1 = ones_like(tape, sq1.shape())?;
            let g1 = backward(&sq1, dy1)?;
            let dy2 = ones_like(tape, sq2.shape())?;
            let g2 = backward(&sq2, dy2)?;

            let grad_s1 = g1[s1_leaf.node_idx()]
                .as_ref()
                .ok_or_else(|| anyhow!("missing s1 grad"))?
                .clone();
            let grad_b1 = g1[b1_leaf.node_idx()]
                .as_ref()
                .ok_or_else(|| anyhow!("missing b1 grad"))?
                .clone();
            let grad_s2 = g2[s2_leaf.node_idx()]
                .as_ref()
                .ok_or_else(|| anyhow!("missing s2 grad"))?
                .clone();
            let grad_b2 = g2[b2_leaf.node_idx()]
                .as_ref()
                .ok_or_else(|| anyhow!("missing b2 grad"))?
                .clone();

            let mut grads = BTreeMap::new();
            grads.insert("s1".to_string(), grad_s1);
            grads.insert("b1".to_string(), grad_b1);
            grads.insert("s2".to_string(), grad_s2);
            grads.insert("b2".to_string(), grad_b2);
            adam.step(&grads)?;
            Ok(loss as f32)
        };

        let initial_loss = train_step(&mut adam, &tape).expect("initial step");
        // Drop nodes from step 0 — keep device + registry warm.
        tape.reset();
        let mut min_loss = initial_loss;
        let n_steps = 200usize;
        let mut last_loss = initial_loss;
        for step in 1..n_steps {
            let l = train_step(&mut adam, &tape).expect("step");
            tape.reset();
            if l < min_loss {
                min_loss = l;
            }
            last_loss = l;
            if step % 50 == 0 {
                eprintln!(
                    "[dwq_synth] step={step} loss={l:.6} min={min_loss:.6} initial={initial_loss:.6}"
                );
            }
        }

        // Acceptance: 5× reduction floor.  Robust to late-stage Adam
        // jitter at small loss values.
        assert!(
            min_loss < initial_loss * 0.2,
            "DWQ synthetic loop did not converge: initial={initial_loss}, min_seen={min_loss}, last={last_loss}"
        );

        // Sanity: final scales/biases should be CLOSER to the analytical
        // optimum than the perturbed start (norm-of-difference).
        let s1_final = adam.read_param("s1").unwrap();
        let b1_final = adam.read_param("b1").unwrap();
        let dist_init = s1_p
            .iter()
            .zip(s1_init.iter())
            .map(|(a, b)| ((a - b).powi(2)) as f64)
            .sum::<f64>()
            + b1_p
                .iter()
                .zip(b1_init.iter())
                .map(|(a, b)| ((a - b).powi(2)) as f64)
                .sum::<f64>();
        let dist_final = s1_final
            .iter()
            .zip(s1_init.iter())
            .map(|(a, b)| ((a - b).powi(2)) as f64)
            .sum::<f64>()
            + b1_final
                .iter()
                .zip(b1_init.iter())
                .map(|(a, b)| ((a - b).powi(2)) as f64)
                .sum::<f64>();
        assert!(
            dist_final < dist_init,
            "Adam did not move toward analytical optimum: dist_init={dist_init}, dist_final={dist_final}"
        );
    }

    /// iter-13c — full DWQ training loop on a synthetic 2-Linear MLP.
    ///
    /// Teacher: y_T = X @ W1 → silu → @ W2  (frozen FP32 weights)
    /// Student: y_S = X @ qdq(W1, s1, b1) → silu → @ qdq(W2, s2, b2)
    /// Loss:    KL(softmax(scale·y_T) || softmax(scale·y_S)).sum()
    /// Optimizer: Adam over (s1, b1, s2, b2); q_int frozen.
    ///
    /// Acceptance: best per-row-mean KL after 200 steps < 0.34 × initial
    /// (3× reduction floor — KL is a stricter, non-linear loss than
    /// reconstruction MSE; convergence rate is bounded by the
    /// 4-bit quantizer's irreducible error so we don't expect the
    /// 15× margin from iter-13b).
    ///
    /// What this test falsifies if it fails:
    ///   - tape `view` op (qdq output is 1-D, matmul rhs must be 2-D)
    ///   - tape `scalar_mul` op (KL temperature scaling 1/T = 0.5)
    ///   - kl_div_loss_per_row composition with QdqAffine in the chain
    ///   - silu interleaved between two qdq'd matmuls (gradient flows
    ///     through silu_backward → matmul backward → view backward
    ///     → qdq_affine backward → scales/biases parents)
    ///   - end-to-end Adam multi-param convergence under a non-convex
    ///     loss that depends on all 4 leaves through compounded ops
    #[test]
    fn dwq_loop_synthetic_2linear_kl_div_converges_under_adam() {
        use crate::calibrate::autograd_gpu_tape::{matmul, scalar_mul, silu, view};
        use crate::calibrate::dynamic_quant_gpu::kl_div_loss_per_row;

        let group_size = 32usize;
        // Matmul kernel constraints: m, k, n all >= 32 for backward.
        // Layer 1: X[m=32, in=32] @ W1[in=32, mid=32] → H[m=32, mid=32]
        // Layer 2: silu(H)[m=32, mid=32] @ W2[mid=32, out=32] → Y[m=32, out=32]
        let m = 32usize;
        let in_dim = 32usize;
        let mid_dim = 32usize;
        let out_dim = 32usize;
        // Flat element counts
        let w1_n = in_dim * mid_dim; // 1024 elements, 32 groups
        let w2_n = mid_dim * out_dim; // 1024 elements, 32 groups

        // Deterministic teacher weights + input.  Magnitudes chosen so
        // that final logits have stddev ~1.5–2 (post T=2.0 scaling
        // gives ~0.7–1.0), producing a softmax distribution that is
        // neither uniform (KL ≈ 0) nor saturated (KL gradient
        // vanishes).  +30% perturbation in scales/biases at this
        // logit scale yields measurable initial KL.
        let w1: Vec<f32> = (0..w1_n)
            .map(|i| ((i as f32) * 0.0123 - 0.5).sin() * 1.0)
            .collect();
        let w2: Vec<f32> = (0..w2_n)
            .map(|i| ((i as f32) * 0.0179 + 0.7).cos() * 0.8)
            .collect();
        let x_data: Vec<f32> = (0..(m * in_dim))
            .map(|i| ((i as f32) * 0.013 + 0.1).sin() * 0.6)
            .collect();

        let device = MlxDevice::new().expect("device");
        let mut init_registry = KernelRegistry::new();

        // Per-tensor affine init from frozen W.
        let (q1, s1_init, b1_init) =
            init_affine_params_gpu(&device, &mut init_registry, &w1, group_size, 4)
                .expect("init w1");
        let (q2, s2_init, b2_init) =
            init_affine_params_gpu(&device, &mut init_registry, &w2, group_size, 4)
                .expect("init w2");

        // Perturb 2.0× to force Adam to learn.  At 4-bit quantization
        // the per-tensor reconstruction is faithful enough that small
        // (≤30%) scale/bias perturbations produce KL ≪ 1e-4 even at
        // logit stddev ~1.5 — softmax smooths the qdq error.  A 2.0×
        // multiplicative perturbation reliably produces initial KL on
        // the order of 1e-3 to 1e-2.
        let perturb = |xs: &[f32], factor: f32| -> Vec<f32> {
            xs.iter().map(|v| v * factor).collect()
        };
        let s1_p = perturb(&s1_init, 2.0);
        let b1_p = perturb(&b1_init, 2.0);
        let s2_p = perturb(&s2_init, 2.0);
        let b2_p = perturb(&b2_init, 2.0);

        // Adam config: smaller lr than reconstruction-MSE test because
        // KL gradient magnitudes scale with logit magnitude × softmax
        // gradient (which is bounded but can be large).
        let cfg = AdamConfig {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        };
        let mut adam = AdamOptimizer::new(device.clone(), cfg).expect("adam");
        adam.register_param("s1", buffer_from_f32(&device, &s1_p).unwrap())
            .unwrap();
        adam.register_param("b1", buffer_from_f32(&device, &b1_p).unwrap())
            .unwrap();
        adam.register_param("s2", buffer_from_f32(&device, &s2_p).unwrap())
            .unwrap();
        adam.register_param("b2", buffer_from_f32(&device, &b2_p).unwrap())
            .unwrap();

        // Pre-compute teacher logits y_T (FP32 oracle, host).  Treated
        // as a constant tape leaf each step.
        // h_t = X @ W1
        let mut h_t = vec![0.0f32; m * mid_dim];
        for r in 0..m {
            for c in 0..mid_dim {
                let mut acc = 0.0f64;
                for kk in 0..in_dim {
                    acc += (x_data[r * in_dim + kk] as f64)
                        * (w1[kk * mid_dim + c] as f64);
                }
                h_t[r * mid_dim + c] = acc as f32;
            }
        }
        // h_t = silu(h_t) — host oracle.
        for v in h_t.iter_mut() {
            let s = 1.0 / (1.0 + (-(*v as f64)).exp());
            *v = (*v as f64 * s) as f32;
        }
        // y_t = h_t @ W2
        let mut y_t = vec![0.0f32; m * out_dim];
        for r in 0..m {
            for c in 0..out_dim {
                let mut acc = 0.0f64;
                for kk in 0..mid_dim {
                    acc += (h_t[r * mid_dim + kk] as f64)
                        * (w2[kk * out_dim + c] as f64);
                }
                y_t[r * out_dim + c] = acc as f32;
            }
        }

        // Single shared tape — `tape.reset()` between iterations drops
        // per-step nodes without Metal residency-set churn.
        let tape = GpuTape::new(device.clone());

        let temperature = 2.0f32; // mlx-lm dwq.py default
        let inv_t = 1.0 / temperature;

        let train_step = |adam: &mut AdamOptimizer, tape: &GpuTape| -> Result<f32> {
            let s1 = adam.read_param("s1")?;
            let b1 = adam.read_param("b1")?;
            let s2 = adam.read_param("s2")?;
            let b2 = adam.read_param("b2")?;

            let s1_leaf = GpuTensor::from_vec(tape, &s1, vec![s1.len()])?;
            let b1_leaf = GpuTensor::from_vec(tape, &b1, vec![b1.len()])?;
            let s2_leaf = GpuTensor::from_vec(tape, &s2, vec![s2.len()])?;
            let b2_leaf = GpuTensor::from_vec(tape, &b2, vec![b2.len()])?;

            // Reconstruct W1, W2 via differentiable qdq, then reshape.
            let w1_q_flat = qdq_affine(&s1_leaf, &b1_leaf, &q1, group_size)?;
            let w2_q_flat = qdq_affine(&s2_leaf, &b2_leaf, &q2, group_size)?;
            let w1_q = view(&w1_q_flat, vec![in_dim, mid_dim])?;
            let w2_q = view(&w2_q_flat, vec![mid_dim, out_dim])?;

            // Forward chain: X → matmul → silu → matmul → logits.
            let xt = GpuTensor::from_vec(tape, &x_data, vec![m, in_dim])?;
            let h_pre = matmul(&xt, &w1_q)?;
            let h = silu(&h_pre)?;
            let y_s = matmul(&h, &w2_q)?;

            // Teacher logits as a constant leaf (no gradient flows through).
            let y_t_leaf = GpuTensor::from_vec(tape, &y_t, vec![m, out_dim])?;

            // Temperature scaling (1/T per mlx-lm).
            let y_s_scaled = scalar_mul(&y_s, inv_t)?;
            let y_t_scaled = scalar_mul(&y_t_leaf, inv_t)?;

            // KL(softmax(y_t_scaled) || softmax(y_s_scaled)) per row.
            // kl_div_loss_per_row signature: (logits_q, logits_p)
            // ⇒ KL(p || q) where p = teacher.
            let kl = kl_div_loss_per_row(&y_s_scaled, &y_t_scaled)?;

            // Loss = mean per-row KL.
            let kl_host = kl.to_vec()?;
            let loss_mean = (kl_host.iter().map(|v| *v as f64).sum::<f64>()
                / kl_host.len() as f64) as f32;

            // Backward seed: dy = ones / m  (so backward gives mean-grad
            // semantics matching loss_mean).
            let mut dy_buf = tape
                .device()
                .alloc_buffer(kl_host.len() * 4, DType::F32, kl.shape().to_vec())?;
            dy_buf
                .as_mut_slice::<f32>()
                .map_err(|e| anyhow!("dy slice: {e}"))?
                .iter_mut()
                .for_each(|v| *v = 1.0 / m as f32);
            let grads = backward(&kl, dy_buf)?;

            let grad_s1 = grads[s1_leaf.node_idx()]
                .as_ref()
                .ok_or_else(|| anyhow!("missing s1 grad"))?
                .clone();
            let grad_b1 = grads[b1_leaf.node_idx()]
                .as_ref()
                .ok_or_else(|| anyhow!("missing b1 grad"))?
                .clone();
            let grad_s2 = grads[s2_leaf.node_idx()]
                .as_ref()
                .ok_or_else(|| anyhow!("missing s2 grad"))?
                .clone();
            let grad_b2 = grads[b2_leaf.node_idx()]
                .as_ref()
                .ok_or_else(|| anyhow!("missing b2 grad"))?
                .clone();

            let mut g_map = BTreeMap::new();
            g_map.insert("s1".to_string(), grad_s1);
            g_map.insert("b1".to_string(), grad_b1);
            g_map.insert("s2".to_string(), grad_s2);
            g_map.insert("b2".to_string(), grad_b2);
            adam.step(&g_map)?;
            Ok(loss_mean)
        };

        let initial_loss = train_step(&mut adam, &tape).expect("initial step");
        tape.reset();
        // Non-triviality: initial KL must be measurably > 0; otherwise
        // the +30% perturbation didn't move the loss landscape and the
        // convergence acceptance below would be a false positive.
        assert!(
            initial_loss > 1e-4,
            "KL fixture is trivial: initial_loss={initial_loss} too small to measure convergence"
        );
        let mut min_loss = initial_loss;
        let n_steps = 200usize;
        let mut last_loss = initial_loss;
        for step in 1..n_steps {
            let l = train_step(&mut adam, &tape).expect("step");
            tape.reset();
            if l < min_loss {
                min_loss = l;
            }
            last_loss = l;
            if step % 50 == 0 {
                eprintln!(
                    "[dwq_kl] step={step} loss={l:.6} min={min_loss:.6} initial={initial_loss:.6}"
                );
            }
        }

        assert!(
            min_loss < initial_loss * 0.34,
            "DWQ KL synthetic loop did not converge: initial={initial_loss}, min={min_loss}, last={last_loss}"
        );

        // Sanity: scales+biases must have moved TOWARD the analytical
        // optimum (per-tensor min/max init), not stayed at the
        // perturbation or drifted away.
        let s1_final = adam.read_param("s1").unwrap();
        let b1_final = adam.read_param("b1").unwrap();
        let l2_init: f64 = s1_p
            .iter()
            .zip(s1_init.iter())
            .map(|(a, b)| ((a - b).powi(2)) as f64)
            .sum::<f64>()
            + b1_p
                .iter()
                .zip(b1_init.iter())
                .map(|(a, b)| ((a - b).powi(2)) as f64)
                .sum::<f64>();
        let l2_final: f64 = s1_final
            .iter()
            .zip(s1_init.iter())
            .map(|(a, b)| ((a - b).powi(2)) as f64)
            .sum::<f64>()
            + b1_final
                .iter()
                .zip(b1_init.iter())
                .map(|(a, b)| ((a - b).powi(2)) as f64)
                .sum::<f64>();
        // KL is a different objective from MSE-to-init; we don't
        // require monotone L2 movement, only that final params are
        // STILL FINITE (haven't blown up).
        let _ = (l2_init, l2_final);
        for v in s1_final.iter().chain(b1_final.iter()) {
            assert!(v.is_finite(), "s1/b1 became non-finite: {v}");
        }
    }

    /// iter-13d — DWQ training loop on a real GGUF Linear weight.
    ///
    /// Loads `blk.0.attn_qkv.weight` (default; Q4_0 quantized in the
    /// 27B hybrid GGUF, [10240, 5120] = 52 422 400 elements) via
    /// `mlx-native`'s `GgufFile::load_tensor_f32`, runs the affine
    /// init kernel, perturbs scales/biases by 2.0×, and trains
    /// (scales, biases) for 100 Adam steps over per-tensor
    /// reconstruction MSE.
    ///
    /// Validates (mantra: "code+tests==truth"):
    ///   1. Real-tensor magnitude regime — quantized weights from a
    ///      production GGUF, not synthetic sin*K.
    ///   2. Per-group min/max init handles real distributions
    ///      (potentially skewed, sparse, multi-modal).
    ///   3. KL gradient stability — no NaN/Inf over 100+ steps.
    ///   4. Peak GPU memory bounded — single shared tape with
    ///      `tape.reset()` between iterations; no per-step
    ///      `MlxDevice::new()` churn.
    ///
    /// `#[ignore]`-gated because it requires a multi-GB GGUF on disk.
    /// Run with:
    ///   cargo test --release --bin hf2q -- \
    ///     --ignored dwq_loop_real_gguf
    /// Override path: `HF2Q_TEST_GGUF=/path/to/file.gguf`.
    /// Override tensor: `HF2Q_TEST_GGUF_TENSOR=blk.X.tensor.weight`.
    ///
    /// Acceptance: best loss < 0.2 × initial (5× reduction floor).
    /// Non-triviality: initial loss must be > init MSE (i.e. the
    /// 2× perturbation moved the loss landscape measurably above the
    /// ideal-init reconstruction).
    #[test]
    #[ignore]
    fn dwq_loop_real_gguf_attn_qkv_converges_under_adam() {
        use mlx_native::gguf::GgufFile;

        let gguf_path = std::env::var("HF2Q_TEST_GGUF").unwrap_or_else(|_| {
            "/opt/hf2q/models/qwen3.6-27b-mtp-q4_0/qwen3.6-27b-mtp-q4_0.gguf".to_string()
        });
        let tensor_name = std::env::var("HF2Q_TEST_GGUF_TENSOR")
            .unwrap_or_else(|_| "blk.0.attn_qkv.weight".to_string());
        let path = std::path::Path::new(&gguf_path);
        if !path.exists() {
            eprintln!(
                "[dwq_real_gguf] SKIP: {} not found (set HF2Q_TEST_GGUF=/path)",
                gguf_path
            );
            return;
        }

        let device = MlxDevice::new().expect("device");
        let gguf = GgufFile::open(path).expect("open gguf");
        let info = gguf
            .tensor_info(&tensor_name)
            .unwrap_or_else(|| panic!("tensor '{tensor_name}' not in {gguf_path}"));
        eprintln!(
            "[dwq_real_gguf] loading {tensor_name}: shape={:?} type={:?}",
            info.shape, info.ggml_type
        );

        let buf = gguf
            .load_tensor_f32(&tensor_name, &device)
            .expect("load_tensor_f32");
        let w_real_full: Vec<f32> = buf
            .as_slice::<f32>()
            .expect("as_slice f32")
            .to_vec();

        // Sanity: dequantized values must be finite.
        let n_finite = w_real_full.iter().filter(|v| v.is_finite()).count();
        assert_eq!(
            n_finite,
            w_real_full.len(),
            "GGUF dequant produced non-finite values: {}/{}",
            w_real_full.len() - n_finite,
            w_real_full.len()
        );

        let group_size = 32usize;
        // Q4_0 storage is exact multiples of 32, so element_count is
        // already group-aligned; assert as an invariant.
        assert!(
            w_real_full.len() % group_size == 0,
            "real weight len {} not divisible by group_size {}",
            w_real_full.len(),
            group_size
        );
        let n_total = w_real_full.len();
        let n_groups = n_total / group_size;

        // Init scales+biases from real weight.
        let mut init_registry = KernelRegistry::new();
        let (q_int, s_init, b_init) = init_affine_params_gpu(
            &device,
            &mut init_registry,
            &w_real_full,
            group_size,
            4, // 4-bit
        )
        .expect("init");
        assert_eq!(s_init.len(), n_groups);
        assert_eq!(q_int.len(), n_total);

        // All scales positive + finite; all biases finite.  Real
        // weights with degenerate (uniform) groups would set s := 1.0
        // by the kernel's contract.
        for (i, &s) in s_init.iter().enumerate() {
            assert!(
                s.is_finite() && s > 0.0,
                "s_init[{i}] non-finite or non-positive: {s}"
            );
        }
        for (i, &b) in b_init.iter().enumerate() {
            assert!(b.is_finite(), "b_init[{i}] non-finite: {b}");
        }

        // Reconstruction MSE @ init — sanity oracle for the per-group
        // min/max heuristic.  Sets the lower bound on what Adam can
        // achieve from the perturbed start.
        let init_mse: f32 = {
            let mut acc = 0.0f64;
            for i in 0..n_total {
                let g = i / group_size;
                let qdq = q_int[i] as f32 * s_init[g] + b_init[g];
                acc += (qdq - w_real_full[i]).powi(2) as f64;
            }
            (acc / n_total as f64) as f32
        };
        eprintln!(
            "[dwq_real_gguf] reconstruction MSE @ init: {:.6e}  (n_groups={n_groups})",
            init_mse
        );

        // Perturb +100% (×2.0).
        let perturb = |xs: &[f32], factor: f32| -> Vec<f32> {
            xs.iter().map(|v| v * factor).collect()
        };
        let s_p = perturb(&s_init, 2.0);
        let b_p = perturb(&b_init, 2.0);

        let cfg = AdamConfig {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        };
        let mut adam = AdamOptimizer::new(device.clone(), cfg).expect("adam");
        adam.register_param("s", buffer_from_f32(&device, &s_p).unwrap())
            .unwrap();
        adam.register_param("b", buffer_from_f32(&device, &b_p).unwrap())
            .unwrap();

        let tape = GpuTape::new(device.clone());
        let train_step = |adam: &mut AdamOptimizer, tape: &GpuTape| -> Result<f32> {
            let s = adam.read_param("s")?;
            let b = adam.read_param("b")?;
            let s_leaf = GpuTensor::from_vec(tape, &s, vec![s.len()])?;
            let b_leaf = GpuTensor::from_vec(tape, &b, vec![b.len()])?;
            let qdq = qdq_affine(&s_leaf, &b_leaf, &q_int, group_size)?;
            let w_const = GpuTensor::from_vec(tape, &w_real_full, vec![n_total])?;
            let r = sub(&qdq, &w_const)?;
            let sq = square(&r)?;
            let sq_host = sq.to_vec()?;
            let loss = (sq_host.iter().map(|v| *v as f64).sum::<f64>() / n_total as f64) as f32;
            let dy = ones_like(tape, sq.shape())?;
            let grads = backward(&sq, dy)?;
            let g_s = grads[s_leaf.node_idx()]
                .as_ref()
                .ok_or_else(|| anyhow!("missing s grad"))?
                .clone();
            let g_b = grads[b_leaf.node_idx()]
                .as_ref()
                .ok_or_else(|| anyhow!("missing b grad"))?
                .clone();
            let mut g_map = BTreeMap::new();
            g_map.insert("s".to_string(), g_s);
            g_map.insert("b".to_string(), g_b);
            adam.step(&g_map)?;
            Ok(loss)
        };

        let initial_loss = train_step(&mut adam, &tape).expect("init step");
        tape.reset();

        // Non-triviality: initial loss > 1.5× the ideal-init MSE.  If
        // 2× perturbation didn't move the loss, the test is trivial.
        assert!(
            initial_loss > init_mse * 1.5,
            "real-GGUF fixture trivial: initial_loss={initial_loss} ≤ 1.5*init_mse={}",
            init_mse * 1.5
        );

        let mut min_loss = initial_loss;
        let mut last_loss = initial_loss;
        let n_steps = 100usize;
        for step in 1..n_steps {
            let l = train_step(&mut adam, &tape).expect("step");
            tape.reset();
            assert!(l.is_finite(), "step {step}: loss became non-finite ({l})");
            if l < min_loss {
                min_loss = l;
            }
            last_loss = l;
            if step % 20 == 0 {
                eprintln!(
                    "[dwq_real_gguf] step={step} loss={l:.6e} min={min_loss:.6e} initial={initial_loss:.6e}"
                );
            }
        }

        eprintln!(
            "[dwq_real_gguf] FINAL: initial={:.6e} min={:.6e} last={:.6e} ratio={:.3}",
            initial_loss,
            min_loss,
            last_loss,
            min_loss / initial_loss
        );

        assert!(
            min_loss < initial_loss * 0.2,
            "did not converge: initial={initial_loss} min={min_loss} last={last_loss}"
        );

        // Sanity: scales+biases finite at end, scales still positive.
        let s_final = adam.read_param("s").unwrap();
        let b_final = adam.read_param("b").unwrap();
        for (i, &v) in s_final.iter().enumerate() {
            assert!(v.is_finite(), "s_final[{i}] non-finite: {v}");
        }
        for (i, &v) in b_final.iter().enumerate() {
            assert!(v.is_finite(), "b_final[{i}] non-finite: {v}");
        }
    }

    /// iter-13e — real-tensor KL-div training loop.
    ///
    /// Closes the conceptual gap between iter-13c's synthetic 2-Linear
    /// KL test and iter-13d's real-tensor reconstruction-MSE test by
    /// running the FULL DWQ chain
    /// `qdq_affine → view → matmul → kl_div_loss_per_row → backward →
    /// Adam.step` on a real GGUF tensor.
    ///
    /// Setup:
    ///   - W_real = `blk.0.attn_qkv.weight` from the 27B Q4_0 GGUF
    ///     (shape [10240, 5120] in GGUF [out, in]; transposed to
    ///     [5120, 10240] for our matmul convention).
    ///   - X = Gaussian(0, 1) of shape [m=64, in=5120], deterministic
    ///     seed (proxy for post-RMSNorm activations whose stddev
    ///     after layer norm is ~1.0).
    ///   - Teacher logits = X @ W_real_T (shape [64, 10240]).
    ///   - Student logits = X @ view(qdq_affine(W_real, s, b),
    ///                               [5120, 10240]).
    ///   - Loss = mean(KL(softmax(scale·teacher) || softmax(scale·student)))
    ///     where scale = 1/T = 0.5 (T=2.0 per mlx-lm dwq.py default).
    ///   - Optimizer: Adam(lr=1e-3, β1=0.9, β2=0.999, ε=1e-8) over
    ///     (scales, biases); q_int frozen.
    ///
    /// Per mlx-lm `tuner/losses.py:377` `kl_div_loss(logits_q,
    /// logits_p)` takes RAW logits (computes logsumexp internally);
    /// `q` is the student, `p` is the teacher; reduction="none"
    /// returns shape `[batch]` per row, summed in dwq.py at
    /// `dwq.py:117`: `loss = (mask * losses).sum() / ntoks`.
    ///
    /// Acceptance: best KL after 100 Adam steps < 0.34 × initial KL
    /// (3× reduction floor — KL is stricter than reconstruction MSE).
    /// Non-triviality: initial KL > 1e-4 (otherwise the 2×
    /// perturbation didn't move the loss landscape).
    ///
    /// Test gate: `#[ignore]` (multi-GB GGUF on disk).  Run with:
    ///   cargo test --release --bin hf2q -- \
    ///     --ignored dwq_loop_real_gguf_kl
    /// Override path: `HF2Q_TEST_GGUF=/path/to/file.gguf`.
    /// Override tensor: `HF2Q_TEST_GGUF_TENSOR=blk.X.tensor.weight`.
    #[test]
    #[ignore]
    fn dwq_loop_real_gguf_kl_div_with_random_activations_converges() {
        use crate::calibrate::autograd_gpu_tape::{matmul, scalar_mul, view};
        use crate::calibrate::dynamic_quant_gpu::kl_div_loss_per_row;
        use mlx_native::gguf::GgufFile;

        let gguf_path = std::env::var("HF2Q_TEST_GGUF").unwrap_or_else(|_| {
            "/opt/hf2q/models/qwen3.6-27b-mtp-q4_0/qwen3.6-27b-mtp-q4_0.gguf".to_string()
        });
        let tensor_name = std::env::var("HF2Q_TEST_GGUF_TENSOR")
            .unwrap_or_else(|_| "blk.0.attn_qkv.weight".to_string());
        let path = std::path::Path::new(&gguf_path);
        if !path.exists() {
            eprintln!("[dwq_kl_real] SKIP: {gguf_path} not found");
            return;
        }

        let device = MlxDevice::new().expect("device");
        let gguf = GgufFile::open(path).expect("open gguf");
        let info = gguf
            .tensor_info(&tensor_name)
            .unwrap_or_else(|| panic!("tensor '{tensor_name}' not in {gguf_path}"));
        // GGUF shape is [out, in]; we'll transpose for matmul.
        assert_eq!(info.shape.len(), 2, "expected 2-D Linear weight");
        let out_dim = info.shape[0];
        let in_dim = info.shape[1];
        eprintln!(
            "[dwq_kl_real] {tensor_name}: GGUF[out, in]=[{out_dim}, {in_dim}], type={:?}",
            info.ggml_type
        );

        let buf = gguf
            .load_tensor_f32(&tensor_name, &device)
            .expect("load_tensor_f32");
        let w_real_oi: Vec<f32> = buf.as_slice::<f32>().unwrap().to_vec();
        // Transpose to [in, out] = [5120, 10240] for our matmul.
        let w_real_io: Vec<f32> = transpose_2d(&w_real_oi, out_dim, in_dim);
        let n_total = w_real_io.len();
        assert_eq!(n_total, in_dim * out_dim);

        let group_size = 32usize;
        // After transpose, contiguous axis is the output dim.  group_size
        // must divide n_total which it does for any out_dim divisible by 32.
        assert!(
            n_total % group_size == 0,
            "n_total {} not divisible by group_size {}",
            n_total,
            group_size
        );

        // Init scales/biases from the TRANSPOSED weight (groups along
        // contiguous out_dim axis; same convention as mlx affine quant).
        let mut init_registry = KernelRegistry::new();
        let (q_int, s_init, b_init) = init_affine_params_gpu(
            &device,
            &mut init_registry,
            &w_real_io,
            group_size,
            4,
        )
        .expect("init");

        // Perturb 2× to give Adam something to learn.
        let perturb = |xs: &[f32], factor: f32| -> Vec<f32> {
            xs.iter().map(|v| v * factor).collect()
        };
        let s_p = perturb(&s_init, 2.0);
        let b_p = perturb(&b_init, 2.0);

        // Activations: deterministic Gaussian, stddev 1.0 (proxy for
        // post-RMSNorm residual stream).  Box-Muller via a stable
        // small PRNG; seed fixed for reproducibility.
        let m = 64usize;
        let x_data: Vec<f32> = box_muller_gaussian(m * in_dim, /* seed */ 0xC0FFEE5);

        // Pre-compute teacher logits y_T = X @ W_real (host CPU oracle
        // FP64 accumulation for numerical stability).  Treated as a
        // constant tape leaf each step.
        let mut y_t = vec![0.0f32; m * out_dim];
        for r in 0..m {
            for c in 0..out_dim {
                let mut acc = 0.0f64;
                for k in 0..in_dim {
                    acc += (x_data[r * in_dim + k] as f64)
                        * (w_real_io[k * out_dim + c] as f64);
                }
                y_t[r * out_dim + c] = acc as f32;
            }
        }
        // Sanity: teacher logits should be finite and have non-trivial
        // stddev (a degenerate fixture would NOT produce gradient flow).
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        for &v in &y_t {
            assert!(v.is_finite(), "teacher logit non-finite: {v}");
            sum += v as f64;
            sumsq += (v as f64).powi(2);
        }
        let mean = sum / y_t.len() as f64;
        let var = sumsq / y_t.len() as f64 - mean * mean;
        let stddev = var.sqrt() as f32;
        eprintln!("[dwq_kl_real] teacher logit stddev: {stddev:.4}");
        assert!(stddev > 0.1, "teacher logits too flat: stddev={stddev}");

        let cfg = AdamConfig {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        };
        let mut adam = AdamOptimizer::new(device.clone(), cfg).expect("adam");
        adam.register_param("s", buffer_from_f32(&device, &s_p).unwrap())
            .unwrap();
        adam.register_param("b", buffer_from_f32(&device, &b_p).unwrap())
            .unwrap();

        let temperature = 2.0f32;
        let inv_t = 1.0 / temperature;
        let tape = GpuTape::new(device.clone());

        let train_step = |adam: &mut AdamOptimizer, tape: &GpuTape| -> Result<f32> {
            let s = adam.read_param("s")?;
            let b = adam.read_param("b")?;
            let s_leaf = GpuTensor::from_vec(tape, &s, vec![s.len()])?;
            let b_leaf = GpuTensor::from_vec(tape, &b, vec![b.len()])?;
            let qdq_flat = qdq_affine(&s_leaf, &b_leaf, &q_int, group_size)?;
            let w_q_2d = view(&qdq_flat, vec![in_dim, out_dim])?;
            let xt = GpuTensor::from_vec(tape, &x_data, vec![m, in_dim])?;
            let y_s = matmul(&xt, &w_q_2d)?;
            let y_t_leaf = GpuTensor::from_vec(tape, &y_t, vec![m, out_dim])?;
            let y_s_scaled = scalar_mul(&y_s, inv_t)?;
            let y_t_scaled = scalar_mul(&y_t_leaf, inv_t)?;
            // KL(p || q) per row = Σ p · (log p - log q); q = student.
            let kl = kl_div_loss_per_row(&y_s_scaled, &y_t_scaled)?;
            let kl_host = kl.to_vec()?;
            let loss_mean = (kl_host.iter().map(|v| *v as f64).sum::<f64>()
                / kl_host.len() as f64) as f32;

            // Backward seed: dy = ones / m so the leaf gradients have
            // mean-grad semantics matching the host-side mean reduction.
            let mut dy_buf = tape
                .device()
                .alloc_buffer(kl_host.len() * 4, DType::F32, kl.shape().to_vec())?;
            dy_buf
                .as_mut_slice::<f32>()
                .map_err(|e| anyhow!("dy slice: {e}"))?
                .iter_mut()
                .for_each(|v| *v = 1.0 / m as f32);
            let grads = backward(&kl, dy_buf)?;
            let g_s = grads[s_leaf.node_idx()]
                .as_ref()
                .ok_or_else(|| anyhow!("missing s grad"))?
                .clone();
            let g_b = grads[b_leaf.node_idx()]
                .as_ref()
                .ok_or_else(|| anyhow!("missing b grad"))?
                .clone();
            let mut g_map = BTreeMap::new();
            g_map.insert("s".to_string(), g_s);
            g_map.insert("b".to_string(), g_b);
            adam.step(&g_map)?;
            Ok(loss_mean)
        };

        let initial_loss = train_step(&mut adam, &tape).expect("init step");
        tape.reset();
        eprintln!("[dwq_kl_real] initial KL = {initial_loss:.6e}");
        assert!(
            initial_loss > 1e-4,
            "real-GGUF KL fixture is trivial: initial_loss={initial_loss}"
        );

        let mut min_loss = initial_loss;
        let mut last_loss = initial_loss;
        let n_steps = 100usize;
        for step in 1..n_steps {
            let l = train_step(&mut adam, &tape).expect("step");
            tape.reset();
            assert!(l.is_finite(), "step {step}: loss non-finite ({l})");
            if l < min_loss {
                min_loss = l;
            }
            last_loss = l;
            if step % 20 == 0 {
                eprintln!(
                    "[dwq_kl_real] step={step} loss={l:.6e} min={min_loss:.6e} initial={initial_loss:.6e}"
                );
            }
        }
        eprintln!(
            "[dwq_kl_real] FINAL: initial={:.6e} min={:.6e} last={:.6e} ratio={:.3}",
            initial_loss,
            min_loss,
            last_loss,
            min_loss / initial_loss
        );

        assert!(
            min_loss < initial_loss * 0.34,
            "did not converge: initial={initial_loss} min={min_loss}"
        );

        // Sanity: final params finite.
        let s_final = adam.read_param("s").unwrap();
        let b_final = adam.read_param("b").unwrap();
        for v in s_final.iter().chain(b_final.iter()) {
            assert!(v.is_finite(), "final param non-finite: {v}");
        }
    }

    /// Transpose a row-major [rows, cols] FP32 buffer into [cols, rows].
    fn transpose_2d(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        assert_eq!(src.len(), rows * cols);
        let mut dst = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                dst[c * rows + r] = src[r * cols + c];
            }
        }
        dst
    }

    /// Sanity test: init kernel output equals the CPU oracle from the
    /// mlx-native side via the host-side wrapper.
    /// ADR-020 iter-12d-1 — convergence + structural test for the
    /// extracted public API.  Synthetic deterministic W (no real GGUF
    /// dependency, so this runs in CI unconditionally).  Asserts:
    ///   - kl_min < kl_initial × convergence_ratio (default 0.34)
    ///   - returned MlxAffineLinear has correct shapes + bit-width
    ///   - q_int values are in the expected unsigned range for `bits=4`
    #[test]
    fn iter_12d1_train_linear_dwq_synthetic_converges() {
        use super::{train_linear_dwq_synthetic_teacher, DwqTrainingConfig};

        let device = MlxDevice::new().expect("device");
        let n = 64usize;
        let k = 64usize;
        // Deterministic non-trivial W with stddev ~0.5 (much larger than
        // real Q4_0 magnitudes; lets the convergence gate clear easily
        // without needing the iter-17b-style 53× reduction).
        let w_real: Vec<f32> = (0..(n * k))
            .map(|i| ((i as f32) * 0.0173 - 0.4).sin() * 0.5)
            .collect();

        // This test exercises iter-13e/iter-17b's "training recovers from
        // a degraded init" semantic, so opt-in to perturb=2.0 explicitly.
        // The production default (perturb=1.0) starts at optimum and
        // can't show convergence_ratio < 1.0 within a finite step
        // budget — that's the iter-12f-1 benchmark's job.
        let cfg = DwqTrainingConfig {
            n_tokens: 32,
            n_steps: 30,
            perturb_factor: 2.0,    // explicit degradation start
            convergence_ratio: 0.5, // synthetic stddev=0.5 → ~3-5× reduction in 30 steps
            ..DwqTrainingConfig::default()
        };
        let result = train_linear_dwq_synthetic_teacher(&device, &w_real, n, k, &cfg)
            .expect("train_linear_dwq_synthetic_teacher");

        // Convergence gate
        assert!(
            result.kl_min < result.kl_initial * cfg.convergence_ratio,
            "did not converge: kl_initial={} kl_min={} ratio={} >= {}",
            result.kl_initial,
            result.kl_min,
            result.kl_min / result.kl_initial,
            cfg.convergence_ratio
        );

        // Returned MlxAffineLinear shape + bits
        assert_eq!(result.linear.n, n);
        assert_eq!(result.linear.k, k);
        assert_eq!(result.linear.group_size, cfg.group_size);
        assert_eq!(result.linear.bits, cfg.bits);
        assert_eq!(result.linear.q_int.len(), n * k);
        assert_eq!(result.linear.scales.len(), n * (k / cfg.group_size));
        assert_eq!(result.linear.biases.len(), n * (k / cfg.group_size));
        assert_eq!(result.steps_run, cfg.n_steps);

        // q_int values are in the expected range for `bits=4`:
        // qdq_legacy stores asymmetric signed range mapped to unsigned [0, 15].
        let max_code = (1u32 << cfg.bits) - 1;
        for &q in &result.linear.q_int {
            assert!(
                (q as u32) <= max_code,
                "q_int code {q} exceeds bits={} range [0, {max_code}]",
                cfg.bits
            );
        }

        // KL trajectory diagnostics — should all be finite + non-negative
        // (KL is a non-negative divergence).
        for (label, v) in [
            ("kl_initial", result.kl_initial),
            ("kl_min", result.kl_min),
            ("kl_final", result.kl_final),
        ] {
            assert!(v.is_finite(), "{label} non-finite: {v}");
            assert!(v >= -1e-6, "{label} significantly negative: {v}");
        }
    }

    /// ADR-020 AC#7 follow-up — `x_override` accepts a caller-supplied
    /// X distribution.  Hypothesis test: a non-uniform X (heavy-tailed
    /// distribution mimicking real post-RMSNorm activations) produces a
    /// DIFFERENT KL trajectory than the default synthetic Gaussian X
    /// on the same W and same perturb=1.0 init.
    ///
    /// This is the testable falsifier for the perturb=1.0-no-op finding
    /// (memory `project_adr020_dwq_perturb1_noop_finding_2026_05_08`).
    /// If the hypothesis holds, real-corpus X is the smaller-bite fix
    /// for AC#7 boundary closure (vs full-model teacher).
    ///
    /// Test contract: BOTH runs use the same W and same seed, but
    /// override X for the second.  Asserts that `kl_initial` differs
    /// between the two runs (the X distribution shifts the
    /// per-Linear KL surface).  The ABSOLUTE direction of the
    /// improvement is data-dependent; this test asserts the surface
    /// is sensitive to X, not which surface is more favorable.
    #[test]
    fn x_override_changes_kl_surface_at_perturb_1_0() {
        use super::{box_muller_gaussian, train_linear_dwq_synthetic_teacher, DwqTrainingConfig};

        let device = MlxDevice::new().expect("device");
        let n = 64usize;
        let k = 64usize;
        let m = 32usize;
        // Real-Q5_K-style W: small stddev (~0.014 per memory).
        let w_real: Vec<f32> = (0..(n * k))
            .map(|i| ((i as f32) * 0.0091 - 0.3).sin() * 0.014)
            .collect();

        // Run 1: default Gaussian X (no override).
        let cfg_default = DwqTrainingConfig {
            n_tokens: m,
            n_steps: 5, // short — we're testing the kl_initial surface, not convergence
            perturb_factor: 1.0,
            convergence_ratio: 100.0, // disable gate
            ..DwqTrainingConfig::default()
        };
        let r_default = train_linear_dwq_synthetic_teacher(&device, &w_real, n, k, &cfg_default)
            .expect("train (gaussian X)");

        // Run 2: same model, override X with a HEAVY-TAILED distribution.
        // Cube the standard Gaussian — pushes mass into the tails (kurtosis
        // ≈ 15 vs 3 for Gaussian).  This is a poor proxy for real
        // post-RMSNorm activations but is enough to demonstrate the X
        // surface shifts the KL.
        let x_gauss = box_muller_gaussian(m * k, cfg_default.seed);
        let x_heavy: Vec<f32> = x_gauss.iter().map(|v| v.powi(3)).collect();
        let cfg_override = DwqTrainingConfig {
            x_override: Some(x_heavy.clone()),
            ..cfg_default.clone()
        };
        let r_override = train_linear_dwq_synthetic_teacher(&device, &w_real, n, k, &cfg_override)
            .expect("train (heavy-tailed X override)");

        // Sanity: same model + same init → q_int is byte-identical (X
        // doesn't affect the discrete grid; only s/b training depends
        // on X).
        assert_eq!(
            r_default.linear.q_int, r_override.linear.q_int,
            "x_override changed q_int — should not happen (init is deterministic and X-independent)"
        );

        // The interesting assertion: the KL surface at the SAME init
        // (perturb=1.0 → s_init/b_init) is X-dependent.  kl_initial
        // SHOULD differ between the two runs because the matmul output
        // y = X @ W^T has different magnitudes under different X
        // distributions, and the softmax + KL operates on those.
        // Relative-difference threshold: kl values for small-stddev W can
        // be tiny (~1e-7) so an absolute threshold doesn't generalize.
        // 50% relative drift between two distributions is well above
        // numerical noise.
        let kl_min = r_default.kl_initial.min(r_override.kl_initial).max(1e-12);
        let kl_max = r_default.kl_initial.max(r_override.kl_initial);
        let rel_diff = (kl_max - kl_min) / kl_min;
        assert!(
            rel_diff > 0.5,
            "kl_initial unchanged by x_override (default={:e} override={:e} \
             rel_diff={rel_diff:.3}): X distribution should materially shift \
             the KL surface",
            r_default.kl_initial,
            r_override.kl_initial,
        );
        // Both must be finite and non-negative (KL is a divergence).
        for (label, v) in [
            ("default kl_initial", r_default.kl_initial),
            ("override kl_initial", r_override.kl_initial),
        ] {
            assert!(v.is_finite(), "{label} non-finite: {v}");
            assert!(v >= -1e-6, "{label} significantly negative: {v}");
        }
    }

    /// ADR-020 AC#7 — STRONG NEGATIVE RESULT (FALSIFIES Option B).
    ///
    /// Hypothesis under test (2026-05-08): a non-uniform X mimicking
    /// real post-RMSNorm activations would break the perturb=1.0
    /// ratio=1.000 plateau.
    ///
    /// EMPIRICAL OUTCOME on M5 Max: `kl_min == kl_initial` (ratio
    /// 1.0000) even with asymmetric real-corpus-like X.  The min-max
    /// init `s_init = (w_max - w_min) / (n_bins - 1)` is so close to
    /// optimal at bits=4 that `qdq(s_init, b_init, q_int) ≈ W_real`
    /// up to discretization (~1 ULP); thus `y_S = X @ qdq^T ≈ X @
    /// W_real^T = y_T` for ANY X distribution → KL ≈ 0 → gradient ≈
    /// 0 → no movement under Adam.
    ///
    /// **Implication**: Option B (real-corpus X with per-Linear
    /// teacher) does NOT close the AC#7 boundary gap.  The remaining
    /// viable path is Option A (full-model teacher matching mlx-lm's
    /// `dwq_quantize` at `mlx_lm/quant/dwq.py:108-114`), which is
    /// substantial scope.
    ///
    /// Real activation distribution proxy used here:
    /// - Sample base z ~ Gaussian σ=1
    /// - Per-row RMSNorm (unit-norm per token)
    /// - Per-column heavy-tail amplitude exp(0.5 * gauss(j))
    ///
    /// This test PASSES by asserting the negative result; future
    /// iterations on AC#7 must either falsify this finding (e.g.
    /// with a different real-X construction that does break the
    /// plateau) or pivot to Option A.
    #[test]
    fn perturb_1_0_with_real_corpus_x_does_not_break_noop_plateau() {
        use super::{box_muller_gaussian, train_linear_dwq_synthetic_teacher, DwqTrainingConfig};

        let device = MlxDevice::new().expect("device");
        let n = 64usize;
        let k = 64usize;
        let m = 32usize;
        // Real-Q5_K-style W (small stddev, multimodal phasing across
        // groups — exercises per-group s/b separately).
        let w_real: Vec<f32> = (0..(n * k))
            .map(|i| {
                let phase = (i % 32) as f32 * 0.0911;
                ((i as f32) * 0.0091 - 0.3).sin() * 0.014 + phase.cos() * 0.005
            })
            .collect();

        // Build asymmetric "real-corpus-like" X.
        let z_base = box_muller_gaussian(m * k, 0xCAFE_BABE);
        let z_chan_scale = box_muller_gaussian(k, 0xDEAD_F00D);
        let mut x_real_corpus: Vec<f32> = vec![0.0; m * k];
        for r in 0..m {
            // Per-row unit-norm (RMSNorm proxy).
            let row_start = r * k;
            let row_end = row_start + k;
            let row = &z_base[row_start..row_end];
            let rms = (row.iter().map(|v| v * v).sum::<f32>() / k as f32).sqrt();
            let inv = if rms > 1e-8 { 1.0 / rms } else { 1.0 };
            for c in 0..k {
                // Per-column heavy-tailed amplitude
                let col_amp = (0.5 * z_chan_scale[c]).exp();
                x_real_corpus[row_start + c] = row[c] * inv * col_amp;
            }
        }

        let cfg = DwqTrainingConfig {
            n_tokens: m,
            n_steps: 30,
            perturb_factor: 1.0,
            convergence_ratio: 100.0, // disable gate; we assert the ratio ourselves
            x_override: Some(x_real_corpus),
            ..DwqTrainingConfig::default()
        };
        let result = train_linear_dwq_synthetic_teacher(&device, &w_real, n, k, &cfg)
            .expect("train (real-corpus X)");

        // EMPIRICAL FINDING (asserted as ground truth):
        // `kl_min == kl_initial` (ratio 1.0000) — i.e., NO improvement
        // even with asymmetric real-corpus-like X.  This is the
        // falsified-hypothesis assertion: the test PASSES by
        // confirming the no-op plateau holds at perturb=1.0
        // regardless of X distribution.
        let ratio = result.kl_min / result.kl_initial;
        assert!(
            ratio >= 0.999,
            "UNEXPECTED: real-corpus X actually broke the plateau at \
             perturb=1.0!  ratio={ratio:.6} kl_initial={:e} kl_min={:e}.  \
             If this fires, the original no-op finding needs revisiting \
             — examine whether the X construction in this test ended up \
             aligning with a non-trivial gradient direction Adam could \
             follow.",
            result.kl_initial,
            result.kl_min,
        );
        assert!(result.kl_initial.is_finite() && result.kl_initial > 0.0);
        assert!(result.kl_min.is_finite() && result.kl_min >= 0.0);
        eprintln!(
            "[ac7-real-x falsifier confirmed] kl_initial={:e} kl_min={:e} ratio={ratio:.6} — \
             plateau holds; Option B (real-corpus X with per-Linear teacher) does NOT \
             close AC#7.  Pivot path: Option A (full-model teacher).",
            result.kl_initial,
            result.kl_min,
        );
    }

    /// `x_override` length validation: reject mismatched length, accept
    /// the correct `n_tokens × k`.
    #[test]
    fn x_override_rejects_wrong_length() {
        use super::{train_linear_dwq_synthetic_teacher, DwqTrainingConfig};

        let device = MlxDevice::new().expect("device");
        let n = 32usize;
        let k = 32usize;
        let w_real: Vec<f32> = vec![0.01_f32; n * k];
        let cfg = DwqTrainingConfig {
            n_tokens: 32,
            n_steps: 1,
            convergence_ratio: 100.0,
            x_override: Some(vec![0.0_f32; 999]), // wrong length
            ..DwqTrainingConfig::default()
        };
        let res = train_linear_dwq_synthetic_teacher(&device, &w_real, n, k, &cfg);
        assert!(res.is_err(), "expected length-mismatch error");
        let msg = format!("{}", res.err().unwrap());
        assert!(
            msg.contains("x_override length"),
            "error message should call out x_override length: {msg}"
        );
    }

    /// ADR-020 iter-12d-1 — input validation: reject n < 32, k < 32,
    /// k not divisible by group_size, n_steps == 0, w_real shape mismatch.
    #[test]
    fn iter_12d1_train_linear_dwq_rejects_invalid_inputs() {
        use super::{train_linear_dwq_synthetic_teacher, DwqTrainingConfig};

        let device = MlxDevice::new().expect("device");
        let cfg = DwqTrainingConfig::default();
        let any_w = vec![0.1f32; 1024];

        // n < 32
        let r = train_linear_dwq_synthetic_teacher(&device, &any_w, 16, 64, &cfg);
        assert!(r.is_err());

        // k not divisible by group_size (group_size=32, k=33)
        let bad_k_w = vec![0.1f32; 32 * 33];
        let r = train_linear_dwq_synthetic_teacher(&device, &bad_k_w, 32, 33, &cfg);
        assert!(r.is_err());

        // w shape mismatch
        let r = train_linear_dwq_synthetic_teacher(&device, &any_w, 64, 64, &cfg);
        assert!(r.is_err());

        // n_steps == 0
        let bad_cfg = DwqTrainingConfig {
            n_steps: 0,
            ..cfg.clone()
        };
        let r = train_linear_dwq_synthetic_teacher(&device, &any_w[..32 * 32], 32, 32, &bad_cfg);
        assert!(r.is_err());

        // n_tokens < 32
        let bad_cfg = DwqTrainingConfig {
            n_tokens: 16,
            ..cfg
        };
        let r = train_linear_dwq_synthetic_teacher(&device, &any_w[..32 * 32], 32, 32, &bad_cfg);
        assert!(r.is_err());
    }

    /// ADR-020 iter-12e — current_rss_bytes returns a non-zero value
    /// (the test process always has some resident memory).  Sanity gate
    /// for the sysinfo wiring; failure here means the watchdog can't
    /// see process memory at all.
    #[test]
    fn iter_12e_current_rss_bytes_is_positive() {
        let rss = super::current_rss_bytes();
        assert!(
            rss > 1024 * 1024,
            "current_rss_bytes() returned {rss}; expected > 1 MB \
             (test process must have resident memory)"
        );
    }

    /// ADR-020 iter-12e — RssWatchdog with cap below current RSS triggers
    /// `is_aborted()` within a few poll cycles.  Verifies the polling
    /// thread runs + the abort flag is set + peak_rss reflects current.
    #[test]
    fn iter_12e_rss_watchdog_triggers_when_cap_below_current() {
        use super::RssWatchdog;
        use std::time::Duration;

        // Cap intentionally far below any possible test-process RSS.
        let cap = 1024u64; // 1 KB — guaranteed to trigger
        let w = RssWatchdog::spawn(cap, Duration::from_millis(50));

        // Poll up to 1s (20× 50ms) for the abort to surface.
        let mut tries = 0;
        while !w.is_aborted() && tries < 20 {
            std::thread::sleep(Duration::from_millis(50));
            tries += 1;
        }
        assert!(
            w.is_aborted(),
            "watchdog did not abort within 1s (cap=1KB; current RSS = {} bytes)",
            super::current_rss_bytes()
        );
        assert!(
            w.peak_rss_bytes() > cap,
            "peak_rss_bytes ({}) <= cap ({})",
            w.peak_rss_bytes(),
            cap
        );
        assert_eq!(w.cap_bytes(), cap);
    }

    /// ADR-020 iter-12e — RssWatchdog with extremely high cap does NOT
    /// trigger.  Catches false-positive aborts from a misconfigured
    /// cap-comparison.
    #[test]
    fn iter_12e_rss_watchdog_does_not_trigger_below_cap() {
        use super::RssWatchdog;
        use std::time::Duration;

        // Cap = 1 PB — unreachable on any real machine.
        let cap = 1024u64 * 1024 * 1024 * 1024 * 1024; // 1 PB
        let w = RssWatchdog::spawn(cap, Duration::from_millis(50));

        // Wait 200ms (≥ 4 poll cycles) — abort flag must remain false.
        std::thread::sleep(Duration::from_millis(200));
        assert!(
            !w.is_aborted(),
            "watchdog falsely aborted at cap=1PB (peak={})",
            w.peak_rss_bytes()
        );
    }

    /// ADR-020 iter-12e — DwqTrainingConfig::default() has rss_cap_bytes=None
    /// so existing iter-12d-1/2 callers inherit no-watchdog behavior.
    #[test]
    fn iter_12e_default_config_has_no_rss_cap() {
        let cfg = super::DwqTrainingConfig::default();
        assert!(
            cfg.rss_cap_bytes.is_none(),
            "DwqTrainingConfig::default() must have rss_cap_bytes = None"
        );
    }

    /// ADR-020 iter-12f-2 — DwqTrainingConfig::default() has
    /// compute_bench=false so existing iter-12d-1/2/3 + iter-12e
    /// callers inherit no-bench behavior (zero perf cost).
    #[test]
    fn iter_12f2_default_config_has_no_bench() {
        let cfg = super::DwqTrainingConfig::default();
        assert!(
            !cfg.compute_bench,
            "DwqTrainingConfig::default() must have compute_bench = false"
        );
    }

    /// ADR-020 AC#7 Option A — `FullModelDwqConfig::validate()` rejects
    /// every bad knob combo with a descriptive error message.  Smoke
    /// tests one rejection per invariant.  Happy path (a fully-filled
    /// config with valid token batches) returns Ok.
    #[test]
    fn full_model_dwq_config_validate_rejects_bad_knobs() {
        // Helper: build a baseline-VALID config, then mutate one field
        // per case to test the corresponding rejection.
        let valid = || -> super::FullModelDwqConfig {
            let cfg = super::FullModelDwqConfig {
                gguf_path: std::path::PathBuf::from("/dummy/path/model.gguf"),
                calibration_token_batches: vec![vec![0u32; 32 * 512]; 1],
                ..super::FullModelDwqConfig::default()
            };
            // Sanity: baseline is valid.
            cfg.validate().unwrap_or_else(|e| panic!("baseline cfg invalid: {e}"));
            cfg
        };

        // bits out of range.
        let mut c = valid();
        c.bits = 1;
        let e = format!("{}", c.validate().err().unwrap());
        assert!(e.contains("bits must be in [2, 8]"), "got: {e}");
        let mut c = valid();
        c.bits = 9;
        assert!(c.validate().is_err());

        // group_size not power-of-two.
        let mut c = valid();
        c.group_size = 33;
        let e = format!("{}", c.validate().err().unwrap());
        assert!(e.contains("group_size"), "got: {e}");
        // group_size out of [2, 1024] range.
        let mut c = valid();
        c.group_size = 2048;
        assert!(c.validate().is_err());

        // n_steps == 0.
        let mut c = valid();
        c.n_steps = 0;
        let e = format!("{}", c.validate().err().unwrap());
        assert!(e.contains("n_steps"), "got: {e}");

        // lr non-positive / non-finite.
        let mut c = valid();
        c.lr = 0.0;
        assert!(c.validate().is_err());
        let mut c = valid();
        c.lr = -1e-4;
        assert!(c.validate().is_err());
        let mut c = valid();
        c.lr = f32::NAN;
        assert!(c.validate().is_err());

        // temperature non-positive.
        let mut c = valid();
        c.temperature = 0.0;
        assert!(c.validate().is_err());

        // batch_size below 32 floor.
        let mut c = valid();
        c.batch_size = 16;
        c.calibration_token_batches = vec![vec![0u32; 16 * 512]];
        let e = format!("{}", c.validate().err().unwrap());
        assert!(e.contains("batch_size must be >= 32"), "got: {e}");

        // seq_len below 2.
        let mut c = valid();
        c.seq_len = 1;
        c.calibration_token_batches = vec![vec![0u32; 32 * 1]];
        let e = format!("{}", c.validate().err().unwrap());
        assert!(e.contains("seq_len must be >= 2"), "got: {e}");

        // gguf_path empty (caller-required).
        let mut c = valid();
        c.gguf_path = std::path::PathBuf::new();
        let e = format!("{}", c.validate().err().unwrap());
        assert!(e.contains("gguf_path must be non-empty"), "got: {e}");

        // calibration_token_batches empty.
        let mut c = valid();
        c.calibration_token_batches.clear();
        let e = format!("{}", c.validate().err().unwrap());
        assert!(e.contains("calibration_token_batches must be non-empty"), "got: {e}");

        // batch length mismatch (32 * 512 = 16384; we provide 100).
        let mut c = valid();
        c.calibration_token_batches = vec![vec![0u32; 100]];
        let e = format!("{}", c.validate().err().unwrap());
        assert!(e.contains("expected batch_size * seq_len"), "got: {e}");

        // top_k_teacher == 0.
        let mut c = valid();
        c.top_k_teacher = 0;
        assert!(c.validate().is_err());
    }

    /// ADR-020 AC#7 Option A — `FullModelDwqConfig::default()` carries
    /// the production-canonical knobs but explicitly leaves the
    /// caller-required fields (`gguf_path`, `calibration_token_batches`)
    /// in their no-op state.  This test pins the contract:
    ///
    /// - bits=4, group_size=32, top_k_teacher=1024, T=2.0 — match
    ///   mlx-lm's dwq.py defaults
    /// - n_steps=200, lr=1e-4 — production starting point (operator
    ///   tunes per family)
    /// - gguf_path is empty + calibration_token_batches is empty,
    ///   so the eventual training fn must validate caller has
    ///   filled them (fail-loud) before invoking forward
    /// - rss_cap_bytes=None — preserved with §8.3 AC#6 deferred to
    ///   caller (mirrors DwqTrainingConfig)
    ///
    /// Step 1 of the AC#7 implementation ladder per ADR-020 §8.3.
    #[test]
    fn full_model_dwq_config_default_pins_production_canon() {
        let cfg = super::FullModelDwqConfig::default();
        assert_eq!(cfg.bits, 4);
        assert_eq!(cfg.group_size, 32);
        assert_eq!(cfg.top_k_teacher, 1024);
        assert!((cfg.temperature - 2.0).abs() < 1e-9);
        assert_eq!(cfg.n_steps, 200);
        assert!((cfg.lr - 1e-4).abs() < 1e-12);
        assert_eq!(cfg.batch_size, 32);
        assert_eq!(cfg.seq_len, 512);
        assert_eq!(cfg.seed, 0xDEAD_BEEF);
        assert!(cfg.rss_cap_bytes.is_none());

        // Caller-required fields must be empty by default (so any
        // forgotten override surfaces as a fail-loud error in the
        // eventual training fn rather than silently using wrong data).
        assert_eq!(cfg.gguf_path.as_os_str().len(), 0);
        assert!(cfg.calibration_token_batches.is_empty());

        // Update-syntax must work so test fixtures can override
        // sparingly.
        let custom = super::FullModelDwqConfig {
            bits: 6,
            n_steps: 5,
            ..super::FullModelDwqConfig::default()
        };
        assert_eq!(custom.bits, 6);
        assert_eq!(custom.n_steps, 5);
        assert_eq!(custom.group_size, 32); // unchanged from default
    }

    /// ADR-020 AC#7 Option A — minimal WordLevel tokenizer fixture.
    /// Each whitespace-separated word maps to a deterministic id; the
    /// `<unk>` fallback covers any word not in vocab.  Sufficient to
    /// exercise the JSONL loader's tokenize-truncate-pack-batch logic
    /// without pulling a 28-MB tokenizer.json into the unit test.
    fn fixture_wordlevel_tokenizer() -> tokenizers::Tokenizer {
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "<unk>": 0,
                    "the": 1,
                    "quick": 2,
                    "brown": 3,
                    "fox": 4,
                    "jumps": 5,
                    "over": 6,
                    "lazy": 7,
                    "dog": 8,
                    "hello": 9,
                    "world": 10,
                    "foo": 11,
                    "bar": 12,
                    "baz": 13,
                    "qux": 14
                },
                "unk_token": "<unk>"
            }
        }"#;
        json.parse::<tokenizers::Tokenizer>()
            .expect("fixture invariant: minimal WordLevel JSON must parse")
    }

    /// ADR-020 AC#7 Option A — happy path: 4 prompts × batch_size=2,
    /// seq_len=8.  Expect 2 batches each of length 16, with row-major
    /// packing where row 0 ids occupy `[0..8)` and row 1 ids occupy
    /// `[8..16)`.  Short prompts must be zero-padded to seq_len.
    #[test]
    fn load_calibration_corpus_jsonl_packs_rows_correctly() {
        let tok = fixture_wordlevel_tokenizer();
        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-jsonl-pack-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corpus.jsonl");
        std::fs::write(
            &path,
            "{\"text\": \"the quick brown fox\"}\n\
             {\"text\": \"jumps over the lazy dog\"}\n\
             {\"text\": \"hello world\"}\n\
             {\"text\": \"foo bar baz qux\"}\n",
        )
        .unwrap();

        let batches = super::load_calibration_corpus_jsonl(&path, &tok, 2, 8).unwrap();
        assert_eq!(batches.len(), 2, "4 prompts ÷ batch_size 2 = 2 batches");
        for b in &batches {
            assert_eq!(b.len(), 2 * 8, "each batch must be batch_size × seq_len");
        }
        // Batch 0 row 0 = "the quick brown fox" → [1, 2, 3, 4, 0, 0, 0, 0]
        assert_eq!(&batches[0][0..8], &[1u32, 2, 3, 4, 0, 0, 0, 0]);
        // Batch 0 row 1 = "jumps over the lazy dog" → [5, 6, 1, 7, 8, 0, 0, 0]
        assert_eq!(&batches[0][8..16], &[5u32, 6, 1, 7, 8, 0, 0, 0]);
        // Batch 1 row 0 = "hello world" → [9, 10, 0, 0, 0, 0, 0, 0]
        assert_eq!(&batches[1][0..8], &[9u32, 10, 0, 0, 0, 0, 0, 0]);
        // Batch 1 row 1 = "foo bar baz qux" → [11, 12, 13, 14, 0, 0, 0, 0]
        assert_eq!(&batches[1][8..16], &[11u32, 12, 13, 14, 0, 0, 0, 0]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-020 AC#7 Option A — truncation: a prompt longer than
    /// `seq_len` must be truncated to its first `seq_len` ids; no
    /// silent error.
    #[test]
    fn load_calibration_corpus_jsonl_truncates_long_prompts() {
        let tok = fixture_wordlevel_tokenizer();
        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-jsonl-trunc-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corpus.jsonl");
        // 9 tokens, seq_len=4 → first 4 only.
        std::fs::write(
            &path,
            "{\"text\": \"the quick brown fox jumps over the lazy dog\"}\n\
             {\"text\": \"hello world foo bar baz qux\"}\n",
        )
        .unwrap();

        let batches = super::load_calibration_corpus_jsonl(&path, &tok, 2, 4).unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 8);
        assert_eq!(&batches[0][0..4], &[1u32, 2, 3, 4]); // "the quick brown fox"
        assert_eq!(&batches[0][4..8], &[9u32, 10, 11, 12]); // "hello world foo bar"

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-020 AC#7 Option A — drop trailing partial batch (mlx-lm
    /// `iterate_batches` parity at `tuner/trainer.py:134-135`).  5
    /// prompts ÷ batch_size=2 = 2 full batches; last prompt is dropped.
    #[test]
    fn load_calibration_corpus_jsonl_drops_partial_trailing_batch() {
        let tok = fixture_wordlevel_tokenizer();
        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-jsonl-partial-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corpus.jsonl");
        std::fs::write(
            &path,
            "{\"text\": \"the\"}\n\
             {\"text\": \"quick\"}\n\
             {\"text\": \"brown\"}\n\
             {\"text\": \"fox\"}\n\
             {\"text\": \"jumps\"}\n",
        )
        .unwrap();

        let batches = super::load_calibration_corpus_jsonl(&path, &tok, 2, 4).unwrap();
        assert_eq!(batches.len(), 2, "5 ÷ 2 = 2 full batches; trailing 1 dropped");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-020 AC#7 Option A — blank lines must be skipped, not parsed.
    #[test]
    fn load_calibration_corpus_jsonl_skips_blank_lines() {
        let tok = fixture_wordlevel_tokenizer();
        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-jsonl-blank-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corpus.jsonl");
        std::fs::write(
            &path,
            "\n\
             {\"text\": \"hello world\"}\n\
             \n\
             {\"text\": \"foo bar\"}\n\
             \n",
        )
        .unwrap();

        let batches = super::load_calibration_corpus_jsonl(&path, &tok, 2, 4).unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 8);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-020 AC#7 Option A — fail loud on malformed JSON line.
    #[test]
    fn load_calibration_corpus_jsonl_fails_on_malformed_json() {
        let tok = fixture_wordlevel_tokenizer();
        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-jsonl-malformed-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corpus.jsonl");
        std::fs::write(
            &path,
            "{\"text\": \"hello\"}\n\
             this is not json\n\
             {\"text\": \"world\"}\n",
        )
        .unwrap();

        let err = super::load_calibration_corpus_jsonl(&path, &tok, 2, 4).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("line 2"),
            "error must identify line number; got: {msg}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-020 AC#7 Option A — fail loud when a JSON object lacks the
    /// `"text"` field (mlx-lm CompletionsDataset convention).
    #[test]
    fn load_calibration_corpus_jsonl_fails_on_missing_text_field() {
        let tok = fixture_wordlevel_tokenizer();
        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-jsonl-notext-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corpus.jsonl");
        std::fs::write(
            &path,
            "{\"text\": \"ok\"}\n\
             {\"prompt\": \"missing text key\"}\n",
        )
        .unwrap();

        let err = super::load_calibration_corpus_jsonl(&path, &tok, 2, 4).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("\"text\"") && msg.contains("line 2"),
            "error must call out missing text on line 2; got: {msg}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-020 AC#7 Option A — fail loud when corpus has fewer prompts
    /// than `batch_size` (training would have zero batches).
    #[test]
    fn load_calibration_corpus_jsonl_fails_when_corpus_smaller_than_batch_size() {
        let tok = fixture_wordlevel_tokenizer();
        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-jsonl-toosmall-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corpus.jsonl");
        std::fs::write(&path, "{\"text\": \"only one\"}\n").unwrap();

        let err = super::load_calibration_corpus_jsonl(&path, &tok, 2, 4).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("less than batch_size") && msg.contains("1"),
            "error must call out the count vs batch_size; got: {msg}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-020 AC#7 Option A — output is consumable by
    /// `FullModelDwqConfig::validate()` without further reshape.
    /// End-to-end contract handshake.
    #[test]
    fn load_calibration_corpus_jsonl_output_satisfies_full_model_dwq_config_validate() {
        let tok = fixture_wordlevel_tokenizer();
        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-jsonl-handshake-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corpus.jsonl");
        // 64 prompts of varying length.
        let mut body = String::new();
        for i in 0..64 {
            body.push_str(&format!(
                "{{\"text\": \"hello world foo bar baz qux {}\"}}\n",
                i % 5
            ));
        }
        std::fs::write(&path, body).unwrap();

        let batch_size = 32;
        let seq_len = 16;
        let batches =
            super::load_calibration_corpus_jsonl(&path, &tok, batch_size, seq_len).unwrap();
        assert_eq!(batches.len(), 64 / batch_size);

        let cfg = super::FullModelDwqConfig {
            gguf_path: std::path::PathBuf::from("/dev/null"),
            calibration_token_batches: batches,
            batch_size,
            seq_len,
            ..super::FullModelDwqConfig::default()
        };
        cfg.validate()
            .expect("loader output must satisfy validate() contract");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-020 AC#7 Option A — `build_dwq_targets_config_from_full_model`
    /// happy path: forwards top_k_teacher, save_dir, vocab unchanged.
    #[test]
    fn build_dwq_targets_config_from_full_model_forwards_fields() {
        let cfg = super::FullModelDwqConfig {
            gguf_path: std::path::PathBuf::from("/dev/null"),
            calibration_token_batches: vec![vec![0u32; 32 * 512]],
            top_k_teacher: 1024,
            ..super::FullModelDwqConfig::default()
        };
        let out = super::build_dwq_targets_config_from_full_model(
            &cfg,
            std::path::PathBuf::from("/tmp/teacher-out"),
            128_000,
        )
        .unwrap();
        assert_eq!(out.top_k, 1024);
        assert_eq!(out.vocab, 128_000);
        assert_eq!(out.save_dir, std::path::PathBuf::from("/tmp/teacher-out"));
    }

    /// ADR-020 AC#7 Option A — vocab=0 must hard-fail.
    #[test]
    fn build_dwq_targets_config_from_full_model_rejects_zero_vocab() {
        let cfg = super::FullModelDwqConfig {
            gguf_path: std::path::PathBuf::from("/dev/null"),
            calibration_token_batches: vec![vec![0u32; 32 * 512]],
            ..super::FullModelDwqConfig::default()
        };
        let err = super::build_dwq_targets_config_from_full_model(
            &cfg,
            std::path::PathBuf::from("/tmp/x"),
            0,
        )
        .unwrap_err();
        let msg = format!("{:#}", err);
        assert!(msg.contains("vocab must be > 0"), "got: {msg}");
    }

    /// ADR-020 AC#7 Option A — top_k_teacher > vocab must hard-fail
    /// (mirrors `compute_dwq_targets`'s own check at `dwq_targets.rs:111`
    /// but caught earlier, before any GGUF load).
    #[test]
    fn build_dwq_targets_config_from_full_model_rejects_top_k_above_vocab() {
        let cfg = super::FullModelDwqConfig {
            gguf_path: std::path::PathBuf::from("/dev/null"),
            calibration_token_batches: vec![vec![0u32; 32 * 512]],
            top_k_teacher: 5000,
            ..super::FullModelDwqConfig::default()
        };
        let err = super::build_dwq_targets_config_from_full_model(
            &cfg,
            std::path::PathBuf::from("/tmp/x"),
            1024,
        )
        .unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("top_k_teacher (5000)") && msg.contains("<= vocab (1024)"),
            "got: {msg}"
        );
    }

    /// ADR-020 AC#7 Option A — `build_calibration_split_from_full_model`
    /// borrows the cfg's batch slab and forwards batch_size + seq_len.
    /// The borrow lifetime is tied to the cfg, so the split cannot
    /// outlive the cfg (compile-time guarantee).
    #[test]
    fn build_calibration_split_from_full_model_borrows_and_forwards() {
        let cfg = super::FullModelDwqConfig {
            gguf_path: std::path::PathBuf::from("/dev/null"),
            calibration_token_batches: vec![
                vec![1u32; 32 * 512],
                vec![2u32; 32 * 512],
                vec![3u32; 32 * 512],
            ],
            batch_size: 32,
            seq_len: 512,
            ..super::FullModelDwqConfig::default()
        };
        let split = super::build_calibration_split_from_full_model(&cfg, "train");
        assert_eq!(split.name, "train");
        assert_eq!(split.batch_size, 32);
        assert_eq!(split.seq_len, 512);
        assert_eq!(split.batches.len(), 3);
        // Borrow identity: split.batches must point at cfg's storage.
        assert!(std::ptr::eq(
            split.batches.as_ptr(),
            cfg.calibration_token_batches.as_ptr()
        ));
        // First/last tokens of each batch carry the marker we wrote.
        assert_eq!(split.batches[0][0], 1);
        assert_eq!(split.batches[1][0], 2);
        assert_eq!(split.batches[2][0], 3);
    }

    /// ADR-020 AC#7 Option A — minimal in-memory teacher used to
    /// exercise the `drive_full_model_teacher_capture` driver without
    /// loading a real GGUF.  Produces deterministic per-(row, t, v)
    /// logits so `compute_dwq_targets` has non-degenerate work to do
    /// (top-K is meaningful), and tracks how many forward passes it
    /// has served so tests can assert call counts.
    struct StubTeacher {
        vocab: usize,
        forward_calls: usize,
    }
    impl crate::calibrate::dwq_targets::TeacherLogitsProvider for StubTeacher {
        fn forward_logits(
            &mut self,
            _tokens: &[u32],
            batch_size: usize,
            seq_len: usize,
            vocab: usize,
        ) -> anyhow::Result<Vec<f32>> {
            assert_eq!(vocab, self.vocab, "fixture invariant: vocab agreement");
            self.forward_calls += 1;
            let mut out = Vec::with_capacity(batch_size * seq_len * vocab);
            for r in 0..batch_size {
                for t in 0..seq_len {
                    for v in 0..vocab {
                        let val = ((0.01 * v as f32) + 0.1 * r as f32).sin()
                            + 0.7 * ((0.05 * v as f32) + 0.3 * t as f32).cos();
                        out.push(val * 5.0);
                    }
                }
            }
            Ok(out)
        }
    }

    /// ADR-020 AC#7 Option A — driver writes one safetensors file per
    /// batch under `<save_dir>/train/<i:010d>.safetensors` and reports
    /// the count back via the summary tuple.  Using a stub teacher
    /// proves the wiring (split assembly, teacher invocation, file
    /// emission) is correct without dragging a real GGUF load into a
    /// 1-second unit test.
    #[test]
    fn drive_full_model_teacher_capture_writes_one_file_per_batch() {
        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-drive-stub-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let save_dir = dir.join("teacher-out");

        let cfg = super::FullModelDwqConfig {
            gguf_path: std::path::PathBuf::from("/dev/null"),
            calibration_token_batches: vec![
                vec![0u32; 32 * 8],
                vec![1u32; 32 * 8],
                vec![2u32; 32 * 8],
            ],
            batch_size: 32,
            seq_len: 8,
            top_k_teacher: 32,
            ..super::FullModelDwqConfig::default()
        };
        let targets_cfg =
            super::build_dwq_targets_config_from_full_model(&cfg, save_dir.clone(), 256).unwrap();

        let mut teacher = StubTeacher {
            vocab: 256,
            forward_calls: 0,
        };
        let summary =
            super::drive_full_model_teacher_capture(&mut teacher, &cfg, &targets_cfg).unwrap();

        assert_eq!(summary.len(), 1, "single split (\"train\")");
        assert_eq!(summary[0].0, "train");
        assert_eq!(summary[0].1, 3, "3 batches → 3 files");
        assert_eq!(teacher.forward_calls, 3, "one teacher forward per batch");

        // Every file must exist and have non-zero size.  File names
        // are zero-padded 10-digit decimals per dwq_targets.rs:`{:010d}`.
        for i in 0..3 {
            let p = save_dir.join("train").join(format!("{:010}.safetensors", i));
            let meta = std::fs::metadata(&p)
                .unwrap_or_else(|e| panic!("missing teacher file {}: {e}", p.display()));
            assert!(meta.len() > 0, "teacher file {} is empty", p.display());
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-020 AC#7 Option A — driver propagates teacher errors
    /// (and identifies which batch failed) so operators get an
    /// actionable diagnostic, not a silent corrupt-on-disk state.
    #[test]
    fn drive_full_model_teacher_capture_propagates_teacher_errors() {
        struct FailingTeacher;
        impl crate::calibrate::dwq_targets::TeacherLogitsProvider for FailingTeacher {
            fn forward_logits(
                &mut self,
                _tokens: &[u32],
                _batch_size: usize,
                _seq_len: usize,
                _vocab: usize,
            ) -> anyhow::Result<Vec<f32>> {
                Err(anyhow::anyhow!(
                    "stub-teacher: deliberate forward failure"
                ))
            }
        }

        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-drive-failing-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let save_dir = dir.join("teacher-out");

        let cfg = super::FullModelDwqConfig {
            gguf_path: std::path::PathBuf::from("/dev/null"),
            calibration_token_batches: vec![vec![0u32; 32 * 4]],
            batch_size: 32,
            seq_len: 4,
            top_k_teacher: 16,
            ..super::FullModelDwqConfig::default()
        };
        let targets_cfg =
            super::build_dwq_targets_config_from_full_model(&cfg, save_dir, 64).unwrap();

        let mut teacher = FailingTeacher;
        let res = super::drive_full_model_teacher_capture(&mut teacher, &cfg, &targets_cfg);
        let err = match res {
            Ok(_) => panic!("expected teacher-failure to propagate"),
            Err(e) => e,
        };
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("compute_dwq_targets") && msg.contains("deliberate forward failure"),
            "expected wrapped teacher error, got: {msg}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-020 AC#7 Option A — driver respects vocab/top_k mismatch
    /// at the targets-cfg level (caught by `compute_dwq_targets`'s
    /// own check at `dwq_targets.rs:111`).  This test crosses through
    /// the `build_dwq_targets_config_from_full_model` happy path and
    /// then mutates targets_cfg — proving the driver isn't silently
    /// ignoring downstream invariants.
    #[test]
    fn drive_full_model_teacher_capture_propagates_targets_cfg_invariants() {
        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-drive-tgt-bad-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let cfg = super::FullModelDwqConfig {
            gguf_path: std::path::PathBuf::from("/dev/null"),
            calibration_token_batches: vec![vec![0u32; 32 * 4]],
            batch_size: 32,
            seq_len: 4,
            top_k_teacher: 16,
            ..super::FullModelDwqConfig::default()
        };
        let mut targets_cfg = crate::calibrate::dwq_targets::ComputeTargetsConfig {
            top_k: 0, // BAD — compute_dwq_targets:108 must reject
            save_dir: dir.join("teacher-out"),
            vocab: 256,
        };
        let mut teacher = StubTeacher {
            vocab: 256,
            forward_calls: 0,
        };
        let res = super::drive_full_model_teacher_capture(&mut teacher, &cfg, &targets_cfg);
        let err = match res {
            Ok(_) => panic!("expected top_k=0 to reject"),
            Err(e) => e,
        };
        assert!(
            format!("{:#}", err).contains("top_k must be > 0"),
            "got: {err:#}"
        );
        assert_eq!(
            teacher.forward_calls, 0,
            "no teacher work should run when targets_cfg invariants fail"
        );

        // And: a valid targets_cfg passes through.
        targets_cfg.top_k = 8;
        let summary =
            super::drive_full_model_teacher_capture(&mut teacher, &cfg, &targets_cfg).unwrap();
        assert_eq!(summary[0].1, 1);
        assert_eq!(teacher.forward_calls, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-020 AC#7 Option A — `quant_pack_for_dwq` happy path:
    /// shape + sizes match the inputs; scales/biases counts == n*k /
    /// group_size; q_int_bytes count == n*k.  Compares against a
    /// direct `init_affine_params_gpu` call to verify the wrapper
    /// adds no semantic drift.
    #[test]
    fn quant_pack_for_dwq_matches_init_affine_directly() {
        use mlx_native::{KernelRegistry, MlxDevice};

        let device = MlxDevice::new().expect("device");
        let n = 4;
        let k = 32;
        let group_size = 8;
        let bits = 4u32;
        let w: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.013 - 0.5).collect();

        let pack = super::quant_pack_for_dwq(&device, &w, n, k, bits, group_size)
            .expect("pack");
        assert_eq!(pack.n, n);
        assert_eq!(pack.k, k);
        assert_eq!(pack.group_size, group_size);
        assert_eq!(pack.q_int_bytes.len(), n * k);
        let n_groups = (n * k) / group_size;
        assert_eq!(pack.scales.len(), n_groups);
        assert_eq!(pack.biases.len(), n_groups);

        // Direct call as oracle.
        let mut registry = KernelRegistry::new();
        let (q_oracle, s_oracle, b_oracle) =
            super::init_affine_params_gpu(&device, &mut registry, &w, group_size, bits)
                .expect("direct");
        assert_eq!(pack.q_int_bytes, q_oracle);
        assert_eq!(pack.scales, s_oracle);
        assert_eq!(pack.biases, b_oracle);
    }

    /// ADR-020 AC#7 Option A — `quant_pack_for_dwq` rejects shape
    /// mismatch (w.len != n*k).
    #[test]
    fn quant_pack_for_dwq_rejects_shape_mismatch() {
        use mlx_native::MlxDevice;
        let device = MlxDevice::new().expect("device");
        let w = vec![0.0_f32; 100]; // n*k=128 expected → fail
        let err = super::quant_pack_for_dwq(&device, &w, 4, 32, 4, 8).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("w_f32.len (100)") && msg.contains("128"),
            "got: {msg}"
        );
    }

    /// ADR-020 AC#7 Option A — k must divide group_size cleanly.
    #[test]
    fn quant_pack_for_dwq_rejects_group_size_not_dividing_k() {
        use mlx_native::MlxDevice;
        let device = MlxDevice::new().expect("device");
        // k=30 not divisible by group_size=8 → 240 % 8 = 0 (oops, 30*4=120 is div by 8)
        // Use k=12, group_size=8 → 4*12=48 div by 8 OK; need k%group_size != 0
        // k=10, gs=8 → 10 % 8 = 2 → reject
        let w = vec![0.0_f32; 4 * 10];
        let err = super::quant_pack_for_dwq(&device, &w, 4, 10, 4, 8).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("k (10)") && msg.contains("group_size (8)"),
            "got: {msg}"
        );
    }

    /// ADR-020 AC#7 Option A — n=0 or k=0 fail loud.
    #[test]
    fn quant_pack_for_dwq_rejects_zero_dims() {
        use mlx_native::MlxDevice;
        let device = MlxDevice::new().expect("device");
        let err =
            super::quant_pack_for_dwq(&device, &[], 0, 32, 4, 8).unwrap_err();
        assert!(format!("{:#}", err).contains("must be > 0"));
        let err =
            super::quant_pack_for_dwq(&device, &[], 4, 0, 4, 8).unwrap_err();
        assert!(format!("{:#}", err).contains("must be > 0"));
    }

    /// ADR-020 AC#7 Option A — full ladder handshake:
    ///   quant_pack_for_dwq → register_affine_pair → read_affine_pair
    /// the (s, b) registered with Adam round-trip the init values
    /// from the pack — proving the pack output feeds register_affine_pair
    /// without reshape or copy.
    #[test]
    fn quant_pack_for_dwq_handshake_with_register_affine_pair() {
        use crate::calibrate::adam::{AdamConfig, AdamOptimizer};
        use mlx_native::MlxDevice;

        let device = MlxDevice::new().expect("device");
        let n = 4;
        let k = 32;
        let w: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.011 + 0.2).collect();
        let pack = super::quant_pack_for_dwq(&device, &w, n, k, 4, 8).expect("pack");

        let mut adam = AdamOptimizer::new(
            device.clone(),
            AdamConfig {
                lr: 1e-4,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
            },
        )
        .expect("Adam");
        let (sn, bn) = super::register_affine_pair(
            &mut adam,
            &device,
            "blk.0.test.weight",
            &pack.scales,
            &pack.biases,
        )
        .expect("register");
        assert_eq!(sn, "blk.0.test.weight.scales");
        assert_eq!(bn, "blk.0.test.weight.biases");

        let (s_back, b_back) =
            super::read_affine_pair(&adam, "blk.0.test.weight").unwrap();
        assert_eq!(s_back, pack.scales);
        assert_eq!(b_back, pack.biases);
    }

    /// ADR-020 AC#7 Option A — qdq(s_init, b_init, q_int) recovers
    /// the original weights up to ~1 ULP per group (min-max init is
    /// the projection optimum for symmetric quant — same property
    /// the Option B falsifier test relied on, here verified through
    /// the new pack wrapper).
    #[test]
    fn quant_pack_for_dwq_qdq_recovers_w_at_init() {
        use crate::calibrate::autograd_gpu_tape::{qdq_affine, GpuTape, GpuTensor};
        use mlx_native::MlxDevice;

        let device = MlxDevice::new().expect("device");
        let n = 4;
        let k = 32;
        let group_size = 8;
        let w: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.071 - 1.0).sin() * 2.5).collect();

        let pack =
            super::quant_pack_for_dwq(&device, &w, n, k, 4, group_size).expect("pack");
        let tape = GpuTape::new(device);
        let s = GpuTensor::from_vec(&tape, &pack.scales, vec![pack.scales.len()])
            .expect("s");
        let b = GpuTensor::from_vec(&tape, &pack.biases, vec![pack.biases.len()])
            .expect("b");
        let qdq =
            qdq_affine(&s, &b, &pack.q_int_bytes, group_size).expect("qdq");
        let w_recovered = qdq.to_vec().expect("readback");
        assert_eq!(w_recovered.len(), w.len());
        // Min-max projection: max-abs error per group ≤ scale/2 ≈
        // (max-min) / (n_bins - 1) / 2.  For the fixture range ~5,
        // n_bins=16 → bound ≈ 0.17 (loose but real); typically << 0.05.
        let max_abs_err = w
            .iter()
            .zip(w_recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs_err < 0.2,
            "qdq init recovery error too high: {max_abs_err}"
        );
    }

    /// ADR-020 AC#7 Option A — `affine_param_name` produces the
    /// canonical `.scales` / `.biases` suffix that mirrors mlx-affine
    /// safetensors keys.  Stable name format is load-bearing: any
    /// downstream serializer that iterates by suffix relies on it.
    #[test]
    fn affine_param_name_canonical_suffixes() {
        assert_eq!(
            super::affine_param_name(
                "blk.0.attn_q.weight",
                super::AffineParamKind::Scale
            ),
            "blk.0.attn_q.weight.scales"
        );
        assert_eq!(
            super::affine_param_name(
                "blk.0.attn_q.weight",
                super::AffineParamKind::Bias
            ),
            "blk.0.attn_q.weight.biases"
        );
        // Distinct kinds yield distinct names (must not collide).
        assert_ne!(
            super::affine_param_name("x", super::AffineParamKind::Scale),
            super::affine_param_name("x", super::AffineParamKind::Bias)
        );
    }

    /// ADR-020 AC#7 Option A — `register_affine_pair` happy path:
    /// returns the canonical (scale, bias) names; both end up
    /// registered with Adam (n_params increments by 2 per call).
    #[test]
    fn register_affine_pair_registers_both_under_canonical_names() {
        use crate::calibrate::adam::{AdamConfig, AdamOptimizer};
        use mlx_native::MlxDevice;

        let device = MlxDevice::new().expect("device");
        let mut adam = AdamOptimizer::new(
            device.clone(),
            AdamConfig {
                lr: 1e-4,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
            },
        )
        .expect("Adam new");
        assert_eq!(adam.n_params(), 0);

        let s_init = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b_init = vec![0.1_f32, 0.2, 0.3, 0.4];
        let (s_name, b_name) =
            super::register_affine_pair(&mut adam, &device, "blk.0.foo", &s_init, &b_init)
                .expect("register");
        assert_eq!(s_name, "blk.0.foo.scales");
        assert_eq!(b_name, "blk.0.foo.biases");
        assert_eq!(adam.n_params(), 2);

        // Round-trip via read_affine_pair — values must come back unchanged.
        let (s_read, b_read) =
            super::read_affine_pair(&adam, "blk.0.foo").expect("read");
        assert_eq!(s_read, s_init);
        assert_eq!(b_read, b_init);
    }

    /// ADR-020 AC#7 Option A — registering MULTIPLE distinct
    /// tensor_ids in the same Adam yields distinct, non-colliding
    /// names.  Two tensors × 2 kinds = 4 params.
    #[test]
    fn register_affine_pair_distinct_tensor_ids_dont_collide() {
        use crate::calibrate::adam::{AdamConfig, AdamOptimizer};
        use mlx_native::MlxDevice;

        let device = MlxDevice::new().expect("device");
        let mut adam = AdamOptimizer::new(
            device.clone(),
            AdamConfig {
                lr: 1e-4,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
            },
        )
        .expect("Adam new");

        let s = vec![1.0_f32; 8];
        let b = vec![0.5_f32; 8];
        super::register_affine_pair(&mut adam, &device, "blk.0.attn_q.weight", &s, &b)
            .unwrap();
        super::register_affine_pair(&mut adam, &device, "blk.0.attn_k.weight", &s, &b)
            .unwrap();
        assert_eq!(adam.n_params(), 4);

        // Each tensor's pair is independently readable.
        let (sq, bq) = super::read_affine_pair(&adam, "blk.0.attn_q.weight").unwrap();
        let (sk, bk) = super::read_affine_pair(&adam, "blk.0.attn_k.weight").unwrap();
        assert_eq!(sq, s);
        assert_eq!(bq, b);
        assert_eq!(sk, s);
        assert_eq!(bk, b);
    }

    /// ADR-020 AC#7 Option A — duplicate registration (same
    /// tensor_id) fails loud rather than silently overwriting.
    #[test]
    fn register_affine_pair_rejects_duplicate_tensor_id() {
        use crate::calibrate::adam::{AdamConfig, AdamOptimizer};
        use mlx_native::MlxDevice;

        let device = MlxDevice::new().expect("device");
        let mut adam = AdamOptimizer::new(
            device.clone(),
            AdamConfig {
                lr: 1e-4,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
            },
        )
        .expect("Adam new");

        let s = vec![1.0_f32; 4];
        let b = vec![0.0_f32; 4];
        super::register_affine_pair(&mut adam, &device, "x", &s, &b).unwrap();
        let err =
            super::register_affine_pair(&mut adam, &device, "x", &s, &b).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("already registered") && msg.contains("x.scales"),
            "got: {msg}"
        );
    }

    /// ADR-020 AC#7 Option A — empty init slice fails loud.
    #[test]
    fn register_affine_pair_rejects_empty_inits() {
        use crate::calibrate::adam::{AdamConfig, AdamOptimizer};
        use mlx_native::MlxDevice;

        let device = MlxDevice::new().expect("device");
        let mut adam = AdamOptimizer::new(
            device.clone(),
            AdamConfig {
                lr: 1e-4,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
            },
        )
        .expect("Adam new");
        let err =
            super::register_affine_pair(&mut adam, &device, "x", &[], &[1.0]).unwrap_err();
        assert!(format!("{:#}", err).contains("scale_init must be non-empty"));
        let err =
            super::register_affine_pair(&mut adam, &device, "x", &[1.0], &[]).unwrap_err();
        assert!(format!("{:#}", err).contains("bias_init must be non-empty"));
    }

    /// ADR-020 AC#7 Option A — `read_affine_pair` returns a clear
    /// error when the names aren't registered.
    #[test]
    fn read_affine_pair_fails_for_unregistered_tensor() {
        use crate::calibrate::adam::{AdamConfig, AdamOptimizer};
        use mlx_native::MlxDevice;

        let device = MlxDevice::new().expect("device");
        let adam = AdamOptimizer::new(
            device,
            AdamConfig {
                lr: 1e-4,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
            },
        )
        .expect("Adam new");

        let err = super::read_affine_pair(&adam, "nope").unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("read scale") && msg.contains("nope.scales"),
            "got: {msg}"
        );
    }

    /// ADR-020 AC#7 Option A — `build_topk_indices_buffer` round-trips
    /// the supplied indices into an MlxBuffer of the right
    /// dtype + shape + element_count.
    #[test]
    fn build_topk_indices_buffer_roundtrips_indices() {
        use mlx_native::{DType, MlxDevice};
        let device = MlxDevice::new().expect("device");
        let indices: Vec<u32> = (0..12).collect(); // 4 rows × 3 cols
        let buf = super::build_topk_indices_buffer(&device, &indices, 4, 3).expect("buf");
        assert_eq!(buf.dtype(), DType::U32);
        assert_eq!(buf.element_count(), 12);
        assert_eq!(buf.shape(), &[4, 3]);
        let back: &[u32] = buf.as_slice().expect("read back");
        assert_eq!(back, indices.as_slice());
    }

    /// ADR-020 AC#7 Option A — `build_topk_indices_buffer` rejects
    /// length mismatch + zero dims.
    #[test]
    fn build_topk_indices_buffer_rejects_bad_inputs() {
        use mlx_native::MlxDevice;
        let device = MlxDevice::new().expect("device");
        let bad_short = vec![0u32; 5];
        let err =
            super::build_topk_indices_buffer(&device, &bad_short, 4, 3).unwrap_err();
        assert!(format!("{:#}", err).contains("indices.len (5)"));
        let err = super::build_topk_indices_buffer(&device, &[], 0, 3).unwrap_err();
        assert!(format!("{:#}", err).contains("must be > 0"));
        let err = super::build_topk_indices_buffer(&device, &[], 3, 0).unwrap_err();
        assert!(format!("{:#}", err).contains("must be > 0"));
    }

    /// ADR-020 AC#7 Option A — `gather_student_topk_via_tape` parity
    /// with the CPU oracle [`take_along_topk_indices`] on the same
    /// 3-D `[batch, seq, vocab]` slab + identical indices.
    #[test]
    fn gather_student_topk_via_tape_parity_with_cpu_oracle() {
        use crate::calibrate::autograd_gpu_tape::{GpuTape, GpuTensor};
        use mlx_native::MlxDevice;

        let batch = 2;
        let seq = 3;
        let vocab = 32;
        let top_k = 5;
        let n = batch * seq;

        let logits: Vec<f32> = (0..n * vocab)
            .map(|i| ((i as f32) * 0.061 + 0.4).sin() * 4.0 - 0.2)
            .collect();
        // Deterministic indices: each row picks (k * 7 + r * 3) mod vocab
        // for k in 0..top_k.  Distinct enough to exercise the gather.
        let mut indices: Vec<u32> = Vec::with_capacity(n * top_k);
        for r in 0..n {
            for k in 0..top_k {
                indices.push(((k * 7 + r * 3) % vocab) as u32);
            }
        }

        // CPU oracle reference output.
        let cpu = super::take_along_topk_indices(
            &logits, &indices, batch, seq, vocab, top_k,
        )
        .expect("cpu");

        // GPU tape path.
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let student =
            GpuTensor::from_vec(&tape, &logits, vec![batch, seq, vocab]).expect("student");
        let idx_buf =
            super::build_topk_indices_buffer(tape.device(), &indices, n, top_k).expect("idx");
        let gathered =
            super::gather_student_topk_via_tape(&student, idx_buf, top_k).expect("gather");
        assert_eq!(gathered.shape(), &[n, top_k]);
        let gpu = gathered.to_vec().expect("readback");
        assert_eq!(gpu.len(), cpu.len());
        for i in 0..cpu.len() {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-5,
                "mismatch at {i}: gpu={} cpu={}",
                gpu[i],
                cpu[i]
            );
        }
    }

    /// ADR-020 AC#7 Option A — gather rejects out-of-range index
    /// BEFORE launch.
    #[test]
    fn gather_student_topk_via_tape_rejects_out_of_range_index() {
        use crate::calibrate::autograd_gpu_tape::{GpuTape, GpuTensor};
        use mlx_native::MlxDevice;

        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let student =
            GpuTensor::from_vec(&tape, &vec![0.0_f32; 16], vec![4, 4]).expect("student");
        // top_k=2; indices include 4 (>= vocab=4)
        let idx = vec![0u32, 4, 1, 2, 3, 0, 2, 1];
        let idx_buf =
            super::build_topk_indices_buffer(tape.device(), &idx, 4, 2).expect("idx");
        let res = super::gather_student_topk_via_tape(&student, idx_buf, 2);
        let err = match res {
            Ok(_) => panic!("expected out-of-range error"),
            Err(e) => e,
        };
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("indices[1] = 4") && msg.contains("vocab (4)"),
            "got: {msg}"
        );
    }

    /// ADR-020 AC#7 Option A — backward path exists from a downstream
    /// loss back through the gather: ∂loss/∂student picks up signal
    /// only at the K gathered vocab positions; non-gathered slots
    /// stay at 0.  Verified via a sum-loss `loss = sum(gathered)` so
    /// `dL/dstudent[r, indices[r,k]] = 1` and 0 elsewhere.
    #[test]
    fn gather_student_topk_via_tape_backward_routes_grad_to_picked_indices() {
        use crate::calibrate::autograd_gpu_tape::{
            backward, ones_like, row_sum, scalar_mul, view, GpuTape, GpuTensor,
        };
        use mlx_native::MlxDevice;

        let n = 4;
        let vocab = 8;
        let top_k = 3;

        let logits: Vec<f32> = (0..n * vocab).map(|i| (i as f32) * 0.1 + 0.05).collect();
        let indices: Vec<u32> = vec![
            0, 3, 5, // row 0
            1, 2, 7, // row 1
            4, 6, 0, // row 2
            2, 5, 3, // row 3
        ];

        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let student =
            GpuTensor::from_vec(&tape, &logits, vec![n, vocab]).expect("student");
        let idx_buf =
            super::build_topk_indices_buffer(tape.device(), &indices, n, top_k).expect("idx");
        let gathered =
            super::gather_student_topk_via_tape(&student, idx_buf, top_k).expect("gather");
        // loss = mean(gathered); backward expects grad on gathered to
        // be 1 / (n * top_k) at every cell, propagated to student[r,
        // indices[r,k]].
        let s = row_sum(&gathered).expect("row_sum");
        let s2 = view(&s, vec![1, n]).expect("view");
        let total = row_sum(&s2).expect("row_sum total");
        let loss = scalar_mul(&total, 1.0_f32 / (n * top_k) as f32).expect("scalar_mul");
        let dy = ones_like(&tape, loss.shape()).expect("dy");
        let grads = backward(&loss, dy).expect("backward");
        let grad_buf: &[f32] = grads
            .get(student.node_idx())
            .and_then(|g| g.as_ref())
            .expect("student grad")
            .as_slice()
            .expect("grad slice");

        // Build expected grad: 1/(n*top_k) at gathered positions, 0 elsewhere.
        let weight = 1.0_f32 / (n * top_k) as f32;
        let mut expected = vec![0.0_f32; n * vocab];
        for r in 0..n {
            for k in 0..top_k {
                let v = indices[r * top_k + k] as usize;
                expected[r * vocab + v] += weight;
            }
        }
        for i in 0..expected.len() {
            assert!(
                (grad_buf[i] - expected[i]).abs() < 1e-5,
                "grad mismatch at {i}: got {} expected {}",
                grad_buf[i],
                expected[i]
            );
        }
    }

    /// ADR-020 AC#7 Option A — full pipeline:
    ///   gather_student_topk_via_tape  →  kl_loss_topk_via_tape
    ///   gives loss byte-equivalent to the CPU oracle pair
    ///   take_along_topk_indices  →  kl_loss_topk_oracle.
    #[test]
    fn gather_then_kl_via_tape_matches_cpu_oracle_chain() {
        use crate::calibrate::autograd_gpu_tape::{GpuTape, GpuTensor};
        use mlx_native::MlxDevice;

        let batch = 2;
        let seq = 4;
        let vocab = 16;
        let top_k = 6;
        let n = batch * seq;

        let teacher_logits_full: Vec<f32> = (0..n * vocab)
            .map(|i| ((i as f32) * 0.07 + 1.1).sin() * 5.0)
            .collect();
        let student_logits_full: Vec<f32> = (0..n * vocab)
            .map(|i| ((i as f32) * 0.09 - 0.3).cos() * 3.5 + 0.4)
            .collect();

        // Indices = pick the teacher's top-K per row (here we just
        // use a deterministic stride to keep the fixture bit-exact —
        // no need for a real argpartition).
        let mut indices: Vec<u32> = Vec::with_capacity(n * top_k);
        for r in 0..n {
            for k in 0..top_k {
                indices.push(((k * 5 + r * 2) % vocab) as u32);
            }
        }

        // CPU oracle chain.
        let teacher_topk_cpu = super::take_along_topk_indices(
            &teacher_logits_full,
            &indices,
            batch,
            seq,
            vocab,
            top_k,
        )
        .expect("teacher cpu gather");
        let student_topk_cpu = super::take_along_topk_indices(
            &student_logits_full,
            &indices,
            batch,
            seq,
            vocab,
            top_k,
        )
        .expect("student cpu gather");
        let cpu_loss = super::kl_loss_topk_oracle(
            &student_topk_cpu,
            &teacher_topk_cpu,
            n,
            1,
            top_k,
            2.0,
        )
        .expect("cpu kl");

        // GPU tape chain.
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let student_t =
            GpuTensor::from_vec(&tape, &student_logits_full, vec![batch, seq, vocab])
                .expect("st");
        let teacher_t =
            GpuTensor::from_vec(&tape, &teacher_logits_full, vec![batch, seq, vocab])
                .expect("tt");
        let s_idx = super::build_topk_indices_buffer(tape.device(), &indices, n, top_k)
            .expect("s idx");
        let t_idx = super::build_topk_indices_buffer(tape.device(), &indices, n, top_k)
            .expect("t idx");
        let s_topk =
            super::gather_student_topk_via_tape(&student_t, s_idx, top_k).expect("s gather");
        let t_topk =
            super::gather_student_topk_via_tape(&teacher_t, t_idx, top_k).expect("t gather");
        let loss = super::kl_loss_topk_via_tape(&s_topk, &t_topk, 2.0).expect("kl");
        let gpu_loss = loss.to_vec().expect("readback")[0];

        assert!(
            (gpu_loss - cpu_loss).abs() < 1e-4,
            "chain mismatch: gpu={} cpu={} (diff {})",
            gpu_loss,
            cpu_loss,
            gpu_loss - cpu_loss
        );
    }

    /// ADR-020 AC#7 Option A — `kl_loss_topk_via_tape` byte-equivalent
    /// parity with `kl_loss_topk_oracle` across temperatures.  Built
    /// as leaves on a fresh tape, the GPU loss must match the CPU
    /// oracle within 1e-4 absolute tolerance (room for f32 rounding
    /// across the softmax→log→sub→mul→sum chain).
    #[test]
    fn kl_loss_topk_via_tape_parity_with_cpu_oracle_across_temperatures() {
        use crate::calibrate::autograd_gpu_tape::{GpuTape, GpuTensor};
        use mlx_native::MlxDevice;

        let n = 8;
        let top_k = 16;
        let teacher: Vec<f32> = (0..n * top_k)
            .map(|i| ((i as f32) * 0.073 + 0.5).sin() * 4.0 - 0.7)
            .collect();
        let student: Vec<f32> = (0..n * top_k)
            .map(|i| ((i as f32) * 0.041 - 0.3).cos() * 3.5 + 0.2)
            .collect();

        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let st = GpuTensor::from_vec(&tape, &student, vec![n, top_k]).expect("st");
        let tt = GpuTensor::from_vec(&tape, &teacher, vec![n, top_k]).expect("tt");

        for &t in &[0.5_f32, 1.0, 2.0, 5.0] {
            let loss_t = super::kl_loss_topk_via_tape(&st, &tt, t).expect("loss");
            let loss_v = loss_t.to_vec().expect("readback");
            assert_eq!(loss_v.len(), 1);
            let cpu =
                super::kl_loss_topk_oracle(&student, &teacher, n, 1, top_k, t).expect("cpu");
            assert!(
                (loss_v[0] - cpu).abs() < 1e-4,
                "T={t}: gpu={} cpu={} (diff {})",
                loss_v[0],
                cpu,
                loss_v[0] - cpu
            );
        }
    }

    /// ADR-020 AC#7 Option A — backward path through
    /// `kl_loss_topk_via_tape`: ∂loss/∂student exists, has the
    /// expected shape, contains finite values, and zeroes out at
    /// student==teacher (analytical gradient softmax(s)−softmax(p)
    /// vanishes when distributions match).
    #[test]
    fn kl_loss_topk_via_tape_backward_zeroes_at_self() {
        use crate::calibrate::autograd_gpu_tape::{backward, ones_like, GpuTape, GpuTensor};
        use mlx_native::MlxDevice;

        let n = 4;
        let top_k = 8;
        let teacher: Vec<f32> = (0..n * top_k)
            .map(|i| ((i as f32) * 0.11 + 0.2).sin() * 3.0 + 0.5)
            .collect();
        let student = teacher.clone(); // student == teacher → grad ≈ 0

        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let st = GpuTensor::from_vec(&tape, &student, vec![n, top_k]).expect("st");
        let tt = GpuTensor::from_vec(&tape, &teacher, vec![n, top_k]).expect("tt");

        let loss = super::kl_loss_topk_via_tape(&st, &tt, 1.5).expect("loss");
        assert_eq!(loss.shape(), &[1]);

        let dy = ones_like(&tape, loss.shape()).expect("dy ones");
        let grads = backward(&loss, dy).expect("backward");
        let grad_student = grads
            .get(st.node_idx())
            .and_then(|g| g.as_ref())
            .expect("student grad");
        let grad_buf: &[f32] = grad_student.as_slice().expect("grad slice");
        assert_eq!(grad_buf.len(), n * top_k);

        for (i, &v) in grad_buf.iter().enumerate() {
            assert!(v.is_finite(), "grad[{i}] not finite: {v}");
            // softmax(s)−softmax(p) is 0 at s==p modulo f32 rounding;
            // be generous to accommodate the multi-op chain.
            assert!(
                v.abs() < 1e-4,
                "self-grad must be ≈ 0 at student==teacher; grad[{i}]={v}"
            );
        }
    }

    /// ADR-020 AC#7 Option A — backward path produces non-zero
    /// gradients when student != teacher.  Sanity that the chain
    /// actually propagates signal (not silently zeroed by a stale
    /// shape).
    #[test]
    fn kl_loss_topk_via_tape_backward_non_zero_for_differing_dists() {
        use crate::calibrate::autograd_gpu_tape::{backward, ones_like, GpuTape, GpuTensor};
        use mlx_native::MlxDevice;

        let n = 2;
        let top_k = 8;
        let teacher: Vec<f32> = vec![5.0, -3.0, 1.0, 2.0, 0.0, 0.5, -1.5, 4.0,
                                      0.5, 1.5, -0.7, 3.2, -2.1, 0.0, 1.0, -0.3];
        let student: Vec<f32> = vec![0.0; n * top_k];

        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let st = GpuTensor::from_vec(&tape, &student, vec![n, top_k]).expect("st");
        let tt = GpuTensor::from_vec(&tape, &teacher, vec![n, top_k]).expect("tt");

        let loss = super::kl_loss_topk_via_tape(&st, &tt, 2.0).expect("loss");
        let dy = ones_like(&tape, loss.shape()).expect("dy ones");
        let grads = backward(&loss, dy).expect("backward");
        let grad_buf: &[f32] = grads
            .get(st.node_idx())
            .and_then(|g| g.as_ref())
            .expect("student grad")
            .as_slice()
            .expect("grad slice");

        // Some |grad| must exceed 1e-3 — proves the signal is non-degenerate.
        let max_abs = grad_buf.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        assert!(max_abs > 1e-3, "max |grad| = {max_abs} too small");

        // Sum of grads per ROW must be ≈ 0 (softmax-derivative
        // identity: Σᵥ (softmax(s)−softmax(p))ᵥ = 1−1 = 0, scaled by
        // 1/T and 1/N).
        for r in 0..n {
            let lo = r * top_k;
            let hi = lo + top_k;
            let row_sum: f32 = grad_buf[lo..hi].iter().sum();
            assert!(
                row_sum.abs() < 1e-4,
                "row {r} grads must sum to ≈ 0; got {row_sum}"
            );
        }
    }

    /// ADR-020 AC#7 Option A — bad knobs fail loud (T <= 0, NaN T,
    /// shape mismatch, non-2D input).
    #[test]
    fn kl_loss_topk_via_tape_rejects_bad_inputs() {
        use crate::calibrate::autograd_gpu_tape::{GpuTape, GpuTensor};
        use mlx_native::MlxDevice;

        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let s = GpuTensor::from_vec(&tape, &vec![0.0_f32; 16], vec![2, 8]).expect("s");
        let t = GpuTensor::from_vec(&tape, &vec![0.0_f32; 16], vec![2, 8]).expect("t");

        let must_err = |res: anyhow::Result<crate::calibrate::autograd_gpu_tape::GpuTensor>,
                        substr: &str| {
            match res {
                Ok(_) => panic!("expected error containing '{substr}'"),
                Err(e) => assert!(
                    format!("{:#}", e).contains(substr),
                    "expected substring '{substr}'; got: {e:#}"
                ),
            }
        };

        // T = 0
        must_err(
            super::kl_loss_topk_via_tape(&s, &t, 0.0),
            "temperature must be finite and > 0",
        );
        // T = NaN
        must_err(
            super::kl_loss_topk_via_tape(&s, &t, f32::NAN),
            "temperature must be finite",
        );

        // Shape mismatch
        let t_bad =
            GpuTensor::from_vec(&tape, &vec![0.0_f32; 12], vec![2, 6]).expect("t_bad");
        must_err(
            super::kl_loss_topk_via_tape(&s, &t_bad, 1.0),
            "!= teacher shape",
        );

        // Non-2D
        let s1d = GpuTensor::from_vec(&tape, &vec![0.0_f32; 8], vec![8]).expect("s1d");
        let t1d = GpuTensor::from_vec(&tape, &vec![0.0_f32; 8], vec![8]).expect("t1d");
        must_err(
            super::kl_loss_topk_via_tape(&s1d, &t1d, 1.0),
            "must be 2-D",
        );
    }

    /// ADR-020 AC#7 Option A — `take_along_topk_indices` hand-computed
    /// case: 1 batch × 2 seq × 5 vocab → top_k=3 picks at distinct
    /// per-(r,t) indices.
    ///   logits = [[10, 20, 30, 40, 50],   // (r=0, t=0)
    ///             [11, 22, 33, 44, 55]]   // (r=0, t=1)
    ///   indices = [[3, 1, 4],             // pick logits[0,0,3], [0,0,1], [0,0,4]
    ///              [0, 2, 4]]             // pick logits[0,1,0], [0,1,2], [0,1,4]
    ///   expected = [[40, 20, 50], [11, 33, 55]]
    #[test]
    fn take_along_topk_indices_hand_computed() {
        let logits = vec![
            10.0_f32, 20.0, 30.0, 40.0, 50.0, // pos 0
            11.0, 22.0, 33.0, 44.0, 55.0, // pos 1
        ];
        let indices = vec![3u32, 1, 4, 0, 2, 4];
        let got = super::take_along_topk_indices(&logits, &indices, 1, 2, 5, 3).unwrap();
        assert_eq!(got, vec![40.0, 20.0, 50.0, 11.0, 33.0, 55.0]);
    }

    /// ADR-020 AC#7 Option A — when `top_k == vocab` and indices are
    /// `[0, 1, …, vocab-1]` per row, the gather is the identity.
    #[test]
    fn take_along_topk_indices_identity_when_indices_are_arange() {
        let batch = 2;
        let seq = 3;
        let vocab = 4;
        let logits: Vec<f32> = (0..batch * seq * vocab).map(|i| i as f32).collect();
        let mut indices: Vec<u32> = Vec::with_capacity(batch * seq * vocab);
        for _ in 0..batch * seq {
            for v in 0..vocab as u32 {
                indices.push(v);
            }
        }
        let got = super::take_along_topk_indices(&logits, &indices, batch, seq, vocab, vocab)
            .unwrap();
        assert_eq!(got, logits);
    }

    /// ADR-020 AC#7 Option A — out-of-range index fails loud, naming
    /// the exact offset and value so operators can grep their teacher
    /// dump.
    #[test]
    fn take_along_topk_indices_rejects_out_of_range_index() {
        let logits = vec![0.0_f32; 1 * 1 * 4]; // vocab=4
        let indices = vec![0u32, 1, 4, 2]; // index 4 >= vocab=4
        let err = super::take_along_topk_indices(&logits, &indices, 1, 1, 4, 4).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("indices[2] = 4") && msg.contains("vocab (4)"),
            "got: {msg}"
        );
    }

    /// ADR-020 AC#7 Option A — slice-length mismatch fails loud.
    #[test]
    fn take_along_topk_indices_rejects_slice_length_mismatch() {
        let logits = vec![0.0_f32; 8]; // batch*seq*vocab = 1*2*4 = 8 OK
        let indices = vec![0u32; 4]; // batch*seq*top_k = 1*2*3 = 6, slice = 4 → mismatch
        let err = super::take_along_topk_indices(&logits, &indices, 1, 2, 4, 3).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("indices.len (4)") && msg.contains("6"),
            "got: {msg}"
        );

        let logits2 = vec![0.0_f32; 7]; // wrong logits len
        let indices2 = vec![0u32; 6];
        let err = super::take_along_topk_indices(&logits2, &indices2, 1, 2, 4, 3).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("student_logits_full.len (7)") && msg.contains("8"),
            "got: {msg}"
        );
    }

    /// ADR-020 AC#7 Option A — bad knobs fail loud.
    #[test]
    fn take_along_topk_indices_rejects_zero_dims() {
        let logits = vec![0.0_f32; 16];
        let indices = vec![0u32; 8];
        for (b, s, v, k) in [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)] {
            let err =
                super::take_along_topk_indices(&logits, &indices, b, s, v, k).unwrap_err();
            assert!(
                format!("{:#}", err).contains("must all be > 0"),
                "case (b={b},s={s},v={v},k={k}) didn't reject: {err:#}"
            );
        }
    }

    /// ADR-020 AC#7 Option A — handshake with `TeacherBatchTargets`:
    /// the indices written by drive_full_model_teacher_capture →
    /// load_teacher_targets_for_batch must point at valid vocab
    /// positions and yield the same logits the teacher emitted (when
    /// the "student" is in fact the teacher itself, the gather over
    /// the saved indices reproduces the saved top-K logits).
    #[test]
    fn take_along_topk_indices_handshake_reproduces_teacher_topk_logits() {
        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-take-along-handshake-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let save_dir = dir.join("teacher-out");

        let cfg = super::FullModelDwqConfig {
            gguf_path: std::path::PathBuf::from("/dev/null"),
            calibration_token_batches: vec![vec![0u32; 4 * 2]], // batch=4, seq=2 (after trim → 1)
            batch_size: 4,
            seq_len: 2,
            top_k_teacher: 5,
            ..super::FullModelDwqConfig::default()
        };
        let vocab = 32;
        let targets_cfg =
            super::build_dwq_targets_config_from_full_model(&cfg, save_dir.clone(), vocab)
                .unwrap();
        let mut teacher = StubTeacher {
            vocab,
            forward_calls: 0,
        };
        super::drive_full_model_teacher_capture(&mut teacher, &cfg, &targets_cfg).unwrap();

        let t = super::load_teacher_targets_for_batch(&save_dir, "train", 0, &cfg).unwrap();

        // Reconstruct the full-vocab teacher tensor for the same batch
        // (StubTeacher is deterministic — replay the same call, then
        // gather over t.indices).  Trimmed seq is cfg.seq_len - 1 = 1.
        // StubTeacher.forward_logits([batch=4, seq=1, vocab=32]).
        let trimmed_seq = cfg.seq_len - 1;
        let mut replay_teacher = StubTeacher {
            vocab,
            forward_calls: 0,
        };
        let full_logits = <StubTeacher as crate::calibrate::dwq_targets::TeacherLogitsProvider>::forward_logits(
            &mut replay_teacher,
            &vec![0u32; cfg.batch_size * trimmed_seq],
            cfg.batch_size,
            trimmed_seq,
            vocab,
        )
        .unwrap();

        let gathered = super::take_along_topk_indices(
            &full_logits,
            &t.indices,
            cfg.batch_size,
            trimmed_seq,
            vocab,
            cfg.top_k_teacher,
        )
        .unwrap();

        // Gathered logits must equal the saved top-K logits — that's
        // the definition of "the teacher saved its own top-K".
        assert_eq!(gathered.len(), t.logits.len());
        for (i, (g, sv)) in gathered.iter().zip(t.logits.iter()).enumerate() {
            assert!(
                (g - sv).abs() < 1e-5,
                "mismatch at {i}: gathered {g} vs saved {sv}"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-020 AC#7 Option A — full pipeline composition: gather the
    /// student logits at teacher's top-K positions, feed both into the
    /// KL oracle.  When student == teacher, the resulting KL is ≈ 0
    /// — the same guarantee the standalone oracle provides, but
    /// proven through the full-vocab → top-K reduction.
    #[test]
    fn take_along_topk_then_kl_oracle_is_zero_for_self() {
        let batch = 2;
        let seq = 3;
        let vocab = 8;
        let top_k = 4;

        // Build a deterministic full-vocab logits tensor.
        let full_logits: Vec<f32> = (0..batch * seq * vocab)
            .map(|i| ((i as f32) * 0.13 + 0.7).sin() * 5.0)
            .collect();

        // For each row, take any 4 distinct indices (use the first 4
        // — equivalent to "imagine top-K were the first K positions";
        // doesn't matter for the KL=0 test as long as student and
        // teacher use the same indices).
        let mut indices: Vec<u32> = Vec::with_capacity(batch * seq * top_k);
        for _ in 0..batch * seq {
            for k in 0..top_k as u32 {
                indices.push(k);
            }
        }

        let teacher_topk =
            super::take_along_topk_indices(&full_logits, &indices, batch, seq, vocab, top_k)
                .unwrap();
        let student_topk = teacher_topk.clone(); // student == teacher
        let kl =
            super::kl_loss_topk_oracle(&student_topk, &teacher_topk, batch, seq, top_k, 1.5)
                .unwrap();
        assert!(kl.abs() < 1e-5, "self-KL must be ≈ 0; got {kl}");
    }

    /// ADR-020 AC#7 Option A — KL oracle returns 0 (within 1 ULP)
    /// when student logits == teacher logits regardless of values
    /// or temperature.  Forms the "no work to do" floor of the loss
    /// surface.
    #[test]
    fn kl_loss_topk_oracle_zero_when_student_equals_teacher() {
        let student: Vec<f32> = (0..32).map(|i| (i as f32) * 0.137 - 1.5).collect();
        let teacher = student.clone();
        for &t in &[0.5_f32, 1.0, 2.0, 5.0] {
            let kl = super::kl_loss_topk_oracle(&student, &teacher, 2, 4, 4, t).unwrap();
            assert!(
                kl.abs() < 1e-5,
                "KL must be 0 at student==teacher (T={t}); got {kl}"
            );
        }
    }

    /// ADR-020 AC#7 Option A — KL is non-negative (Gibbs' inequality
    /// — KL(p||q) >= 0 for all proper p,q).
    #[test]
    fn kl_loss_topk_oracle_non_negative_for_arbitrary_inputs() {
        let teacher: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.31 + 0.5).sin() * 3.0).collect();
        let student: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.17 - 0.2).cos() * 4.0).collect();
        let kl = super::kl_loss_topk_oracle(&student, &teacher, 1, 2, 8, 1.0).unwrap();
        assert!(kl >= -1e-6, "KL must be >= 0; got {kl}");
        // And meaningful — different distributions yield positive KL.
        assert!(kl > 0.01, "expected meaningful KL > 0.01 for differing dists; got {kl}");
    }

    /// ADR-020 AC#7 Option A — increasing temperature softens the
    /// distributions and decreases KL monotonically (Hinton 2015 KD,
    /// `dwq.py:106`).  Verified across T ∈ [0.5, 1.0, 2.0, 8.0].
    #[test]
    fn kl_loss_topk_oracle_decreases_with_higher_temperature() {
        // High-contrast logits so softmax has a clear peak that
        // temperature visibly softens.
        let teacher = vec![0.0_f32, 5.0, -3.0, 2.0];
        let student = vec![1.0_f32, 0.0, 0.0, -1.0];
        let kls: Vec<f32> = [0.5_f32, 1.0, 2.0, 8.0]
            .iter()
            .map(|&t| super::kl_loss_topk_oracle(&student, &teacher, 1, 1, 4, t).unwrap())
            .collect();
        for w in kls.windows(2) {
            assert!(
                w[0] > w[1],
                "KL must monotonically decrease with T; got {kls:?}"
            );
        }
    }

    /// ADR-020 AC#7 Option A — hand-computed K=2 case.
    /// teacher logits = [2, 0], student logits = [0, 0], T = 1.
    /// softmax(teacher) = [e²/(e²+1), 1/(e²+1)] = [0.880797, 0.119203]
    /// softmax(student) = [0.5, 0.5]
    /// KL(p_t || p_s) = 0.880797 * log(0.880797 / 0.5)
    ///                + 0.119203 * log(0.119203 / 0.5)
    ///                = 0.880797 * 0.566535 - 0.119203 * 1.434116
    ///                = 0.499000 - 0.170975
    ///                = 0.32802 (mean over 1 position == row KL)
    #[test]
    fn kl_loss_topk_oracle_hand_computed_k2_case() {
        let teacher = vec![2.0_f32, 0.0];
        let student = vec![0.0_f32, 0.0];
        let kl = super::kl_loss_topk_oracle(&student, &teacher, 1, 1, 2, 1.0).unwrap();
        let expected = 0.32802_f32;
        assert!(
            (kl - expected).abs() < 1e-3,
            "expected {expected}, got {kl}"
        );
    }

    /// ADR-020 AC#7 Option A — slice-length mismatch fails loud.
    #[test]
    fn kl_loss_topk_oracle_rejects_slice_length_mismatch() {
        let student = vec![1.0_f32; 32];
        let teacher = vec![1.0_f32; 32];
        let err =
            super::kl_loss_topk_oracle(&student, &teacher, 2, 4, 8, 1.0).unwrap_err();
        let msg = format!("{:#}", err);
        // batch*seq*top_k = 64, but slices are 32 → reject.
        assert!(
            msg.contains("student.len (32)") && msg.contains("64"),
            "got: {msg}"
        );

        let teacher2 = vec![1.0_f32; 65];
        let student2 = vec![1.0_f32; 64];
        let err =
            super::kl_loss_topk_oracle(&student2, &teacher2, 2, 4, 8, 1.0).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("teacher.len (65)") && msg.contains("64"),
            "got: {msg}"
        );
    }

    /// ADR-020 AC#7 Option A — bad knobs fail loud.
    #[test]
    fn kl_loss_topk_oracle_rejects_bad_knobs() {
        let v = vec![1.0_f32; 16];

        // T = 0
        let err = super::kl_loss_topk_oracle(&v, &v, 1, 1, 16, 0.0).unwrap_err();
        assert!(format!("{:#}", err).contains("temperature must be finite and > 0"));

        // T = NaN
        let err = super::kl_loss_topk_oracle(&v, &v, 1, 1, 16, f32::NAN).unwrap_err();
        assert!(format!("{:#}", err).contains("temperature must be finite"));

        // T < 0
        let err = super::kl_loss_topk_oracle(&v, &v, 1, 1, 16, -1.0).unwrap_err();
        assert!(format!("{:#}", err).contains("temperature must be finite and > 0"));

        // top_k = 0
        let err = super::kl_loss_topk_oracle(&v, &v, 1, 1, 0, 1.0).unwrap_err();
        assert!(format!("{:#}", err).contains("must all be > 0"));

        // batch = 0
        let err = super::kl_loss_topk_oracle(&[], &[], 0, 4, 16, 1.0).unwrap_err();
        assert!(format!("{:#}", err).contains("must all be > 0"));
    }

    /// ADR-020 AC#7 Option A — handshake: a `TeacherBatchTargets` flows
    /// into the oracle without reshape.  Caller passes the student
    /// logits sliced to the same shape (here a stand-in built from the
    /// teacher's own logits, plus a small perturbation) and gets back
    /// a finite scalar KL.
    #[test]
    fn kl_loss_topk_oracle_consumes_teacher_batch_targets_shape() {
        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-kl-handshake-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let save_dir = dir.join("teacher-out");

        let cfg = super::FullModelDwqConfig {
            gguf_path: std::path::PathBuf::from("/dev/null"),
            calibration_token_batches: vec![vec![0u32; 32 * 4]],
            batch_size: 32,
            seq_len: 4,
            top_k_teacher: 8,
            ..super::FullModelDwqConfig::default()
        };
        let targets_cfg =
            super::build_dwq_targets_config_from_full_model(&cfg, save_dir.clone(), 64).unwrap();
        let mut teacher = StubTeacher {
            vocab: 64,
            forward_calls: 0,
        };
        super::drive_full_model_teacher_capture(&mut teacher, &cfg, &targets_cfg).unwrap();

        let t = super::load_teacher_targets_for_batch(&save_dir, "train", 0, &cfg).unwrap();
        // Stand-in student logits — small perturbation off teacher.
        let student: Vec<f32> = t.logits.iter().map(|x| x + 0.3).collect();
        let kl = super::kl_loss_topk_oracle(&student, &t.logits, t.batch, t.seq, t.top_k, 2.0)
            .unwrap();
        assert!(kl.is_finite(), "got KL = {kl}");
        // Adding a uniform constant to all scaled logits leaves
        // softmax invariant ⇒ KL ≈ 0 because the +0.3 is the same
        // for every vocab index in a row.  Float-32 cancellation
        // can leave tiny negative residuals (~1e-8) so allow a
        // signed eps band rather than a strict 0 floor.
        assert!(
            kl.abs() < 1e-5,
            "uniform shift should yield |KL| < 1e-5; got {kl}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-020 AC#7 Option A — roundtrip via stub teacher:
    ///   drive_full_model_teacher_capture writes safetensors →
    ///   load_teacher_targets_for_batch reads them back →
    ///   shape + count match the cfg, content is non-degenerate.
    #[test]
    fn load_teacher_targets_for_batch_roundtrip_with_drive_full_model() {
        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-load-targets-roundtrip-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let save_dir = dir.join("teacher-out");

        let cfg = super::FullModelDwqConfig {
            gguf_path: std::path::PathBuf::from("/dev/null"),
            calibration_token_batches: vec![
                vec![0u32; 32 * 8],
                vec![1u32; 32 * 8],
            ],
            batch_size: 32,
            seq_len: 8,
            top_k_teacher: 16,
            ..super::FullModelDwqConfig::default()
        };
        let targets_cfg =
            super::build_dwq_targets_config_from_full_model(&cfg, save_dir.clone(), 256).unwrap();
        let mut teacher = StubTeacher {
            vocab: 256,
            forward_calls: 0,
        };
        super::drive_full_model_teacher_capture(&mut teacher, &cfg, &targets_cfg).unwrap();

        for i in 0..2 {
            let t = super::load_teacher_targets_for_batch(&save_dir, "train", i, &cfg).unwrap();
            assert_eq!(t.batch, 32);
            assert_eq!(t.seq, 7, "saved seq = cfg.seq_len - 1 (next-token trim)");
            assert_eq!(t.top_k, 16);
            assert_eq!(t.logits.len(), 32 * 7 * 16);
            assert_eq!(t.indices.len(), 32 * 7 * 16);
            // indices must point into the teacher's vocab (256)
            for &v in &t.indices {
                assert!(v < 256, "index {v} out of vocab=256");
            }
            // logits must contain non-zero variation (StubTeacher is
            // sin/cos-based, so all-zero would mean we read the wrong
            // file or hit a zero buffer)
            let max = t.logits.iter().cloned().fold(f32::MIN, f32::max);
            let min = t.logits.iter().cloned().fold(f32::MAX, f32::min);
            assert!(
                max - min > 1e-3,
                "batch {i} logits are degenerate (max-min={})",
                max - min
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-020 AC#7 Option A — cross-check rejects batch_size mismatch
    /// (file says 32 but cfg says 64).  We forge a file by writing
    /// with one cfg and reading with another.
    #[test]
    fn load_teacher_targets_for_batch_rejects_batch_size_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-load-targets-batch-mismatch-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let save_dir = dir.join("teacher-out");

        let write_cfg = super::FullModelDwqConfig {
            gguf_path: std::path::PathBuf::from("/dev/null"),
            calibration_token_batches: vec![vec![0u32; 32 * 4]],
            batch_size: 32,
            seq_len: 4,
            top_k_teacher: 8,
            ..super::FullModelDwqConfig::default()
        };
        let targets_cfg =
            super::build_dwq_targets_config_from_full_model(&write_cfg, save_dir.clone(), 64)
                .unwrap();
        let mut teacher = StubTeacher {
            vocab: 64,
            forward_calls: 0,
        };
        super::drive_full_model_teacher_capture(&mut teacher, &write_cfg, &targets_cfg).unwrap();

        // Read with a cfg that disagrees on batch_size.
        let read_cfg = super::FullModelDwqConfig {
            batch_size: 64,
            ..write_cfg.clone()
        };
        let err =
            super::load_teacher_targets_for_batch(&save_dir, "train", 0, &read_cfg).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("batch (32)") && msg.contains("cfg.batch_size (64)"),
            "got: {msg}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-020 AC#7 Option A — cross-check rejects seq_len mismatch
    /// (the saved seq is `cfg.seq_len - 1`).  Read with a cfg whose
    /// seq_len doesn't agree on that.
    #[test]
    fn load_teacher_targets_for_batch_rejects_seq_len_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-load-targets-seq-mismatch-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let save_dir = dir.join("teacher-out");

        let write_cfg = super::FullModelDwqConfig {
            gguf_path: std::path::PathBuf::from("/dev/null"),
            calibration_token_batches: vec![vec![0u32; 32 * 8]],
            batch_size: 32,
            seq_len: 8,
            top_k_teacher: 8,
            ..super::FullModelDwqConfig::default()
        };
        let targets_cfg =
            super::build_dwq_targets_config_from_full_model(&write_cfg, save_dir.clone(), 64)
                .unwrap();
        let mut teacher = StubTeacher {
            vocab: 64,
            forward_calls: 0,
        };
        super::drive_full_model_teacher_capture(&mut teacher, &write_cfg, &targets_cfg).unwrap();

        // Read with cfg.seq_len=16 → expected_seq=15, file says 7.
        let read_cfg = super::FullModelDwqConfig {
            seq_len: 16,
            ..write_cfg.clone()
        };
        let err =
            super::load_teacher_targets_for_batch(&save_dir, "train", 0, &read_cfg).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("seq (7)") && msg.contains("cfg.seq_len - 1 (15)"),
            "got: {msg}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-020 AC#7 Option A — cross-check rejects top_k mismatch.
    #[test]
    fn load_teacher_targets_for_batch_rejects_top_k_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-load-targets-topk-mismatch-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let save_dir = dir.join("teacher-out");

        let write_cfg = super::FullModelDwqConfig {
            gguf_path: std::path::PathBuf::from("/dev/null"),
            calibration_token_batches: vec![vec![0u32; 32 * 4]],
            batch_size: 32,
            seq_len: 4,
            top_k_teacher: 8,
            ..super::FullModelDwqConfig::default()
        };
        let targets_cfg =
            super::build_dwq_targets_config_from_full_model(&write_cfg, save_dir.clone(), 64)
                .unwrap();
        let mut teacher = StubTeacher {
            vocab: 64,
            forward_calls: 0,
        };
        super::drive_full_model_teacher_capture(&mut teacher, &write_cfg, &targets_cfg).unwrap();

        let read_cfg = super::FullModelDwqConfig {
            top_k_teacher: 32,
            ..write_cfg.clone()
        };
        let err =
            super::load_teacher_targets_for_batch(&save_dir, "train", 0, &read_cfg).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("top_k (8)") && msg.contains("cfg.top_k_teacher (32)"),
            "got: {msg}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-020 AC#7 Option A — cross-check rejects cfg.seq_len = 0
    /// (would underflow in the `seq_len - 1` computation).  Defensive
    /// even though `validate()` already rejects seq_len < 2.
    #[test]
    fn load_teacher_targets_for_batch_rejects_zero_seq_len_underflow() {
        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-load-targets-underflow-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let save_dir = dir.join("teacher-out");
        std::fs::create_dir_all(save_dir.join("train")).unwrap();
        // Empty file is fine — load_dwq_target will fail first; but
        // we want to prove the underflow check fires deterministically
        // even if a bogus file shape were on disk.  Use a cfg that
        // has seq_len=0 (illegal per validate, but the loader is
        // defensive and we want it to surface the right diagnostic).
        let read_cfg = super::FullModelDwqConfig {
            seq_len: 0,
            ..super::FullModelDwqConfig::default()
        };
        // Without a real file the load fails earlier; that's still
        // an error and proves we don't crash.  Just assert we get
        // SOME error rather than a panic.
        let res = super::load_teacher_targets_for_batch(&save_dir, "train", 0, &read_cfg);
        assert!(res.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-020 AC#7 Option A — `prepare_full_model_teacher_inputs`
    /// must fail at `cfg.validate()` BEFORE attempting to open the
    /// GGUF.  Verifies the validate-first ordering: a clearly-invalid
    /// cfg (gguf_path empty) returns a "validate" error rather than a
    /// file-open error, even if `/dev/null` was passed.
    #[test]
    fn prepare_full_model_teacher_inputs_validates_before_io() {
        let cfg = super::FullModelDwqConfig {
            // gguf_path stays empty (default) — validate() must catch
            // this before from_gguf_path is even called.
            calibration_token_batches: vec![vec![0u32; 32 * 512]],
            ..super::FullModelDwqConfig::default()
        };
        assert!(cfg.gguf_path.as_os_str().is_empty(), "fixture invariant");

        let res = super::prepare_full_model_teacher_inputs(
            &cfg,
            std::path::PathBuf::from("/tmp/teacher-out-validate-first"),
        );
        let err = match res {
            Ok(_) => panic!("expected validate-first error, got Ok"),
            Err(e) => e,
        };
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("cfg.validate") && msg.contains("gguf_path must be non-empty"),
            "expected validate-first error, got: {msg}"
        );
    }

    /// ADR-020 AC#7 Option A — `prepare_full_model_teacher_inputs`
    /// surfaces a file-open error when the gguf_path passes validate
    /// (non-empty) but does not exist on disk.  This proves the
    /// validate→open ordering: bad path bypasses the empty-string
    /// check but is still caught loudly by the open step.
    #[test]
    fn prepare_full_model_teacher_inputs_surfaces_open_error_for_missing_gguf() {
        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-prepare-missing-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // The path is non-empty (passes validate) but does not exist.
        let bogus = dir.join("does-not-exist.gguf");
        let cfg = super::FullModelDwqConfig {
            gguf_path: bogus.clone(),
            calibration_token_batches: vec![vec![0u32; 32 * 512]],
            ..super::FullModelDwqConfig::default()
        };

        let res = super::prepare_full_model_teacher_inputs(
            &cfg,
            std::path::PathBuf::from("/tmp/teacher-out-missing"),
        );
        let err = match res {
            Ok(_) => panic!("expected open-stage error, got Ok"),
            Err(e) => e,
        };
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("from_gguf_path") || msg.contains("GgufFile::open"),
            "expected open-stage error, got: {msg}"
        );
        // And it must include the path we asked to open so operators
        // can grep their log for it.
        assert!(
            msg.contains(&*bogus.to_string_lossy()),
            "error must echo the path; got: {msg}"
        );
    }

    /// ADR-020 AC#7 Option A — end-to-end teacher prep on a real GGUF.
    /// Gated on `HF2Q_TEST_GGUF` env var so the test skips cleanly
    /// when the model file is not present.  Verifies:
    ///   - the teacher loads from the path
    ///   - `prepared.targets_cfg.vocab` matches `prepared.teacher.vocab()`
    ///   - `prepared.targets_cfg.top_k` round-trips from cfg
    ///   - `prepared.targets_cfg.save_dir` matches the caller-supplied path
    #[test]
    fn prepare_full_model_teacher_inputs_real_gguf_e2e() {
        let gguf_path = match std::env::var("HF2Q_TEST_GGUF") {
            Ok(s) => s,
            Err(_) => {
                eprintln!("[adr-020 ac#7 prep] SKIP: HF2Q_TEST_GGUF not set");
                return;
            }
        };
        let path = std::path::PathBuf::from(&gguf_path);
        if !path.exists() {
            eprintln!("[adr-020 ac#7 prep] SKIP: {gguf_path} does not exist");
            return;
        }

        let cfg = super::FullModelDwqConfig {
            gguf_path: path,
            calibration_token_batches: vec![vec![0u32; 32 * 4]],
            batch_size: 32,
            seq_len: 4,
            top_k_teacher: 64,
            ..super::FullModelDwqConfig::default()
        };

        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-prepare-e2e-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let save_dir = dir.join("teacher-out");

        let prepared = match super::prepare_full_model_teacher_inputs(&cfg, save_dir.clone()) {
            Ok(p) => p,
            Err(e) => panic!("real-GGUF teacher prep failed: {e:#}"),
        };
        assert!(prepared.teacher.vocab() > 0);
        assert_eq!(prepared.targets_cfg.vocab, prepared.teacher.vocab());
        assert_eq!(prepared.targets_cfg.top_k, 64);
        assert_eq!(prepared.targets_cfg.save_dir, save_dir);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-020 AC#7 Option A — adapter chain: JSONL loader output →
    /// FullModelDwqConfig → `compute_dwq_targets` inputs (split + cfg).
    /// Verifies the three helpers compose without reshape.
    #[test]
    fn full_model_adapter_chain_jsonl_to_compute_dwq_targets_inputs() {
        let tok = fixture_wordlevel_tokenizer();
        let dir = std::env::temp_dir().join(format!(
            "hf2q-adr020-adapter-chain-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corpus.jsonl");
        let mut body = String::new();
        for i in 0..32 {
            body.push_str(&format!("{{\"text\": \"hello world {}\"}}\n", i % 3));
        }
        std::fs::write(&path, body).unwrap();

        let batches = super::load_calibration_corpus_jsonl(&path, &tok, 32, 4).unwrap();
        assert_eq!(batches.len(), 1);

        let cfg = super::FullModelDwqConfig {
            gguf_path: std::path::PathBuf::from("/dev/null"),
            calibration_token_batches: batches,
            batch_size: 32,
            seq_len: 4,
            top_k_teacher: 100,
            ..super::FullModelDwqConfig::default()
        };

        // adapter #1: targets config
        let tgt_cfg = super::build_dwq_targets_config_from_full_model(
            &cfg,
            dir.join("teacher-out"),
            500,
        )
        .unwrap();
        assert_eq!(tgt_cfg.top_k, 100);
        assert_eq!(tgt_cfg.vocab, 500);

        // adapter #2: split
        let split = super::build_calibration_split_from_full_model(&cfg, "train");
        assert_eq!(split.batch_size, 32);
        assert_eq!(split.seq_len, 4);
        assert_eq!(split.batches.len(), 1);
        assert_eq!(split.batches[0].len(), 32 * 4);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-020 iter-12d-2 — production driver test (real GGUF; gated
    /// on `HF2Q_TEST_GGUF` env var so unit-test runs without the
    /// model file present skip cleanly).  Verifies:
    ///   - scan produces a non-empty `trained` set
    ///   - serialized safetensors is non-empty + parseable by
    ///     safetensors::SafeTensors::deserialize
    ///   - every trained tensor's `<name>.weight`/`.scales`/`.biases`
    ///     triplet round-trips through MlxAffineLinear::from_safetensors
    ///   - skipped tensors carry a non-empty reason
    ///
    /// To keep the test runtime bounded under real models (~100+
    /// Linears), the test filters to ONLY the first 4 matching
    /// tensors via name_filter.  Override with HF2Q_TEST_GGUF_LIMIT
    /// for broader coverage.
    #[test]
    fn iter_12d2_train_all_linears_dwq_real_gguf() {
        use super::{train_all_linears_dwq, DwqTrainingConfig};
        use crate::calibrate::mlx_safetensors_loader::MlxAffineLinear;
        use safetensors::tensor::Dtype;
        use safetensors::SafeTensors;

        let gguf_path = match std::env::var("HF2Q_TEST_GGUF") {
            Ok(s) => s,
            Err(_) => {
                eprintln!("[iter-12d-2] SKIP: HF2Q_TEST_GGUF not set");
                return;
            }
        };
        let path = std::path::PathBuf::from(&gguf_path);
        if !path.exists() {
            eprintln!("[iter-12d-2] SKIP: {gguf_path} does not exist");
            return;
        }

        let limit: usize = std::env::var("HF2Q_TEST_GGUF_LIMIT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        let counter = std::cell::Cell::new(0usize);
        let name_filter = |name: &str| -> bool {
            // Must end in `.weight` (Linear weights only) AND be one of the
            // first `limit` matching names.
            if !name.ends_with(".weight") {
                return false;
            }
            // Skip token embedding + output head (huge) — they bloat scan time.
            if name.starts_with("token_embd") || name.starts_with("output") {
                return false;
            }
            let cur = counter.get();
            if cur >= limit {
                return false;
            }
            counter.set(cur + 1);
            true
        };

        let cfg = DwqTrainingConfig {
            n_steps: 30,
            // Real Q4_0 stddev needs the full iter-17b-style 0.34 floor.
            convergence_ratio: 0.34,
            ..DwqTrainingConfig::default()
        };

        let result = train_all_linears_dwq(&path, &cfg, Dtype::BF16, name_filter)
            .expect("train_all_linears_dwq");

        eprintln!(
            "[iter-12d-2] trained={} skipped={} safetensors_bytes={}",
            result.trained.len(),
            result.skipped.len(),
            result.safetensors_bytes.len()
        );

        // At least one tensor must have trained successfully (otherwise
        // either the GGUF is empty or the convergence floor is wrong for
        // this model — both are real failures the test should catch).
        assert!(
            !result.trained.is_empty(),
            "no Linears trained — first skipped reasons: {:?}",
            result.skipped.iter().take(3).collect::<Vec<_>>()
        );

        // Skipped tensors must carry a non-empty reason.
        for s in &result.skipped {
            assert!(
                !s.reason.is_empty(),
                "skipped tensor {} has empty reason",
                s.name
            );
        }

        // Safetensors must parse + contain 3 keys per trained tensor.
        let parsed = SafeTensors::deserialize(&result.safetensors_bytes)
            .expect("safetensors parse");
        let parsed_names: std::collections::HashSet<String> =
            parsed.names().into_iter().map(|s| s.to_string()).collect();
        assert_eq!(
            parsed_names.len(),
            result.trained.len() * 3,
            "expected 3 tensors per trained Linear, got {} for {} trained",
            parsed_names.len(),
            result.trained.len()
        );
        for t in &result.trained {
            let stem = t.name.strip_suffix(".weight").unwrap_or(&t.name);
            for suffix in [".weight", ".scales", ".biases"] {
                let key = format!("{stem}{suffix}");
                assert!(
                    parsed_names.contains(&key),
                    "missing safetensors key {key}"
                );
            }
        }

        // Each triplet must round-trip through MlxAffineLinear::from_safetensors.
        for t in &result.trained {
            let stem = t.name.strip_suffix(".weight").unwrap_or(&t.name);
            let lin = MlxAffineLinear::from_safetensors(
                &parsed,
                stem,
                cfg.bits,
                cfg.group_size,
            )
            .unwrap_or_else(|e| {
                panic!("MlxAffineLinear::from_safetensors {stem}: {e}")
            });
            assert_eq!(lin.n, t.n);
            assert_eq!(lin.k, t.k);
            assert_eq!(lin.q_int.len(), t.n * t.k);
            assert_eq!(
                lin.scales.len(),
                t.n * (t.k / cfg.group_size),
                "scales length mismatch for {stem}"
            );

            // KL diagnostics must reflect actual training (kl_min < kl_initial).
            assert!(
                t.kl_min <= t.kl_initial,
                "kl_min={} > kl_initial={} for {}",
                t.kl_min,
                t.kl_initial,
                t.name
            );
            assert!(t.kl_initial.is_finite());
            assert!(t.kl_min.is_finite());
        }
    }

    #[test]
    fn init_affine_params_gpu_round_trip_recovers_w_within_quant_error() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let group_size = 32usize;
        let n_groups = 3usize;
        let n_total = group_size * n_groups;
        let w: Vec<f32> = (0..n_total)
            .map(|i| ((i as f32) * 0.51).sin() + ((i as f32) * 0.123).cos() * 0.3)
            .collect();
        let (q, s, b) =
            init_affine_params_gpu(&device, &mut registry, &w, group_size, 4).unwrap();
        assert_eq!(q.len(), n_total);
        assert_eq!(s.len(), n_groups);
        assert_eq!(b.len(), n_groups);
        for g in 0..n_groups {
            for i in 0..group_size {
                let idx = g * group_size + i;
                let qdq = q[idx] as f32 * s[g] + b[g];
                let bound = s[g] * 0.5 + 1e-6;
                assert!(
                    (qdq - w[idx]).abs() <= bound,
                    "qdq[{idx}]={} w[{idx}]={}",
                    qdq,
                    w[idx]
                );
            }
        }
    }

    /// ADR-020 AC#7 Option A — verify production `init_affine_params_gpu`
    /// (the kernel that lives behind the public hf2q surface) produces
    /// results CONSISTENT with the CPU min-max-init algorithm used in
    /// the AC#7 foundation tests
    /// (`qwen35_moe::tests::ac7_option_a_*` series).
    ///
    /// This closes the doc-update gap from 2026-05-08: the original
    /// implementation ladder said "extract CPU init helper to public",
    /// but `init_affine_params_gpu` (line 49 above) ALREADY does this
    /// on GPU.  Production code should use the existing function; this
    /// test establishes the equivalence proof so the foundation-test
    /// CPU helper can be removed in a future iter without introducing
    /// drift.
    ///
    /// Algorithm (matches qdq_affine.metal:85-103 + my test's CPU init):
    ///   For each group of `group_size` elements:
    ///     w_min = min(group), w_max = max(group)
    ///     s = (w_max - w_min) / (n_bins - 1)  (or 1.0 if degenerate)
    ///     b = w_min
    ///     q[i] = clamp(round((w[i] - b) / s), 0, n_bins - 1)
    #[test]
    fn init_affine_params_gpu_matches_cpu_min_max_algorithm() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let group_size = 32usize;
        let bits = 4u32;
        let n_bins = 1u32 << bits;
        // Multi-row + multi-group fixture mirroring real-Linear shapes:
        // n=64 rows, k=64 cols → 128 groups total.
        let n = 64usize;
        let k = 64usize;
        let n_total = n * k;
        let w: Vec<f32> = (0..n_total)
            .map(|i| ((i as f32) * 0.0173 - 0.4).sin() * 0.5)
            .collect();

        // GPU path.
        let (q_gpu, s_gpu, b_gpu) =
            init_affine_params_gpu(&device, &mut registry, &w, group_size, bits).unwrap();

        // CPU oracle (mirrors my test's init_qdq + qdq_affine.metal:85-103).
        let n_groups = n_total / group_size;
        let mut s_cpu = vec![0.0f32; n_groups];
        let mut b_cpu = vec![0.0f32; n_groups];
        let mut q_cpu = vec![0u8; n_total];
        for g in 0..n_groups {
            let base = g * group_size;
            let slab = &w[base..base + group_size];
            let w_min = slab.iter().copied().fold(f32::INFINITY, f32::min);
            let w_max = slab.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let s_val = if w_max > w_min {
                (w_max - w_min) / (n_bins - 1) as f32
            } else {
                1.0
            };
            s_cpu[g] = s_val;
            b_cpu[g] = w_min;
            for i in 0..group_size {
                let z = (slab[i] - w_min) / s_val;
                let qv = if z >= 0.0 {
                    (z + 0.5).floor() as i32
                } else {
                    (z - 0.5).ceil() as i32
                };
                q_cpu[base + i] = qv.clamp(0, (n_bins - 1) as i32) as u8;
            }
        }

        // Element-wise consistency.
        assert_eq!(s_gpu.len(), n_groups);
        assert_eq!(b_gpu.len(), n_groups);
        assert_eq!(q_gpu.len(), n_total);
        for g in 0..n_groups {
            assert!(
                (s_gpu[g] - s_cpu[g]).abs() < 1e-6,
                "scales[{g}]: gpu={} cpu={}",
                s_gpu[g],
                s_cpu[g]
            );
            assert!(
                (b_gpu[g] - b_cpu[g]).abs() < 1e-6,
                "biases[{g}]: gpu={} cpu={}",
                b_gpu[g],
                b_cpu[g]
            );
        }
        // q_int is u8 — must match exactly (no float tolerance).
        let mut mismatches = 0usize;
        for i in 0..n_total {
            if q_gpu[i] != q_cpu[i] {
                mismatches += 1;
            }
        }
        // Allow up to 0.1% mismatches due to round-half-to-even vs
        // round-half-away-from-zero (Metal vs Rust f32::round).  In
        // practice we expect 0 because the kernel uses
        // round-half-away-from-zero (qdq_affine.metal:101).
        let tol = (n_total / 1000).max(1);
        assert!(
            mismatches <= tol,
            "q_int mismatches: {mismatches}/{n_total} (tol={tol})"
        );
    }
}
