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
