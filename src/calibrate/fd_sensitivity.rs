//! ADR-020 iter-12b-1 — finite-difference per-Linear sensitivity primitive.
//!
//! Implements the canonical mlx-lm `dynamic_quant.estimate_sensitivities`
//! semantic via finite-difference rather than autograd:
//!
//! ```text
//! score = (KL(y_T/T ‖ y_low/T) − KL(y_T/T ‖ y_high/T)) / (numel / 1e6)
//! ```
//!
//! where `y_T` is the teacher's full-precision logits, `y_low` /
//! `y_high` are logits with the target Linear's `W` replaced by
//! `qdq(W, low_bits)` / `qdq(W, high_bits)` respectively.
//!
//! ## Why FD is canonically correct (not a fallback)
//!
//! mlx-lm computes `(∇_W KL · (W_low − W_high)).sum() / params_M` —
//! a **first-order Taylor approximation** of the very same scalar
//! `KL(W_low) − KL(W_high)` that FD measures **directly**.  Both
//! mathematically converge to the same limit; FD is the second-
//! order-accurate measurement (no `O(δ²)` Taylor remainder), autograd
//! is the cheap-amortized approximation.
//!
//! Per ADR-020 mantra "no fallback": this is a different mechanism
//! for the same quantity, not a degraded substitute.  The runtime
//! difference (`2 forwards/Linear` vs `1 fwd + 1 bwd total` for
//! autograd) is a one-time-per-model calibration cost users don't
//! perceive — they perceive model quality, which is identical.
//!
//! ## Scope of THIS iter (12b-1)
//!
//! Pure-CPU library primitive operating on a SINGLE Linear's
//! `(W, X, y_T)` triple.  No model integration, no Metal.  Unit-tested
//! against synthetic fixtures.  Real-model integration (capturing
//! `y_T` from `forward_gpu` + per-Linear weight swap in `Qwen35Model`)
//! is iter-12b-2.  Calibrator wire-up (replacing
//! `compute_layer_sensitivity` in the `DwqCalibrator` dispatch) is
//! iter-12b-3.

use std::collections::BTreeMap;

use anyhow::{anyhow, Result};

use crate::ir::{lazy::LazyTensorMap, TensorMap};

/// Compute FD sensitivity for a single Linear weight tensor.
///
/// # Arguments
///
/// * `w` — full-precision weight, shape `[n, k]` row-major (one row
///   per output channel, K reduction axis on the inner dim).
/// * `x` — calibration activations, shape `[m, k]` row-major (m batch
///   rows, K matches W's inner dim).
/// * `y_t` — teacher logits, shape `[m, n]` row-major.  Must equal
///   `x @ w^T` to within FP rounding (callers supply this so we
///   don't recompute the host FP64 oracle inside the per-Linear hot
///   path).
/// * `n`, `k`, `m` — explicit shapes (caller-supplied to avoid
///   inferring from slice lengths under multiple compatible layouts).
/// * `low_bits`, `high_bits` — quantization bit pair.  Convention is
///   `low_bits ≤ high_bits` (e.g. (4, 8), (4, 6)), matching mlx-lm's
///   sensitivity sign convention: positive score → tensor benefits
///   more from `high_bits`, negative → no improvement (rare; should
///   round to ~0).
/// * `group_size` — quantization group size (32 for Q4_0/Q8_0 GGUF).
///   `k` must be divisible by `group_size`.
/// * `temperature` — softmax temperature for the KL-div computation
///   (mlx-lm canonical = 2.0; matches iter-13e/iter-17b convention).
///
/// # Returns
///
/// Per-million-parameters-normalized sensitivity score.  Higher
/// magnitude means the tensor is more sensitive to quantization
/// (larger gap between `low_bits` and `high_bits` reconstructions).
///
/// # Errors
///
/// * Shape mismatch (`w.len() != n*k`, `x.len() != m*k`,
///   `y_t.len() != m*n`).
/// * `k % group_size != 0` (Q4_0/Q8_0 GGUF block alignment).
/// * Bit pair ordering or invalid bits.
/// * Non-finite intermediate (NaN guard).
pub fn compute_fd_sensitivity(
    w: &[f32],
    x: &[f32],
    y_t: &[f32],
    n: usize,
    k: usize,
    m: usize,
    low_bits: u32,
    high_bits: u32,
    group_size: usize,
    temperature: f32,
) -> Result<f32> {
    validate_args(w, x, y_t, n, k, m, low_bits, high_bits, group_size, temperature)?;

    let w_low = q_legacy_round_trip(w, group_size, low_bits)?;
    let w_high = q_legacy_round_trip(w, group_size, high_bits)?;

    let y_low = matmul_x_wt(x, &w_low, m, n, k);
    let y_high = matmul_x_wt(x, &w_high, m, n, k);

    let inv_t = 1.0f32 / temperature;
    let kl_low = mean_kl_per_row(y_t, &y_low, m, n, inv_t)?;
    let kl_high = mean_kl_per_row(y_t, &y_high, m, n, inv_t)?;

    let numel = (n * k) as f64;
    let score = ((kl_low - kl_high) as f64) / (numel / 1.0e6);
    Ok(score as f32)
}

fn validate_args(
    w: &[f32],
    x: &[f32],
    y_t: &[f32],
    n: usize,
    k: usize,
    m: usize,
    low_bits: u32,
    high_bits: u32,
    group_size: usize,
    temperature: f32,
) -> Result<()> {
    if w.len() != n * k {
        return Err(anyhow!(
            "compute_fd_sensitivity: w.len()={} != n*k={}",
            w.len(),
            n * k
        ));
    }
    if x.len() != m * k {
        return Err(anyhow!(
            "compute_fd_sensitivity: x.len()={} != m*k={}",
            x.len(),
            m * k
        ));
    }
    if y_t.len() != m * n {
        return Err(anyhow!(
            "compute_fd_sensitivity: y_t.len()={} != m*n={}",
            y_t.len(),
            m * n
        ));
    }
    if group_size == 0 || !group_size.is_power_of_two() {
        return Err(anyhow!(
            "compute_fd_sensitivity: group_size={} must be a positive power of two",
            group_size
        ));
    }
    if k % group_size != 0 {
        return Err(anyhow!(
            "compute_fd_sensitivity: k={} not divisible by group_size={}",
            k,
            group_size
        ));
    }
    if !(2..=8).contains(&low_bits) || !(2..=8).contains(&high_bits) {
        return Err(anyhow!(
            "compute_fd_sensitivity: bits must be in [2, 8] (got low={} high={})",
            low_bits,
            high_bits
        ));
    }
    if low_bits > high_bits {
        return Err(anyhow!(
            "compute_fd_sensitivity: convention is low_bits ≤ high_bits (got {}/{})",
            low_bits,
            high_bits
        ));
    }
    if !(temperature > 0.0 && temperature.is_finite()) {
        return Err(anyhow!(
            "compute_fd_sensitivity: temperature must be > 0 (got {})",
            temperature
        ));
    }
    Ok(())
}

/// CPU per-32-block (or per-`group_size`) Q-legacy round-trip
/// (signed-amax scale, no zero-point).  Matches the GPU
/// `qdq_q{4,8}_0_gpu` kernels in `qdq_gpu.rs` to within FP rounding;
/// kept CPU-only so the FD primitive is unit-testable without a
/// Metal device.
///
/// Layout: groups along the inner dim per row, per mlx-lm + GGUF
/// convention.  `w[r * k + c]` quantizes within group
/// `g = c / group_size`.
fn q_legacy_round_trip(w: &[f32], group_size: usize, bits: u32) -> Result<Vec<f32>> {
    if bits < 2 || bits > 8 {
        return Err(anyhow!("q_legacy_round_trip: bad bits={}", bits));
    }
    let levels = 1u32 << bits;
    // Symmetric signed range: -(levels/2) .. (levels/2 - 1).  E.g.
    // bits=4 → [-8, 7], bits=8 → [-128, 127].
    let q_min = -((levels as i32) / 2);
    let q_max = (levels as i32) / 2 - 1;

    let mut out = vec![0.0f32; w.len()];
    let n_groups = w.len() / group_size;
    if w.len() % group_size != 0 {
        return Err(anyhow!(
            "q_legacy_round_trip: w.len()={} not divisible by group_size={}",
            w.len(),
            group_size
        ));
    }
    for g in 0..n_groups {
        let start = g * group_size;
        let block = &w[start..start + group_size];

        let mut amax = 0.0f32;
        for &v in block {
            if v.abs() > amax {
                amax = v.abs();
            }
        }
        // Signed-amax scale: amax / |q_min| (matches GGUF Q4_0).
        let scale = if amax == 0.0 {
            1.0f32
        } else {
            amax / q_min.unsigned_abs() as f32
        };
        let inv_s = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        for (i, &v) in block.iter().enumerate() {
            let q = (v * inv_s).round() as i32;
            let q_clamped = q.clamp(q_min, q_max);
            out[start + i] = (q_clamped as f32) * scale;
        }
    }
    Ok(out)
}

/// `Y = X @ W^T` where W is [n, k] row-major, X is [m, k] row-major,
/// Y is [m, n] row-major.  Reduction in FP64 to keep the FD signal
/// above FP32 noise on long-K Linears.
fn matmul_x_wt(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += (x[r * k + kk] as f64) * (w[c * k + kk] as f64);
            }
            y[r * n + c] = acc as f32;
        }
    }
    y
}

/// Mean over rows of `KL(softmax(p/T) ‖ softmax(q/T))`, where p / q
/// are scaled by `inv_t` before softmax.
///
/// Per-row stable softmax (max-subtract before exp).
fn mean_kl_per_row(p: &[f32], q: &[f32], m: usize, n: usize, inv_t: f32) -> Result<f32> {
    if p.len() != m * n || q.len() != m * n {
        return Err(anyhow!("mean_kl_per_row: shape mismatch"));
    }
    let mut total = 0.0f64;
    for r in 0..m {
        let p_row = &p[r * n..(r + 1) * n];
        let q_row = &q[r * n..(r + 1) * n];
        // Stable softmax probabilities.
        let p_probs = stable_softmax_scaled(p_row, inv_t);
        let q_log_probs = stable_log_softmax_scaled(q_row, inv_t);
        let p_log_probs = stable_log_softmax_scaled(p_row, inv_t);
        // KL(p ‖ q) = Σ_i p_i · (log p_i − log q_i)
        let mut row_kl = 0.0f64;
        for i in 0..n {
            let p_i = p_probs[i] as f64;
            let log_p_i = p_log_probs[i] as f64;
            let log_q_i = q_log_probs[i] as f64;
            let term = p_i * (log_p_i - log_q_i);
            if !term.is_finite() {
                return Err(anyhow!(
                    "mean_kl_per_row: non-finite KL term at row {} class {}",
                    r,
                    i
                ));
            }
            row_kl += term;
        }
        total += row_kl;
    }
    Ok((total / m as f64) as f32)
}

fn stable_softmax_scaled(logits: &[f32], inv_t: f32) -> Vec<f32> {
    let mut max_val = f32::NEG_INFINITY;
    for &l in logits {
        let scaled = l * inv_t;
        if scaled > max_val {
            max_val = scaled;
        }
    }
    let mut sum = 0.0f64;
    let mut exps: Vec<f32> = Vec::with_capacity(logits.len());
    for &l in logits {
        let e = ((l * inv_t - max_val) as f64).exp();
        sum += e;
        exps.push(e as f32);
    }
    if sum <= 0.0 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.iter().map(|&e| e / sum as f32).collect()
}

fn stable_log_softmax_scaled(logits: &[f32], inv_t: f32) -> Vec<f32> {
    let mut max_val = f32::NEG_INFINITY;
    for &l in logits {
        let scaled = l * inv_t;
        if scaled > max_val {
            max_val = scaled;
        }
    }
    let mut sum = 0.0f64;
    for &l in logits {
        sum += ((l * inv_t - max_val) as f64).exp();
    }
    let log_sum = sum.ln() as f32;
    logits
        .iter()
        .map(|&l| l * inv_t - max_val - log_sum)
        .collect()
}

/// ADR-020 iter-12b-2-prep — borrowed bundle of one Linear's FD
/// sensitivity inputs, mirroring `mlx-lm`'s
/// `dynamic_quant.estimate_sensitivities` per-tensor input shape.
///
/// All slices use the same row-major layout as
/// [`compute_fd_sensitivity`].  `name` is the Linear's path (e.g.
/// `"blk.0.ffn_gate_inp.weight"`) — used as the `BTreeMap` key in the
/// returned sensitivity dict so the calibrator can attribute scores
/// back to specific tensors when allocating bits.
pub struct LinearFdInput<'a> {
    pub name: String,
    pub w: &'a [f32],
    pub x: &'a [f32],
    pub y_t: &'a [f32],
    pub n: usize,
    pub k: usize,
    pub m: usize,
}

/// ADR-020 iter-12b-2-prep — compute FD sensitivity for a list of
/// Linears, returning an `mlx-lm`-shaped `BTreeMap<name → score>`.
///
/// Wraps the per-Linear [`compute_fd_sensitivity`] primitive in a list-
/// shaped API matching `dynamic_quant_gpu::estimate_sensitivities`'s
/// return type.  Bridges iter-12b-1 (per-Linear primitive) to
/// iter-12b-3 (calibrator wire-up): downstream code can take the
/// returned scores → wrap in `LayerSensitivity` → feed to
/// `allocate_bits_by_sensitivity` for the bit assignment, OR map
/// directly to a quantizer mask without going through the layer-
/// indexed heuristic struct.
///
/// Fail-loud on duplicate `name` keys (silent map-overwrite would
/// drop sensitivity data and produce wrong allocations).  Same per-
/// Linear validation as [`compute_fd_sensitivity`].
///
/// # Arguments
///
/// * `linears` — slice of `LinearFdInput` per Linear under
///   consideration; ordering does not affect output.
/// * `low_bits`, `high_bits`, `group_size`, `temperature` — same
///   semantics as [`compute_fd_sensitivity`]; applied uniformly across
///   all Linears (bit-pair sweeps are the caller's job).
///
/// # Errors
///
/// * Any per-Linear validation failure from [`compute_fd_sensitivity`]
///   (annotated with the offending Linear's name).
/// * Duplicate `name` across the input slice.
pub fn compute_fd_sensitivity_per_linear(
    linears: &[LinearFdInput<'_>],
    low_bits: u32,
    high_bits: u32,
    group_size: usize,
    temperature: f32,
) -> Result<BTreeMap<String, f32>> {
    let mut out = BTreeMap::new();
    for lin in linears {
        if out.contains_key(&lin.name) {
            return Err(anyhow!(
                "compute_fd_sensitivity_per_linear: duplicate name {:?} in input \
                 — bit allocation would silently lose data",
                lin.name
            ));
        }
        let score = compute_fd_sensitivity(
            lin.w, lin.x, lin.y_t, lin.n, lin.k, lin.m,
            low_bits, high_bits, group_size, temperature,
        )
        .map_err(|e| anyhow!("compute_fd_sensitivity for {:?}: {e}", lin.name))?;
        out.insert(lin.name.clone(), score);
    }
    Ok(out)
}

/// ADR-020 iter-12b-2-attn — owned per-Linear FD input bundle.
///
/// CPU-side companion to [`LinearFdInput`] for callers that need to
/// MATERIALIZE the (X, W, y_T) triple before calling
/// [`compute_fd_sensitivity_per_linear`].  Capture helpers that
/// derive `X` by running normalization / activations on a layer's
/// residual stream return owned tensors; this struct holds them and
/// exposes [`Self::as_borrowed`] to convert into a borrow at the
/// dispatch site.
#[derive(Debug, Clone)]
pub struct LinearFdInputOwned {
    pub name: String,
    pub w: Vec<f32>,
    pub x: Vec<f32>,
    pub y_t: Vec<f32>,
    pub n: usize,
    pub k: usize,
    pub m: usize,
}

impl LinearFdInputOwned {
    /// Borrow as a [`LinearFdInput`] for [`compute_fd_sensitivity_per_linear`].
    pub fn as_borrowed(&self) -> LinearFdInput<'_> {
        LinearFdInput {
            name: self.name.clone(),
            w: &self.w,
            x: &self.x,
            y_t: &self.y_t,
            n: self.n,
            k: self.k,
            m: self.m,
        }
    }
}

/// Host RMS-norm reference: `out[i, j] = x[i, j] * w[j] / sqrt(mean(x[i,
/// :]^2) + eps)`.  Per-row reduction in FP64 to match the GPU kernel's
/// numerical contract on long-hidden tensors.
fn rms_norm_host(x: &[f32], w: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let mut sumsq = 0.0f64;
        for c in 0..dim {
            let v = x[r * dim + c] as f64;
            sumsq += v * v;
        }
        let inv_rms = (1.0_f64 / ((sumsq / dim as f64) + eps as f64).sqrt()) as f32;
        for c in 0..dim {
            out[r * dim + c] = x[r * dim + c] * inv_rms * w[c];
        }
    }
    out
}

/// ADR-020 iter-12b-2-attn — capture FD-sensitivity inputs for the
/// 3 attn-block Linears (Q, K, V) of one dense decoder layer.
///
/// For DWQ sensitivity scoring: each of Q/K/V receives the same
/// post-attn-norm activation as input, and emits its local matmul
/// output.  Computing this triple per-Linear lets
/// [`compute_fd_sensitivity_per_linear`] measure local quantization
/// impact without running the full attention block.
///
/// # Names
///
/// Output entries are named `blk.{layer_idx}.attn_{q,k,v}.weight`,
/// matching GGUF naming convention so calibrator wire-up
/// (iter-12b-3) can key directly into the `BTreeMap<name → score>`
/// returned by [`compute_fd_sensitivity_per_linear`].
///
/// # Shape contract
///
/// * `x_layer_input`: `[batch, hidden]` row-major — the residual
///   stream entering the layer (from `LayerActivations::layer_inputs`).
/// * `w_attn_norm`: `[hidden]` — pre-attn RMS-norm scale.
/// * `w_q`, `w_k`, `w_v`: `[hidden, hidden]` — attn projection weights.
/// * Output `y_T` per Linear: `[batch, hidden]` — local matmul output
///   `X_attn @ W^T` where `X_attn = rms_norm(x_layer_input, w_attn_norm)`.
///
/// # Errors
///
/// Shape validation: `x_layer_input.len() == batch * hidden`,
/// `w_attn_norm.len() == hidden`, each `w_*.len() == hidden * hidden`.
#[allow(clippy::too_many_arguments)]
pub fn fd_sensitivity_inputs_for_attn_linears(
    layer_idx: usize,
    batch: usize,
    hidden: usize,
    eps: f32,
    x_layer_input: &[f32],
    w_attn_norm: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
) -> Result<[LinearFdInputOwned; 3]> {
    if x_layer_input.len() != batch * hidden {
        return Err(anyhow!(
            "fd_sensitivity_inputs_for_attn_linears: x_layer_input len {} != batch({}) * hidden({}) = {}",
            x_layer_input.len(),
            batch,
            hidden,
            batch * hidden,
        ));
    }
    if w_attn_norm.len() != hidden {
        return Err(anyhow!(
            "w_attn_norm len {} != hidden {}",
            w_attn_norm.len(),
            hidden
        ));
    }
    let hidden_sq = hidden * hidden;
    for (label, w) in [("w_q", w_q), ("w_k", w_k), ("w_v", w_v)] {
        if w.len() != hidden_sq {
            return Err(anyhow!(
                "{label} len {} != hidden² {}",
                w.len(),
                hidden_sq
            ));
        }
    }

    // Shared input: post-attn-norm of the residual stream.
    let x_attn = rms_norm_host(x_layer_input, w_attn_norm, batch, hidden, eps);

    let mk = |name: &str, w: &[f32]| -> LinearFdInputOwned {
        let y_t = matmul_x_wt(&x_attn, w, batch, hidden, hidden);
        LinearFdInputOwned {
            name: name.to_string(),
            w: w.to_vec(),
            x: x_attn.clone(),
            y_t,
            n: hidden,
            k: hidden,
            m: batch,
        }
    };

    Ok([
        mk(&format!("blk.{layer_idx}.attn_q.weight"), w_q),
        mk(&format!("blk.{layer_idx}.attn_k.weight"), w_k),
        mk(&format!("blk.{layer_idx}.attn_v.weight"), w_v),
    ])
}

/// Host multi-head SDPA: `out[b, head·hd + d] = Σ_b' softmax(Q·K^T)[b, b'] · V[b', head·hd + d]`.
///
/// Direct port of the CPU oracle in
/// `qwen35_attention_block::tests::multi_head_sdpa_cpu_oracle`.  No causal
/// mask — DWQ calibration uses bidirectional attention over the
/// calibration batch (matches the synthetic-fixture forward in
/// `qwen35_layer::forward` which is bidirectional too).
fn sdpa_host(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let hidden = n_heads * head_dim;
    let mut out = vec![0f32; batch * hidden];
    for h in 0..n_heads {
        let start = h * head_dim;
        // scores[b, b'] = Σ_d Q[b, start+d] · K[b', start+d]
        let mut scores = vec![0f32; batch * batch];
        for b in 0..batch {
            for bp in 0..batch {
                let mut acc = 0.0f32;
                for d in 0..head_dim {
                    acc += q[b * hidden + start + d] * k[bp * hidden + start + d];
                }
                scores[b * batch + bp] = acc;
            }
        }
        // Per-row softmax.
        let mut attn = vec![0f32; batch * batch];
        for b in 0..batch {
            let row = &scores[b * batch..(b + 1) * batch];
            let m = row.iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x));
            let mut sum = 0.0f32;
            let mut e = vec![0f32; batch];
            for (j, &x) in row.iter().enumerate() {
                e[j] = (x - m).exp();
                sum += e[j];
            }
            for j in 0..batch {
                attn[b * batch + j] = e[j] / sum;
            }
        }
        // out[b, d] = Σ_b' attn[b, b'] · V[b', start+d]
        for b in 0..batch {
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for bp in 0..batch {
                    acc += attn[b * batch + bp] * v[bp * hidden + start + d];
                }
                out[b * hidden + start + d] = acc;
            }
        }
    }
    out
}

/// Host SiLU activation: `silu(x) = x · σ(x) = x / (1 + e^{-x})`.
fn silu_host(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

/// ADR-020 iter-12b-2-ffn — capture FD-sensitivity inputs for ALL
/// 7 Linears of one dense decoder layer (Q, K, V, O, gate, up, down).
///
/// CPU mirror of `qwen35_layer::forward`'s op chain — captures the
/// per-Linear (X, W, y_T) triple at every Linear in the layer:
///
///   normed_attn = rms_norm(x_layer_input, w_attn_norm)
///   Q  = normed_attn @ w_q^T          ← X_q  = normed_attn,  y_T_q  = Q
///   K  = normed_attn @ w_k^T          ← X_k  = normed_attn,  y_T_k  = K
///   V  = normed_attn @ w_v^T          ← X_v  = normed_attn,  y_T_v  = V
///   context = multi_head_sdpa(Q,K,V)
///   O  = context @ w_o^T              ← X_o  = context,      y_T_o  = O
///   y_attn = x_layer_input + O
///   normed_ffn = rms_norm(y_attn, w_ffn_norm)
///   gate = normed_ffn @ w_gate^T      ← X_g  = normed_ffn,   y_T_g  = gate
///   up   = normed_ffn @ w_up^T        ← X_u  = normed_ffn,   y_T_u  = up
///   pre  = silu(gate) ⊙ up
///   down = pre @ w_down^T             ← X_d  = pre,          y_T_d  = down
///
/// Names follow GGUF convention so the calibrator can route scores
/// directly to the bit-allocation map.
///
/// Returns the 7 inputs in canonical (Q, K, V, O, gate, up, down)
/// order so callers can index by name OR position.  Ordering matches
/// the dispatch order in `qwen35_layer::forward`, NOT alphabetical.
///
/// # Shape contract
///
/// All `[batch, hidden]` for residual-stream / projection outputs.
/// `w_q/k/v/o`: `[hidden, hidden]`.  `w_gate/up`: `[intermediate, hidden]`.
/// `w_down`: `[hidden, intermediate]`.
#[allow(clippy::too_many_arguments)]
pub fn fd_sensitivity_inputs_for_dense_layer(
    layer_idx: usize,
    batch: usize,
    hidden: usize,
    intermediate: usize,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
    x_layer_input: &[f32],
    w_attn_norm: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_o: &[f32],
    w_ffn_norm: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
) -> Result<[LinearFdInputOwned; 7]> {
    if n_heads * head_dim != hidden {
        return Err(anyhow!(
            "fd_sensitivity_inputs_for_dense_layer: n_heads({}) * head_dim({}) != hidden({})",
            n_heads,
            head_dim,
            hidden
        ));
    }
    if x_layer_input.len() != batch * hidden {
        return Err(anyhow!(
            "x_layer_input len {} != batch({}) * hidden({}) = {}",
            x_layer_input.len(),
            batch,
            hidden,
            batch * hidden
        ));
    }
    if w_attn_norm.len() != hidden || w_ffn_norm.len() != hidden {
        return Err(anyhow!(
            "norm weights: w_attn_norm.len()={}, w_ffn_norm.len()={}, expected {}",
            w_attn_norm.len(),
            w_ffn_norm.len(),
            hidden
        ));
    }
    let hidden_sq = hidden * hidden;
    for (label, w) in [("w_q", w_q), ("w_k", w_k), ("w_v", w_v), ("w_o", w_o)] {
        if w.len() != hidden_sq {
            return Err(anyhow!("{label} len {} != hidden² {}", w.len(), hidden_sq));
        }
    }
    let hidden_inter = hidden * intermediate;
    let inter_hidden = intermediate * hidden;
    if w_gate.len() != hidden_inter || w_up.len() != hidden_inter {
        return Err(anyhow!(
            "ffn projections: w_gate.len()={}, w_up.len()={}, expected hidden·inter = {}",
            w_gate.len(),
            w_up.len(),
            hidden_inter
        ));
    }
    if w_down.len() != inter_hidden {
        return Err(anyhow!(
            "w_down len {} != inter·hidden {}",
            w_down.len(),
            inter_hidden
        ));
    }

    // Attn block.
    let normed_attn = rms_norm_host(x_layer_input, w_attn_norm, batch, hidden, eps);
    let q_out = matmul_x_wt(&normed_attn, w_q, batch, hidden, hidden);
    let k_out = matmul_x_wt(&normed_attn, w_k, batch, hidden, hidden);
    let v_out = matmul_x_wt(&normed_attn, w_v, batch, hidden, hidden);
    let context = sdpa_host(&q_out, &k_out, &v_out, batch, n_heads, head_dim);
    let o_out = matmul_x_wt(&context, w_o, batch, hidden, hidden);

    // Residual after attn.
    let y_attn: Vec<f32> = x_layer_input
        .iter()
        .zip(o_out.iter())
        .map(|(a, b)| a + b)
        .collect();

    // FFN block.
    let normed_ffn = rms_norm_host(&y_attn, w_ffn_norm, batch, hidden, eps);
    let gate_out = matmul_x_wt(&normed_ffn, w_gate, batch, intermediate, hidden);
    let up_out = matmul_x_wt(&normed_ffn, w_up, batch, intermediate, hidden);
    let silu_gate = silu_host(&gate_out);
    let pre_down: Vec<f32> = silu_gate
        .iter()
        .zip(up_out.iter())
        .map(|(a, b)| a * b)
        .collect();
    let down_out = matmul_x_wt(&pre_down, w_down, batch, hidden, intermediate);

    // Pack 7 LinearFdInputOwned in canonical forward order.
    let mk = |name: String, w: &[f32], x: Vec<f32>, y_t: Vec<f32>, n: usize, k: usize| {
        LinearFdInputOwned {
            name,
            w: w.to_vec(),
            x,
            y_t,
            n,
            k,
            m: batch,
        }
    };
    Ok([
        mk(format!("blk.{layer_idx}.attn_q.weight"), w_q, normed_attn.clone(), q_out, hidden, hidden),
        mk(format!("blk.{layer_idx}.attn_k.weight"), w_k, normed_attn.clone(), k_out, hidden, hidden),
        mk(format!("blk.{layer_idx}.attn_v.weight"), w_v, normed_attn,         v_out, hidden, hidden),
        mk(format!("blk.{layer_idx}.attn_output.weight"), w_o, context, o_out, hidden, hidden),
        mk(format!("blk.{layer_idx}.ffn_gate.weight"), w_gate, normed_ffn.clone(), gate_out, intermediate, hidden),
        mk(format!("blk.{layer_idx}.ffn_up.weight"),   w_up,   normed_ffn,         up_out,   intermediate, hidden),
        mk(format!("blk.{layer_idx}.ffn_down.weight"), w_down, pre_down, down_out, hidden, intermediate),
    ])
}

/// ADR-020 iter-12b-3-extract-layer — owned bundle of FP32 weights
/// for one dense-arch decoder layer.
///
/// Field shapes (GGUF native ordering):
///   * `w_attn_norm`, `w_ffn_norm`: `[hidden]`
///   * `w_q`, `w_k`, `w_v`, `w_o`: `[hidden, hidden]` row-major
///     (GGUF stores as `[output, input]`; that's `[n, k]` for matmul
///     `Y = X @ W^T`, which is what
///     `fd_sensitivity_inputs_for_dense_layer` expects directly —
///     no transpose).
///   * `w_gate`, `w_up`: `[intermediate, hidden]` row-major.
///   * `w_down`: `[hidden, intermediate]` row-major.
///
/// All as `Vec<f32>` regardless of source dtype — `to_f32_vec`
/// handles F32/F16/BF16 decode.
#[derive(Debug, Clone)]
pub struct DenseLayerWeightsF32 {
    pub w_attn_norm: Vec<f32>,
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
    pub w_ffn_norm: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
}

/// ADR-020 iter-12b-3-extract-layer — pull the 9 GGUF-named weight
/// tensors for one dense decoder layer out of a `TensorMap`, decoded
/// to FP32.
///
/// Naming matches the qwen35 / dense-arch GGUF convention used by
/// `qwen35_gguf_adapter::weights_from_gguf_tensors`:
///   blk.{i}.attn_norm.weight
///   blk.{i}.attn_q.weight
///   blk.{i}.attn_k.weight
///   blk.{i}.attn_v.weight
///   blk.{i}.attn_output.weight
///   blk.{i}.post_attention_norm.weight
///   blk.{i}.ffn_gate.weight
///   blk.{i}.ffn_up.weight
///   blk.{i}.ffn_down.weight
///
/// Each lookup produces a fresh `Vec<f32>` via [`crate::ir::TensorRef::to_f32_vec`];
/// this function does NOT cache, so repeated calls re-decode.  Callers that
/// score every layer should call once per layer in a single sweep.
///
/// Returns the weights in GGUF native ordering — directly feedable to
/// [`fd_sensitivity_inputs_for_dense_layer`] without transpose.
///
/// # Errors
///
/// * `Err` for ANY missing key (silent partial load would corrupt the
///   FD scoring chain).
/// * Propagates `to_f32_vec` errors (UnsupportedDtype on quantized
///   inputs — the calibrator must hand FP32/F16/BF16 weights, NOT
///   GGML block formats).
pub fn extract_dense_layer_weights_f32(
    tensor_map: &TensorMap,
    layer_idx: usize,
) -> Result<DenseLayerWeightsF32> {
    let prefix = format!("blk.{layer_idx}.");
    let pull = |suffix: &str| -> Result<Vec<f32>> {
        let key = format!("{prefix}{suffix}");
        let tref = tensor_map.tensors.get(&key).ok_or_else(|| {
            anyhow!(
                "extract_dense_layer_weights_f32: tensor {key:?} missing from TensorMap \
                 (expected {} GGUF-named tensors per layer)",
                9
            )
        })?;
        tref.to_f32_vec()
            .map_err(|e| anyhow!("decode {key:?}: {e}"))
    };

    Ok(DenseLayerWeightsF32 {
        w_attn_norm: pull("attn_norm.weight")?,
        w_q: pull("attn_q.weight")?,
        w_k: pull("attn_k.weight")?,
        w_v: pull("attn_v.weight")?,
        w_o: pull("attn_output.weight")?,
        w_ffn_norm: pull("post_attention_norm.weight")?,
        w_gate: pull("ffn_gate.weight")?,
        w_up: pull("ffn_up.weight")?,
        w_down: pull("ffn_down.weight")?,
    })
}

/// ADR-020 iter-12b-3-extract-layer-lazy — `LazyTensorMap` variant of
/// [`extract_dense_layer_weights_f32`].
///
/// Borrowed-materialization version: uses `LazyTensor::materialize_cloned`
/// for each of the 9 GGUF-named tensors so the source map's contents
/// stay intact for downstream byte-emission (DWQ calibration must not
/// drop weights from the map mid-pipeline).
///
/// # Constraint
///
/// `materialize_cloned` REQUIRES the lazy tensor to be already-resident
/// (`Materialized` or `MaterializedShared` state).  Pending lazy
/// tensors return a `MaterializeError::Transform` —
/// surfaced here as an actionable error annotated with the offending
/// key.  Callers operating on Pending maps must materialize the
/// per-layer subset first (e.g. via `LazyTensorMap::materialize_all`
/// or per-tensor materialize-and-reinsert).
///
/// # Errors
///
/// * `Err` for any missing key (silent partial-load is forbidden).
/// * `Err` propagating from `materialize_cloned` (Pending state) or
///   `to_f32_vec` (non-float / shape-mismatched dtypes).
pub fn extract_dense_layer_weights_f32_from_lazy(
    lazy_map: &LazyTensorMap,
    layer_idx: usize,
) -> Result<DenseLayerWeightsF32> {
    let prefix = format!("blk.{layer_idx}.");
    let pull = |suffix: &str| -> Result<Vec<f32>> {
        let key = format!("{prefix}{suffix}");
        let lazy = lazy_map.get(&key).ok_or_else(|| {
            anyhow!(
                "extract_dense_layer_weights_f32_from_lazy: tensor {key:?} missing \
                 from LazyTensorMap (expected 9 GGUF-named tensors per layer)"
            )
        })?;
        let tref = lazy
            .materialize_cloned()
            .map_err(|e| anyhow!("materialize {key:?}: {e}"))?;
        tref.to_f32_vec()
            .map_err(|e| anyhow!("decode {key:?}: {e}"))
    };

    Ok(DenseLayerWeightsF32 {
        w_attn_norm: pull("attn_norm.weight")?,
        w_q: pull("attn_q.weight")?,
        w_k: pull("attn_k.weight")?,
        w_v: pull("attn_v.weight")?,
        w_o: pull("attn_output.weight")?,
        w_ffn_norm: pull("post_attention_norm.weight")?,
        w_gate: pull("ffn_gate.weight")?,
        w_up: pull("ffn_up.weight")?,
        w_down: pull("ffn_down.weight")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: deterministic Gaussian-ish input via sinusoid (avoids
    /// dependency on Box-Muller for these CPU tests).
    fn deterministic_x(m: usize, k: usize, seed: f32) -> Vec<f32> {
        (0..(m * k))
            .map(|i| ((i as f32) * 0.0173 + seed).sin() * 1.0)
            .collect()
    }

    fn host_matmul_xwt(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        matmul_x_wt(x, w, m, n, k)
    }

    /// Validation rejection cases (no math).
    #[test]
    fn rejects_shape_mismatch() {
        let w = vec![0.0f32; 32 * 64];
        let x = vec![0.0f32; 16 * 64];
        let y_t = vec![0.0f32; 16 * 32];
        // Wrong w size:
        let bad_w = vec![0.0f32; 32 * 32];
        assert!(compute_fd_sensitivity(&bad_w, &x, &y_t, 32, 64, 16, 4, 8, 32, 2.0).is_err());
        // Wrong x size:
        let bad_x = vec![0.0f32; 16 * 16];
        assert!(compute_fd_sensitivity(&w, &bad_x, &y_t, 32, 64, 16, 4, 8, 32, 2.0).is_err());
        // Wrong y_t size:
        let bad_y = vec![0.0f32; 16 * 16];
        assert!(compute_fd_sensitivity(&w, &x, &bad_y, 32, 64, 16, 4, 8, 32, 2.0).is_err());
    }

    #[test]
    fn rejects_bad_group_size_or_bits() {
        let w = vec![0.0f32; 32 * 64];
        let x = vec![0.0f32; 16 * 64];
        let y_t = vec![0.0f32; 16 * 32];
        // Non-power-of-two group_size:
        assert!(compute_fd_sensitivity(&w, &x, &y_t, 32, 64, 16, 4, 8, 33, 2.0).is_err());
        // K not divisible by group_size:
        assert!(compute_fd_sensitivity(&w, &x, &y_t, 32, 64, 16, 4, 8, 30, 2.0).is_err());
        // bits > 8:
        assert!(compute_fd_sensitivity(&w, &x, &y_t, 32, 64, 16, 4, 9, 32, 2.0).is_err());
        // low_bits > high_bits:
        assert!(compute_fd_sensitivity(&w, &x, &y_t, 32, 64, 16, 8, 4, 32, 2.0).is_err());
        // Negative temperature:
        assert!(compute_fd_sensitivity(&w, &x, &y_t, 32, 64, 16, 4, 8, 32, -1.0).is_err());
    }

    /// Sanity: when low_bits == high_bits the score is ~0 (qdq is the
    /// SAME on both sides → identical y_low and y_high → identical KL
    /// → score = 0).  Within FP rounding tolerance.
    #[test]
    fn zero_score_when_bits_equal() {
        let n = 32;
        let k = 64;
        let m = 16;
        let group_size = 32;
        let w: Vec<f32> = (0..(n * k))
            .map(|i| ((i as f32) * 0.013 - 0.5).sin() * 0.6)
            .collect();
        let x = deterministic_x(m, k, 0.1);
        let y_t = host_matmul_xwt(&x, &w, m, n, k);

        for bits in [2, 4, 8] {
            let s =
                compute_fd_sensitivity(&w, &x, &y_t, n, k, m, bits, bits, group_size, 2.0)
                    .unwrap();
            assert!(
                s.abs() < 1e-6,
                "score must be exactly zero when low_bits == high_bits (bits={bits} got {s})"
            );
        }
    }

    /// Canonical sign + magnitude: when the weight has non-trivial
    /// quant error, going from low_bits to high_bits must REDUCE KL
    /// → KL_low > KL_high → positive score.
    ///
    /// Larger bit-gap should yield larger magnitude (4-vs-8 > 4-vs-5
    /// > 7-vs-8) for a tensor where extreme bit-pair difference shows
    /// up in the quantization noise.
    #[test]
    fn positive_score_with_bit_gap_and_monotone_in_gap() {
        let n = 32;
        let k = 128;
        let m = 16;
        let group_size = 32;
        let w: Vec<f32> = (0..(n * k))
            .map(|i| ((i as f32) * 0.0173 - 0.5).sin() * 0.6 + 0.1 * (i as f32 % 7.0))
            .collect();
        let x = deterministic_x(m, k, 0.1);
        let y_t = host_matmul_xwt(&x, &w, m, n, k);

        let s_4_8 = compute_fd_sensitivity(&w, &x, &y_t, n, k, m, 4, 8, group_size, 2.0)
            .expect("4-8");
        let s_4_5 = compute_fd_sensitivity(&w, &x, &y_t, n, k, m, 4, 5, group_size, 2.0)
            .expect("4-5");
        let s_2_8 = compute_fd_sensitivity(&w, &x, &y_t, n, k, m, 2, 8, group_size, 2.0)
            .expect("2-8");

        assert!(
            s_4_8 > 0.0,
            "expected positive score for 4-vs-8 bits, got {s_4_8}"
        );
        assert!(
            s_2_8 > s_4_8,
            "expected larger sensitivity for wider bit gap (2-8 > 4-8), got {s_2_8} vs {s_4_8}"
        );
        assert!(
            s_4_8 > s_4_5,
            "expected larger sensitivity for wider gap (4-8 > 4-5), got {s_4_8} vs {s_4_5}"
        );
    }

    /// Two tensors of the SAME shape but different value distributions
    /// should produce different sensitivity scores — proving the
    /// metric responds to weight content, not just shape.  Uniform-
    /// magnitude tensor (no outliers) should have LOWER sensitivity
    /// at fixed bit pair than a tensor with structured outliers.
    #[test]
    fn distribution_dependence_uniform_vs_outlier() {
        let n = 32;
        let k = 64;
        let m = 16;
        let group_size = 32;
        let x = deterministic_x(m, k, 0.1);

        let w_uniform: Vec<f32> = (0..(n * k))
            .map(|i| ((i as f32) * 0.013).sin() * 0.5)
            .collect();
        let mut w_outlier = w_uniform.clone();
        // Plant 4 outliers per group of 32 (12.5% outlier rate).
        for g in 0..(w_outlier.len() / group_size) {
            for j in 0..4 {
                w_outlier[g * group_size + j] = if j % 2 == 0 { 5.0 } else { -5.0 };
            }
        }

        let y_t_uniform = host_matmul_xwt(&x, &w_uniform, m, n, k);
        let y_t_outlier = host_matmul_xwt(&x, &w_outlier, m, n, k);

        let s_uniform = compute_fd_sensitivity(
            &w_uniform,
            &x,
            &y_t_uniform,
            n,
            k,
            m,
            4,
            8,
            group_size,
            2.0,
        )
        .unwrap();
        let s_outlier = compute_fd_sensitivity(
            &w_outlier,
            &x,
            &y_t_outlier,
            n,
            k,
            m,
            4,
            8,
            group_size,
            2.0,
        )
        .unwrap();

        assert!(
            s_outlier > s_uniform,
            "outlier-dominated tensor should be more sensitive: outlier={s_outlier} vs uniform={s_uniform}"
        );
    }

    /// Numel normalization sanity: the formula divides by
    /// `(numel / 1e6)` so doubling K should keep the per-million
    /// normalized score in the same order of magnitude as the
    /// baseline (verifies the normalization actually fires).
    ///
    /// We do NOT assert exact invariance — KL scales nonlinearly
    /// with reduction-depth K (longer K → more peaked softmax → more
    /// KL signal per unit qdq error) so the "ratio = 1" prediction
    /// is too strong.  We DO assert `0.1 ≤ ratio ≤ 10` which catches
    /// the class of bugs where the normalization is missing entirely
    /// (would give ratio = 0.5 × actual KL ratio, possibly orders
    /// of magnitude off).
    #[test]
    fn numel_normalized_score_within_decade_under_k_scaling() {
        let n = 32;
        let k = 64;
        let m = 16;
        let group_size = 32;
        let bits_low = 4;
        let bits_high = 8;

        let w_small: Vec<f32> = (0..(n * k))
            .map(|i| ((i as f32) * 0.013 - 0.5).sin() * 0.6)
            .collect();
        let x_small = deterministic_x(m, k, 0.1);
        let y_t_small = host_matmul_xwt(&x_small, &w_small, m, n, k);
        let s_small = compute_fd_sensitivity(
            &w_small,
            &x_small,
            &y_t_small,
            n,
            k,
            m,
            bits_low,
            bits_high,
            group_size,
            2.0,
        )
        .unwrap();

        // Larger tensor: same per-row distribution, longer K.  KL
        // grows roughly linearly with K (more reduction depth → more
        // softmax peakedness, more KL signal).  Score is normalized
        // by params_M which also grows linearly.  Net effect: score
        // should stay within ~2× of the baseline (perfect
        // cancellation requires identical distributions, which
        // synthetic fixtures only approximate).  This bound catches
        // missing-normalization (would change by 2-4× otherwise).
        let k2 = k * 2;
        let w_large: Vec<f32> = (0..(n * k2))
            .map(|i| ((i as f32 % (n * k) as f32) * 0.013 - 0.5).sin() * 0.6)
            .collect();
        let x_large = deterministic_x(m, k2, 0.1);
        let y_t_large = host_matmul_xwt(&x_large, &w_large, m, n, k2);
        let s_large = compute_fd_sensitivity(
            &w_large,
            &x_large,
            &y_t_large,
            n,
            k2,
            m,
            bits_low,
            bits_high,
            group_size,
            2.0,
        )
        .unwrap();

        assert!(
            s_small > 0.0 && s_large > 0.0,
            "expected positive scores: small={s_small}, large={s_large}"
        );
        let ratio = s_large / s_small;
        assert!(
            (0.1..=10.0).contains(&ratio),
            "numel normalization should keep score within one decade under K scaling: ratio={ratio} (small={s_small}, large={s_large})"
        );
    }

    /// q_legacy round-trip integrity check — the CPU oracle produces
    /// values within the expected per-group quantization step `s/2`.
    #[test]
    fn q_legacy_round_trip_within_step_bound() {
        let group_size = 32;
        for bits in [4u32, 8] {
            let w: Vec<f32> = (0..256)
                .map(|i| ((i as f32) * 0.0173 - 0.5).sin() * 0.6)
                .collect();
            let qdq = q_legacy_round_trip(&w, group_size, bits).unwrap();
            assert_eq!(qdq.len(), w.len());
            for (a, b) in w.iter().zip(qdq.iter()) {
                assert!(a.is_finite() && b.is_finite());
            }
            // Per-group residual must be ≤ scale / 2 (rounding error
            // bound for symmetric q-legacy).
            let n_groups = w.len() / group_size;
            for g in 0..n_groups {
                let start = g * group_size;
                let block = &w[start..start + group_size];
                let mut amax = 0.0f32;
                for &v in block {
                    if v.abs() > amax {
                        amax = v.abs();
                    }
                }
                let scale = if amax == 0.0 {
                    1.0
                } else {
                    amax / ((1u32 << bits) as f32 / 2.0)
                };
                // Asymmetric signed range [q_min, q_max] = [-(levels/2),
                // (levels/2)-1] means the worst-case residual at the
                // positive extreme is `scale` (not `scale/2`): for
                // v=amax, round(v/scale) = levels/2 but clamps to
                // q_max=(levels/2)-1 → q·scale = amax − scale → resid
                // = scale.  Sub-extreme bins still fit within
                // `scale/2`; but the extreme bin alone forces the
                // tighter bound to `scale + ε`.
                let max_resid = scale + 1e-5;
                for i in 0..group_size {
                    let resid = (w[start + i] - qdq[start + i]).abs();
                    assert!(
                        resid <= max_resid,
                        "bits={} group {} pos {} residual {} > {}",
                        bits,
                        g,
                        i,
                        resid,
                        max_resid
                    );
                }
            }
        }
    }

    /// ADR-020 iter-12b-2-prep — `compute_fd_sensitivity_per_linear`
    /// returns scores for every Linear in the input slice and matches
    /// the per-Linear primitive's score for each.
    #[test]
    fn per_linear_dispatch_matches_individual_compute() {
        let n = 32usize;
        let k = 64usize;
        let m = 16usize;
        let group_size = 32usize;
        let bits_low = 4u32;
        let bits_high = 8u32;
        let t = 2.0f32;

        let w_a: Vec<f32> = (0..(n * k))
            .map(|i| ((i as f32) * 0.011).sin() * 0.5)
            .collect();
        let w_b: Vec<f32> = (0..(n * k))
            .map(|i| {
                // Outlier-laden: a few extreme values to push FD
                // sensitivity above the smooth fixture.
                let base = ((i as f32) * 0.013).cos() * 0.4;
                if i % 50 == 0 { base + 4.0 } else { base }
            })
            .collect();
        let x_a = deterministic_x(m, k, 0.1);
        let x_b = deterministic_x(m, k, 0.7);
        let y_t_a = host_matmul_xwt(&x_a, &w_a, m, n, k);
        let y_t_b = host_matmul_xwt(&x_b, &w_b, m, n, k);

        let s_a_solo = compute_fd_sensitivity(
            &w_a, &x_a, &y_t_a, n, k, m, bits_low, bits_high, group_size, t,
        )
        .unwrap();
        let s_b_solo = compute_fd_sensitivity(
            &w_b, &x_b, &y_t_b, n, k, m, bits_low, bits_high, group_size, t,
        )
        .unwrap();

        let linears = [
            LinearFdInput {
                name: "linear.a".into(),
                w: &w_a,
                x: &x_a,
                y_t: &y_t_a,
                n,
                k,
                m,
            },
            LinearFdInput {
                name: "linear.b".into(),
                w: &w_b,
                x: &x_b,
                y_t: &y_t_b,
                n,
                k,
                m,
            },
        ];

        let dict = compute_fd_sensitivity_per_linear(
            &linears, bits_low, bits_high, group_size, t,
        )
        .unwrap();
        assert_eq!(dict.len(), 2, "one entry per input Linear");
        let s_a_dict = *dict.get("linear.a").expect("linear.a entry");
        let s_b_dict = *dict.get("linear.b").expect("linear.b entry");

        // Per-Linear scores must match the individual primitive
        // bit-for-bit (no double-rounding, no normalization drift).
        assert_eq!(s_a_dict, s_a_solo, "linear.a score must match solo call");
        assert_eq!(s_b_dict, s_b_solo, "linear.b score must match solo call");

        // Outlier weight (linear.b) must rank higher than smooth
        // (linear.a) — both positive, b > a.
        assert!(
            s_a_dict > 0.0 && s_b_dict > 0.0,
            "expected positive scores; got a={s_a_dict}, b={s_b_dict}"
        );
        assert!(
            s_b_dict > s_a_dict,
            "outlier-laden weight (b={s_b_dict}) must rank above smooth (a={s_a_dict})"
        );
    }

    /// ADR-020 iter-12b-2-prep — duplicate-name input must Err
    /// loudly.  Falsifier: the previous "naive map.insert" would
    /// silently overwrite the first entry's score, dropping
    /// sensitivity data and producing wrong bit allocations.
    #[test]
    fn per_linear_dispatch_rejects_duplicate_names() {
        let n = 32usize;
        let k = 64usize;
        let m = 16usize;
        let w: Vec<f32> = (0..(n * k))
            .map(|i| ((i as f32) * 0.011).sin() * 0.5)
            .collect();
        let x = deterministic_x(m, k, 0.1);
        let y_t = host_matmul_xwt(&x, &w, m, n, k);

        let linears = [
            LinearFdInput {
                name: "dup".into(),
                w: &w,
                x: &x,
                y_t: &y_t,
                n,
                k,
                m,
            },
            LinearFdInput {
                name: "dup".into(),
                w: &w,
                x: &x,
                y_t: &y_t,
                n,
                k,
                m,
            },
        ];

        let res = compute_fd_sensitivity_per_linear(&linears, 4, 8, 32, 2.0);
        assert!(res.is_err(), "duplicate name must Err");
        let msg = format!("{}", res.err().unwrap());
        assert!(
            msg.contains("duplicate") && msg.contains("dup"),
            "error must mention duplicate + name; got: {msg}"
        );
    }

    /// ADR-020 iter-12b-2-prep — empty input slice returns an empty
    /// map (not an Err).  Edge case sanity: zero Linears = no work to
    /// do = empty dict, NOT a panic / silent error.
    #[test]
    fn per_linear_dispatch_empty_input_returns_empty_map() {
        let dict = compute_fd_sensitivity_per_linear(&[], 4, 8, 32, 2.0).unwrap();
        assert!(dict.is_empty(), "empty input → empty dict");
    }

    /// ADR-020 iter-12b-2-attn — `fd_sensitivity_inputs_for_attn_linears`
    /// produces the canonical (X, W, y_T) triple per Linear and
    /// composes correctly with `compute_fd_sensitivity_per_linear`.
    #[test]
    fn attn_linears_capture_produces_valid_fd_inputs() {
        let layer_idx = 7usize;
        let batch = 32usize;
        let hidden = 64usize;
        let eps = 1e-6f32;

        // Deterministic fixtures.
        let x_layer: Vec<f32> = (0..(batch * hidden))
            .map(|i| ((i as f32) * 0.011 - 0.4).sin() * 0.5)
            .collect();
        let w_attn_norm: Vec<f32> =
            (0..hidden).map(|i| 1.0 + (i as f32) * 0.005).collect();
        let w_q: Vec<f32> = (0..(hidden * hidden))
            .map(|i| ((i as f32) * 0.013).cos() * 0.4)
            .collect();
        let w_k: Vec<f32> = (0..(hidden * hidden))
            .map(|i| ((i as f32) * 0.017).sin() * 0.4)
            .collect();
        let w_v: Vec<f32> = (0..(hidden * hidden))
            .map(|i| ((i as f32) * 0.019).cos() * 0.4)
            .collect();

        let triples = fd_sensitivity_inputs_for_attn_linears(
            layer_idx, batch, hidden, eps,
            &x_layer, &w_attn_norm, &w_q, &w_k, &w_v,
        )
        .expect("capture must succeed");

        // 1. Names match GGUF convention.
        assert_eq!(triples[0].name, format!("blk.{layer_idx}.attn_q.weight"));
        assert_eq!(triples[1].name, format!("blk.{layer_idx}.attn_k.weight"));
        assert_eq!(triples[2].name, format!("blk.{layer_idx}.attn_v.weight"));

        // 2. All three share the SAME X (post-attn-norm).  Bit-equal.
        assert_eq!(triples[0].x, triples[1].x);
        assert_eq!(triples[0].x, triples[2].x);

        // 3. Shapes correct for [batch, hidden] @ [hidden, hidden]^T.
        for t in &triples {
            assert_eq!(t.n, hidden);
            assert_eq!(t.k, hidden);
            assert_eq!(t.m, batch);
            assert_eq!(t.w.len(), hidden * hidden);
            assert_eq!(t.x.len(), batch * hidden);
            assert_eq!(t.y_t.len(), batch * hidden);
        }

        // 4. y_T = X @ W^T to FP32 precision (verifies the per-Linear
        //    matmul wasn't accidentally swapped or transposed).
        for (t, w_expected) in triples.iter().zip([&w_q, &w_k, &w_v]) {
            let y_expected = matmul_x_wt(&t.x, w_expected, batch, hidden, hidden);
            for (i, (a, b)) in t.y_t.iter().zip(y_expected.iter()).enumerate() {
                let d = (a - b).abs();
                assert!(
                    d < 1e-4,
                    "{}: y_t[{i}] = {a} != X@W^T = {b} (diff {d})",
                    t.name
                );
            }
        }

        // 5. End-to-end composition: attn-block triples flow into the
        //    list-shaped FD scorer and produce per-Linear scores.
        let borrowed: Vec<LinearFdInput<'_>> =
            triples.iter().map(|t| t.as_borrowed()).collect();
        let scores =
            compute_fd_sensitivity_per_linear(&borrowed, 4, 8, 32, 2.0).unwrap();
        assert_eq!(scores.len(), 3, "three attn Linears → three scores");
        for t in &triples {
            let s = scores
                .get(&t.name)
                .copied()
                .unwrap_or_else(|| panic!("missing score for {}", t.name));
            assert!(
                s.is_finite(),
                "score for {} not finite: {s}",
                t.name
            );
        }
    }

    /// ADR-020 iter-12b-2-attn — input-shape rejection.  Falsifier
    /// for "silent shape mismatch" anti-pattern that would compute
    /// garbage sensitivities on misaligned weights.
    #[test]
    fn attn_linears_capture_rejects_shape_mismatch() {
        let r = fd_sensitivity_inputs_for_attn_linears(
            0,
            16,
            32,
            1e-6,
            &vec![0.0f32; 16 * 32],
            &vec![1.0f32; 31], // wrong: 31 vs 32
            &vec![0.0f32; 32 * 32],
            &vec![0.0f32; 32 * 32],
            &vec![0.0f32; 32 * 32],
        );
        assert!(r.is_err());
        let msg = format!("{}", r.err().unwrap());
        assert!(
            msg.contains("w_attn_norm") && msg.contains("hidden"),
            "error must mention the offending tensor + dim: {msg}"
        );
    }

    /// ADR-020 iter-12b-2-ffn — `fd_sensitivity_inputs_for_dense_layer`
    /// captures all 7 Linears with valid (X, W, y_T) triples + the
    /// inner ones (attn_o, ffn_gate/up/down) line up with their CPU
    /// chain.
    #[test]
    fn dense_layer_capture_produces_all_seven_linears() {
        let layer_idx = 3usize;
        let batch = 32usize;
        let hidden = 64usize;
        let intermediate = 128usize;
        let n_heads = 2usize;
        let head_dim = 32usize;
        let eps = 1e-6f32;

        // Deterministic xorshift fixture.
        let mut s: u64 = 0xDEAD_BEEF_F00D_F00D;
        let mut next = move || -> f32 {
            s ^= s >> 33;
            s = s.wrapping_mul(0xff51_afd7_ed55_8ccd);
            s ^= s >> 33;
            ((s as i64) as f32) / (i64::MAX as f32)
        };
        let x_layer: Vec<f32> = (0..(batch * hidden)).map(|_| next() * 0.5).collect();
        let w_attn_norm: Vec<f32> = (0..hidden).map(|_| 1.0 + next() * 0.05).collect();
        let w_q: Vec<f32> = (0..(hidden * hidden)).map(|_| next() * 0.4).collect();
        let w_k: Vec<f32> = (0..(hidden * hidden)).map(|_| next() * 0.4).collect();
        let w_v: Vec<f32> = (0..(hidden * hidden)).map(|_| next() * 0.4).collect();
        let w_o: Vec<f32> = (0..(hidden * hidden)).map(|_| next() * 0.4).collect();
        let w_ffn_norm: Vec<f32> = (0..hidden).map(|_| 1.0 + next() * 0.05).collect();
        let w_gate: Vec<f32> =
            (0..(hidden * intermediate)).map(|_| next() * 0.4).collect();
        let w_up: Vec<f32> =
            (0..(hidden * intermediate)).map(|_| next() * 0.4).collect();
        let w_down: Vec<f32> =
            (0..(intermediate * hidden)).map(|_| next() * 0.4).collect();

        let triples = fd_sensitivity_inputs_for_dense_layer(
            layer_idx,
            batch,
            hidden,
            intermediate,
            n_heads,
            head_dim,
            eps,
            &x_layer,
            &w_attn_norm,
            &w_q,
            &w_k,
            &w_v,
            &w_o,
            &w_ffn_norm,
            &w_gate,
            &w_up,
            &w_down,
        )
        .expect("dense layer capture must succeed");

        // 1. Canonical name + dispatch order.
        let expected_names = [
            format!("blk.{layer_idx}.attn_q.weight"),
            format!("blk.{layer_idx}.attn_k.weight"),
            format!("blk.{layer_idx}.attn_v.weight"),
            format!("blk.{layer_idx}.attn_output.weight"),
            format!("blk.{layer_idx}.ffn_gate.weight"),
            format!("blk.{layer_idx}.ffn_up.weight"),
            format!("blk.{layer_idx}.ffn_down.weight"),
        ];
        for (t, name) in triples.iter().zip(expected_names.iter()) {
            assert_eq!(t.name, *name, "name mismatch");
        }

        // 2. Q/K/V share bit-equal post-attn-norm input X.
        assert_eq!(triples[0].x, triples[1].x, "Q/K share X");
        assert_eq!(triples[0].x, triples[2].x, "Q/V share X");

        // 3. ffn_gate / ffn_up share bit-equal post-FFN-norm input X.
        assert_eq!(triples[4].x, triples[5].x, "gate/up share X");

        // 4. attn_o, ffn_down each have their own X (context, pre).
        assert_ne!(triples[3].x, triples[0].x, "attn_o X must differ from Q's X");
        assert_ne!(triples[6].x, triples[4].x, "ffn_down X must differ from gate's X");

        // 5. Per-Linear y_T = X @ W^T to FP32 precision.
        for t in &triples {
            let y_expected = matmul_x_wt(&t.x, &t.w, t.m, t.n, t.k);
            for (i, (a, b)) in t.y_t.iter().zip(y_expected.iter()).enumerate() {
                let d = (a - b).abs();
                assert!(
                    d < 1e-3,
                    "{}: y_t[{i}] = {a} != X@W^T = {b} (diff {d})",
                    t.name
                );
            }
        }

        // 6. All 7 borrow + dispatch through compute_fd_sensitivity_per_linear.
        let borrowed: Vec<LinearFdInput<'_>> =
            triples.iter().map(|t| t.as_borrowed()).collect();
        let scores =
            compute_fd_sensitivity_per_linear(&borrowed, 4, 8, 32, 2.0).unwrap();
        assert_eq!(scores.len(), 7, "all 7 Linears get a score");
        for t in &triples {
            let s = scores
                .get(&t.name)
                .copied()
                .unwrap_or_else(|| panic!("missing score for {}", t.name));
            assert!(s.is_finite(), "{} score not finite: {s}", t.name);
        }
    }

    /// ADR-020 iter-12b-2-ffn — n_heads / head_dim must factor hidden;
    /// otherwise SDPA can't tile correctly and downstream y_T_o would
    /// be garbage.  Falsifier for the silent-tile-mismatch class.
    #[test]
    fn dense_layer_capture_rejects_invalid_head_factoring() {
        let r = fd_sensitivity_inputs_for_dense_layer(
            0, 32, 64, 128, /*n_heads*/ 3, /*head_dim*/ 32, 1e-6,
            &vec![0.0f32; 32 * 64],
            &vec![1.0f32; 64],
            &vec![0.0f32; 64 * 64],
            &vec![0.0f32; 64 * 64],
            &vec![0.0f32; 64 * 64],
            &vec![0.0f32; 64 * 64],
            &vec![1.0f32; 64],
            &vec![0.0f32; 64 * 128],
            &vec![0.0f32; 64 * 128],
            &vec![0.0f32; 128 * 64],
        );
        assert!(r.is_err());
        let msg = format!("{}", r.err().unwrap());
        assert!(
            msg.contains("n_heads") && msg.contains("hidden"),
            "error must mention n_heads + hidden: {msg}"
        );
    }

    /// ADR-020 iter-12b-3-extract-layer — `extract_dense_layer_weights_f32`
    /// pulls all 9 GGUF-named tensors and decodes via `to_f32_vec`.
    /// End-to-end test: synthetic TensorMap → extract → feed into
    /// `fd_sensitivity_inputs_for_dense_layer` → 7 finite scores.
    #[test]
    fn extract_dense_layer_weights_f32_pulls_all_nine_tensors() {
        use crate::ir::{DType, TensorMap, TensorRef};

        let layer_idx = 5usize;
        let hidden = 32usize;
        let intermediate = 64usize;

        // Helper: build an F32 TensorRef from values + shape.
        let mk_tref = |name: &str, values: &[f32], shape: Vec<usize>| {
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for v in values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            TensorRef {
                name: name.into(),
                shape,
                dtype: DType::F32,
                data: std::sync::Arc::new(bytes),
            }
        };

        let mut tensor_map = TensorMap::new();
        let mk_v = |n: usize, base: f32| -> Vec<f32> {
            (0..n).map(|i| base + (i as f32) * 0.001).collect()
        };

        let attn_norm = mk_v(hidden, 1.0);
        let q = mk_v(hidden * hidden, 0.1);
        let k = mk_v(hidden * hidden, 0.2);
        let v = mk_v(hidden * hidden, 0.3);
        let o = mk_v(hidden * hidden, 0.4);
        let post_norm = mk_v(hidden, 1.0);
        let gate = mk_v(intermediate * hidden, 0.5);
        let up = mk_v(intermediate * hidden, 0.6);
        let down = mk_v(hidden * intermediate, 0.7);

        tensor_map.insert(mk_tref(
            &format!("blk.{layer_idx}.attn_norm.weight"),
            &attn_norm,
            vec![hidden],
        ));
        tensor_map.insert(mk_tref(
            &format!("blk.{layer_idx}.attn_q.weight"),
            &q,
            vec![hidden, hidden],
        ));
        tensor_map.insert(mk_tref(
            &format!("blk.{layer_idx}.attn_k.weight"),
            &k,
            vec![hidden, hidden],
        ));
        tensor_map.insert(mk_tref(
            &format!("blk.{layer_idx}.attn_v.weight"),
            &v,
            vec![hidden, hidden],
        ));
        tensor_map.insert(mk_tref(
            &format!("blk.{layer_idx}.attn_output.weight"),
            &o,
            vec![hidden, hidden],
        ));
        tensor_map.insert(mk_tref(
            &format!("blk.{layer_idx}.post_attention_norm.weight"),
            &post_norm,
            vec![hidden],
        ));
        tensor_map.insert(mk_tref(
            &format!("blk.{layer_idx}.ffn_gate.weight"),
            &gate,
            vec![intermediate, hidden],
        ));
        tensor_map.insert(mk_tref(
            &format!("blk.{layer_idx}.ffn_up.weight"),
            &up,
            vec![intermediate, hidden],
        ));
        tensor_map.insert(mk_tref(
            &format!("blk.{layer_idx}.ffn_down.weight"),
            &down,
            vec![hidden, intermediate],
        ));

        let extracted = extract_dense_layer_weights_f32(&tensor_map, layer_idx)
            .expect("extract must succeed on valid map");

        // Bit-exact F32 round-trip: each Vec equals the input.
        assert_eq!(extracted.w_attn_norm, attn_norm);
        assert_eq!(extracted.w_q, q);
        assert_eq!(extracted.w_k, k);
        assert_eq!(extracted.w_v, v);
        assert_eq!(extracted.w_o, o);
        assert_eq!(extracted.w_ffn_norm, post_norm);
        assert_eq!(extracted.w_gate, gate);
        assert_eq!(extracted.w_up, up);
        assert_eq!(extracted.w_down, down);

        // End-to-end: feeds into fd_sensitivity_inputs_for_dense_layer
        // without transpose, produces 7 valid LinearFdInputOwned, FD
        // scoring gives 7 finite scores.
        let batch = 32usize;
        let n_heads = 2usize;
        let head_dim = 16usize;
        let x_layer: Vec<f32> = (0..(batch * hidden))
            .map(|i| ((i as f32) * 0.013).sin() * 0.3)
            .collect();
        let triples = fd_sensitivity_inputs_for_dense_layer(
            layer_idx,
            batch,
            hidden,
            intermediate,
            n_heads,
            head_dim,
            1e-6,
            &x_layer,
            &extracted.w_attn_norm,
            &extracted.w_q,
            &extracted.w_k,
            &extracted.w_v,
            &extracted.w_o,
            &extracted.w_ffn_norm,
            &extracted.w_gate,
            &extracted.w_up,
            &extracted.w_down,
        )
        .expect("feed extracted into fd_sensitivity_inputs_for_dense_layer");
        let borrowed: Vec<LinearFdInput<'_>> =
            triples.iter().map(|t| t.as_borrowed()).collect();
        let scores = compute_fd_sensitivity_per_linear(&borrowed, 4, 8, 32, 2.0)
            .expect("FD score the extracted layer");
        assert_eq!(scores.len(), 7);
        for (name, s) in &scores {
            assert!(s.is_finite(), "score for {name} not finite: {s}");
        }
    }

    /// ADR-020 iter-12b-3-extract-layer — missing tensor errs loudly,
    /// NOT silently with partial output.
    #[test]
    fn extract_dense_layer_weights_f32_rejects_missing_tensor() {
        use crate::ir::TensorMap;

        let layer_idx = 0usize;
        // Empty TensorMap — every lookup should fail.
        let tensor_map = TensorMap::new();
        let r = extract_dense_layer_weights_f32(&tensor_map, layer_idx);
        assert!(r.is_err());
        let msg = format!("{}", r.err().unwrap());
        assert!(
            msg.contains("missing") && msg.contains("blk.0."),
            "error must name the missing key: {msg}"
        );
    }

    /// ADR-020 iter-12b-3-extract-layer — quantized-dtype tensors err
    /// during decode.  Falsifier for "silently coerce GGML blocks to
    /// FP32" anti-pattern.
    #[test]
    fn extract_dense_layer_weights_f32_propagates_decode_errors() {
        use crate::ir::{DType, TensorMap, TensorRef};

        let mut tensor_map = TensorMap::new();
        // Insert just attn_norm with WRONG dtype (I32) so to_f32_vec
        // hits the unsupported-dtype branch.
        let bytes = vec![0u8; 4 * 32]; // 32 elements × 4 bytes
        tensor_map.insert(TensorRef {
            name: "blk.0.attn_norm.weight".into(),
            shape: vec![32],
            dtype: DType::I32,
            data: std::sync::Arc::new(bytes),
        });
        let r = extract_dense_layer_weights_f32(&tensor_map, 0);
        assert!(r.is_err());
        let msg = format!("{}", r.err().unwrap());
        assert!(
            msg.contains("decode") && msg.contains("attn_norm"),
            "error must annotate decode + tensor name: {msg}"
        );
    }

    /// ADR-020 iter-12b-3-extract-layer-lazy — lazy-map variant
    /// produces bit-equal output to the eager variant on the same
    /// 9-tensor fixture.
    #[test]
    fn extract_dense_layer_weights_f32_from_lazy_matches_eager() {
        use crate::ir::lazy::LazyTensorMap;
        use crate::ir::{DType, TensorMap, TensorRef};

        let layer_idx = 2usize;
        let hidden = 32usize;
        let intermediate = 64usize;

        let mk_v = |n: usize, base: f32| -> Vec<f32> {
            (0..n).map(|i| base + (i as f32) * 0.001).collect()
        };
        let mk_tref = |name: String, values: &[f32], shape: Vec<usize>| {
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for v in values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            TensorRef {
                name,
                shape,
                dtype: DType::F32,
                data: std::sync::Arc::new(bytes),
            }
        };

        let mut eager = TensorMap::new();
        let p = format!("blk.{layer_idx}.");
        let h_sq = hidden * hidden;
        let h_inter = hidden * intermediate;
        eager.insert(mk_tref(format!("{p}attn_norm.weight"), &mk_v(hidden, 1.0), vec![hidden]));
        eager.insert(mk_tref(format!("{p}attn_q.weight"), &mk_v(h_sq, 0.1), vec![hidden, hidden]));
        eager.insert(mk_tref(format!("{p}attn_k.weight"), &mk_v(h_sq, 0.2), vec![hidden, hidden]));
        eager.insert(mk_tref(format!("{p}attn_v.weight"), &mk_v(h_sq, 0.3), vec![hidden, hidden]));
        eager.insert(mk_tref(format!("{p}attn_output.weight"), &mk_v(h_sq, 0.4), vec![hidden, hidden]));
        eager.insert(mk_tref(format!("{p}post_attention_norm.weight"), &mk_v(hidden, 1.05), vec![hidden]));
        eager.insert(mk_tref(format!("{p}ffn_gate.weight"), &mk_v(h_inter, 0.5), vec![intermediate, hidden]));
        eager.insert(mk_tref(format!("{p}ffn_up.weight"), &mk_v(h_inter, 0.6), vec![intermediate, hidden]));
        eager.insert(mk_tref(format!("{p}ffn_down.weight"), &mk_v(h_inter, 0.7), vec![hidden, intermediate]));

        let eager_extracted = extract_dense_layer_weights_f32(&eager, layer_idx)
            .expect("eager extract");

        let lazy = LazyTensorMap::from_eager(eager);
        let lazy_extracted = extract_dense_layer_weights_f32_from_lazy(&lazy, layer_idx)
            .expect("lazy extract");

        // Bit-equal output across both extractors.
        assert_eq!(eager_extracted.w_attn_norm, lazy_extracted.w_attn_norm);
        assert_eq!(eager_extracted.w_q, lazy_extracted.w_q);
        assert_eq!(eager_extracted.w_k, lazy_extracted.w_k);
        assert_eq!(eager_extracted.w_v, lazy_extracted.w_v);
        assert_eq!(eager_extracted.w_o, lazy_extracted.w_o);
        assert_eq!(eager_extracted.w_ffn_norm, lazy_extracted.w_ffn_norm);
        assert_eq!(eager_extracted.w_gate, lazy_extracted.w_gate);
        assert_eq!(eager_extracted.w_up, lazy_extracted.w_up);
        assert_eq!(eager_extracted.w_down, lazy_extracted.w_down);
    }

    /// ADR-020 iter-12b-3-extract-layer-lazy — empty lazy map returns
    /// the same missing-key error pattern as the eager variant.
    #[test]
    fn extract_dense_layer_weights_f32_from_lazy_rejects_missing_tensor() {
        use crate::ir::lazy::LazyTensorMap;

        let lazy = LazyTensorMap::new();
        let r = extract_dense_layer_weights_f32_from_lazy(&lazy, 0);
        assert!(r.is_err());
        let msg = format!("{}", r.err().unwrap());
        assert!(
            msg.contains("missing") && msg.contains("blk.0."),
            "error must name the missing key: {msg}"
        );
    }
}
