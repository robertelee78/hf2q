//! Qwen 3.5 attention block — GpuTape composition for ADR-020 iter-10c.
//!
//! Composes the standard transformer attention block on hf2q's GPU
//! autograd tape:
//!
//! ```text
//!   normed   = rms_norm(x, w_n, eps)
//!   q        = matmul(normed, W_q)               // [batch, head_dim]
//!   k        = matmul(normed, W_k)               // [batch, head_dim]
//!   v        = matmul(normed, W_v)               // [batch, head_dim]
//!   k_t      = transpose(k)                      // [head_dim, batch]
//!   scores   = matmul(q, k_t)                    // [batch, batch]
//!     (the 1/sqrt(d_head) attention scale folds into W_q at construction)
//!   attn     = softmax(scores)                   // [batch, batch]
//!   context  = matmul(attn, v)                   // [batch, head_dim]
//!   out      = matmul(context, W_o)              // [batch, hidden]
//! ```
//!
//! The four projection weights `W_q`, `W_k`, `W_v`, `W_o` are the
//! quantizable Linears.  RMSNorm weight `w_n` is frozen
//! (not quantized — matches mlx-lm `dynamic_quant.py:60-61` where
//! only `to_quantized` modules are unfrozen for gradient computation).
//!
//! For ADR-020 Track 1 sensitivity scoring:
//! 1. Each `W_*` lands on the tape as the qdq'd low-bit version (W_low).
//! 2. We pre-compute the high-bit `W_high` separately for the
//!    sensitivity-formula `(grad · (W_low − W_high)).sum() / (numel/1e6)`
//!    via `estimate_sensitivities`.
//!
//! Single-head construction (n_heads=1) — multi-head SDPA composes
//! the same primitives at scale; the single-head case is the
//! load-bearing autograd test fixture (all dimensions ≥ 32 to clear
//! the matmul backward kernel's k/m floors).

use anyhow::{anyhow, Result};

use crate::calibrate::autograd_gpu_tape::{
    concat_cols, matmul, rms_norm, slice_cols, softmax, transpose, GpuTape, GpuTensor,
};
use crate::calibrate::dynamic_quant_gpu::{estimate_sensitivities, QuantizableInput};

/// Configuration for a single-head attention block fixture.
///
/// All dimensions must be ≥ 32 (mlx-native dense f32 matmul kernel
/// requires k ≥ 32; matmul backward additionally requires m ≥ 32).
#[derive(Debug, Clone, Copy)]
pub struct AttentionBlockConfig {
    /// Sequence length / batch dim.  Must be ≥ 32.
    pub batch: usize,
    /// Hidden dimension (input + output of the block).  Must be ≥ 32.
    pub hidden: usize,
    /// Per-head dimension (single-head).  Must be ≥ 32.
    pub head_dim: usize,
    /// RMSNorm epsilon (typically 1e-6 for Qwen 3.5).
    pub eps: f32,
}

impl AttentionBlockConfig {
    /// Smallest valid config — all dims at the kernel floor (32).
    pub fn smallest() -> Self {
        Self {
            batch: 32,
            hidden: 32,
            head_dim: 32,
            eps: 1e-6,
        }
    }

    /// Validate dimension floors loud rather than letting the matmul
    /// kernel return a cryptic error mid-graph.
    pub fn validate(&self) -> Result<()> {
        if self.batch < 32 || self.hidden < 32 || self.head_dim < 32 {
            return Err(anyhow!(
                "AttentionBlockConfig: all dims must be ≥ 32; got batch={} hidden={} head_dim={}",
                self.batch,
                self.hidden,
                self.head_dim
            ));
        }
        if !self.eps.is_finite() || self.eps < 0.0 {
            return Err(anyhow!(
                "AttentionBlockConfig: eps must be finite + non-negative; got {}",
                self.eps
            ));
        }
        Ok(())
    }
}

/// Owned weights for a single attention block (single-head).  Each
/// matrix is row-major fp32.
///
/// The 1/√d_head attention scale is folded into `w_q` at construction
/// (callers pass the unscaled W_q to [`AttentionBlockWeights::new`];
/// the constructor produces the scaled buffer used in the forward
/// pass — matches the standard "scale factor folded into Q" trick
/// used in production attention implementations).
#[derive(Debug, Clone)]
pub struct AttentionBlockWeights {
    /// `[hidden]` — RMSNorm weight (frozen, not quantized).
    pub w_n: Vec<f32>,
    /// `[hidden, head_dim]` — Q projection, scaled by 1/√d_head.
    pub w_q: Vec<f32>,
    /// `[hidden, head_dim]` — K projection (unscaled).
    pub w_k: Vec<f32>,
    /// `[hidden, head_dim]` — V projection (unscaled).
    pub w_v: Vec<f32>,
    /// `[head_dim, hidden]` — O (output) projection.
    pub w_o: Vec<f32>,
}

impl AttentionBlockWeights {
    /// Construct from caller-supplied unscaled weights.  Folds the
    /// 1/√d_head scale into `w_q`.
    pub fn new(
        cfg: &AttentionBlockConfig,
        w_n: Vec<f32>,
        w_q_unscaled: Vec<f32>,
        w_k: Vec<f32>,
        w_v: Vec<f32>,
        w_o: Vec<f32>,
    ) -> Result<Self> {
        cfg.validate()?;
        if w_n.len() != cfg.hidden {
            return Err(anyhow!(
                "w_n length {} != hidden {}",
                w_n.len(),
                cfg.hidden
            ));
        }
        if w_q_unscaled.len() != cfg.hidden * cfg.head_dim {
            return Err(anyhow!(
                "w_q length {} != hidden*head_dim {}",
                w_q_unscaled.len(),
                cfg.hidden * cfg.head_dim
            ));
        }
        if w_k.len() != cfg.hidden * cfg.head_dim {
            return Err(anyhow!(
                "w_k length {} != hidden*head_dim {}",
                w_k.len(),
                cfg.hidden * cfg.head_dim
            ));
        }
        if w_v.len() != cfg.hidden * cfg.head_dim {
            return Err(anyhow!(
                "w_v length {} != hidden*head_dim {}",
                w_v.len(),
                cfg.hidden * cfg.head_dim
            ));
        }
        if w_o.len() != cfg.head_dim * cfg.hidden {
            return Err(anyhow!(
                "w_o length {} != head_dim*hidden {}",
                w_o.len(),
                cfg.head_dim * cfg.hidden
            ));
        }
        // Fold attention scale into W_q.
        let scale = 1.0_f32 / (cfg.head_dim as f32).sqrt();
        let w_q = w_q_unscaled.iter().map(|v| v * scale).collect::<Vec<f32>>();
        Ok(Self {
            w_n,
            w_q,
            w_k,
            w_v,
            w_o,
        })
    }
}

/// Tape leaves for the four quantizable projection weights.  Each
/// holds the qdq'd `W_low` GPU buffer that participates in forward
/// + backward.
#[derive(Clone)]
pub struct AttentionBlockLeaves {
    pub w_n: GpuTensor,
    pub w_q: GpuTensor,
    pub w_k: GpuTensor,
    pub w_v: GpuTensor,
    pub w_o: GpuTensor,
}

impl AttentionBlockLeaves {
    /// Place the weights on the tape as fp32 leaves verbatim (no qdq).
    /// Used as the "teacher" forward pass — consumes the full-precision
    /// reference weights.
    pub fn from_weights(
        tape: &GpuTape,
        cfg: &AttentionBlockConfig,
        w: &AttentionBlockWeights,
    ) -> Result<Self> {
        Ok(Self {
            w_n: GpuTensor::from_vec(tape, &w.w_n, vec![cfg.hidden])?,
            w_q: GpuTensor::from_vec(tape, &w.w_q, vec![cfg.hidden, cfg.head_dim])?,
            w_k: GpuTensor::from_vec(tape, &w.w_k, vec![cfg.hidden, cfg.head_dim])?,
            w_v: GpuTensor::from_vec(tape, &w.w_v, vec![cfg.hidden, cfg.head_dim])?,
            w_o: GpuTensor::from_vec(tape, &w.w_o, vec![cfg.head_dim, cfg.hidden])?,
        })
    }

    /// Place the weights on the tape with the four quantizable
    /// projections (W_q, W_k, W_v, W_o) qdq'd through `qdq_fn`.
    /// `w_n` lands verbatim (RMSNorm weight is frozen, not quantized).
    ///
    /// `qdq_fn` is typically [`crate::calibrate::qdq_gpu::qdq_q4_0_gpu`]
    /// (low-bit) for the student / W_low pass.
    pub fn from_weights_qdq<F>(
        tape: &GpuTape,
        cfg: &AttentionBlockConfig,
        w: &AttentionBlockWeights,
        qdq_fn: F,
    ) -> Result<Self>
    where
        F: Fn(&[f32]) -> Result<Vec<f32>>,
    {
        let w_q_qdq = qdq_fn(&w.w_q)?;
        let w_k_qdq = qdq_fn(&w.w_k)?;
        let w_v_qdq = qdq_fn(&w.w_v)?;
        let w_o_qdq = qdq_fn(&w.w_o)?;
        Ok(Self {
            w_n: GpuTensor::from_vec(tape, &w.w_n, vec![cfg.hidden])?,
            w_q: GpuTensor::from_vec(tape, &w_q_qdq, vec![cfg.hidden, cfg.head_dim])?,
            w_k: GpuTensor::from_vec(tape, &w_k_qdq, vec![cfg.hidden, cfg.head_dim])?,
            w_v: GpuTensor::from_vec(tape, &w_v_qdq, vec![cfg.hidden, cfg.head_dim])?,
            w_o: GpuTensor::from_vec(tape, &w_o_qdq, vec![cfg.head_dim, cfg.hidden])?,
        })
    }
}

/// Forward the attention block on the GpuTape; returns the block
/// output as a `[batch, hidden]` GpuTensor.  Does NOT add the residual
/// `x + out` — callers compose that explicitly if they need it,
/// keeping this function focused on the attention compute itself.
pub fn forward(
    cfg: &AttentionBlockConfig,
    x: &GpuTensor,
    leaves: &AttentionBlockLeaves,
) -> Result<GpuTensor> {
    cfg.validate()?;
    if x.shape() != [cfg.batch, cfg.hidden] {
        return Err(anyhow!(
            "forward: x shape {:?} != [batch={}, hidden={}]",
            x.shape(),
            cfg.batch,
            cfg.hidden
        ));
    }

    let normed = rms_norm(x, &leaves.w_n, cfg.eps)?;
    let q = matmul(&normed, &leaves.w_q)?; // [batch, head_dim]
    let k = matmul(&normed, &leaves.w_k)?; // [batch, head_dim]
    let v = matmul(&normed, &leaves.w_v)?; // [batch, head_dim]
    let k_t = transpose(&k)?; // [head_dim, batch]
    let scores = matmul(&q, &k_t)?; // [batch, batch]
    let attn = softmax(&scores)?; // [batch, batch]
    let context = matmul(&attn, &v)?; // [batch, head_dim]
    let out = matmul(&context, &leaves.w_o)?; // [batch, hidden]
    Ok(out)
}

/// Run estimate_sensitivities on the four quantizable weights of an
/// attention block, given a student forward output (built with qdq'd
/// W_low leaves) and a teacher forward output (built with the
/// reference-precision W_high weights).
///
/// Returns a map keyed by `"W_q" / "W_k" / "W_v" / "W_o"` with the
/// per-tensor sensitivity scalar (gradient-aligned alignment between
/// the qdq error vector `W_low - W_high` and the loss-gradient at
/// W_low; matches mlx-lm `dynamic_quant.py:88-94`).
pub fn estimate_attention_block_sensitivities(
    tape: &GpuTape,
    student_logits: &GpuTensor,
    teacher_logits: &GpuTensor,
    student_leaves: &AttentionBlockLeaves,
    w_high: &AttentionBlockWeights,
) -> Result<std::collections::BTreeMap<String, f64>> {
    let quantizables = vec![
        QuantizableInput {
            path: "W_q".to_string(),
            w_low: student_leaves.w_q.clone(),
            w_high_values: w_high.w_q.clone(),
        },
        QuantizableInput {
            path: "W_k".to_string(),
            w_low: student_leaves.w_k.clone(),
            w_high_values: w_high.w_k.clone(),
        },
        QuantizableInput {
            path: "W_v".to_string(),
            w_low: student_leaves.w_v.clone(),
            w_high_values: w_high.w_v.clone(),
        },
        QuantizableInput {
            path: "W_o".to_string(),
            w_low: student_leaves.w_o.clone(),
            w_high_values: w_high.w_o.clone(),
        },
    ];
    estimate_sensitivities(tape, student_logits, teacher_logits, &quantizables)
}

/// Multi-head scaled-dot-product attention on the GpuTape.
///
/// Inputs: `q`, `k`, `v` all of shape `[batch, n_heads * head_dim]`.
/// Output: `[batch, n_heads * head_dim]` — the per-head contexts
/// concatenated along the column dim.
///
/// Per-head pipeline (matches the iter-10c single-head composition):
/// ```text
///   q_h = slice_cols(Q, h*head_dim, head_dim)    [batch, head_dim]
///   k_h = slice_cols(K, h*head_dim, head_dim)
///   v_h = slice_cols(V, h*head_dim, head_dim)
///   k_h_t = transpose(k_h)                        [head_dim, batch]
///   scores_h = matmul(q_h, k_h_t)                 [batch, batch]
///   attn_h = softmax(scores_h)
///   context_h = matmul(attn_h, v_h)               [batch, head_dim]
/// ```
///
/// All `context_h` are then chained left-fold via `concat_cols` into
/// the output `[batch, n_heads * head_dim]`.
///
/// The 1/√head_dim attention scale is assumed folded into the Q
/// projection's weight upstream (matches `AttentionBlockWeights::new`).
///
/// Requires `head_dim ≥ 32` (mlx-native f32 dense matmul kernel
/// constraint) and `n_heads ≥ 1`.  At `n_heads = 1`, the function
/// is structurally equivalent to the single-head SDPA in `forward`
/// (one slice = identity, no concat).
pub fn multi_head_sdpa(
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    n_heads: usize,
    head_dim: usize,
) -> Result<GpuTensor> {
    if n_heads == 0 {
        return Err(anyhow!("multi_head_sdpa: n_heads must be > 0"));
    }
    if head_dim < 32 {
        return Err(anyhow!(
            "multi_head_sdpa: head_dim={head_dim} but mlx-native f32 matmul requires head_dim ≥ 32"
        ));
    }
    let hidden = n_heads * head_dim;
    for (label, t) in [("q", q), ("k", k), ("v", v)] {
        if t.shape().len() != 2 {
            return Err(anyhow!(
                "multi_head_sdpa: {label} must be 2-D [batch, hidden]; got shape={:?}",
                t.shape()
            ));
        }
        if t.shape()[1] != hidden {
            return Err(anyhow!(
                "multi_head_sdpa: {label}.cols={} != n_heads*head_dim = {n_heads}*{head_dim} = {hidden}",
                t.shape()[1]
            ));
        }
    }
    if q.shape()[0] != k.shape()[0] || q.shape()[0] != v.shape()[0] {
        return Err(anyhow!(
            "multi_head_sdpa: row mismatch q={} k={} v={}",
            q.shape()[0],
            k.shape()[0],
            v.shape()[0]
        ));
    }
    let batch = q.shape()[0];
    if batch < 32 {
        // matmul backward requires m ≥ 32; and Q@K^T's first dim is `batch`.
        return Err(anyhow!(
            "multi_head_sdpa: batch={batch} but matmul backward requires batch ≥ 32"
        ));
    }

    let mut contexts: Vec<GpuTensor> = Vec::with_capacity(n_heads);
    for h in 0..n_heads {
        let start = h * head_dim;
        let q_h = slice_cols(q, start, head_dim)?;
        let k_h = slice_cols(k, start, head_dim)?;
        let v_h = slice_cols(v, start, head_dim)?;
        let k_h_t = transpose(&k_h)?;
        let scores_h = matmul(&q_h, &k_h_t)?;
        let attn_h = softmax(&scores_h)?;
        let context_h = matmul(&attn_h, &v_h)?;
        contexts.push(context_h);
    }
    // Left-fold concat: concat(concat(concat(c0, c1), c2), c3) ...
    let mut acc = contexts[0].clone();
    for c in contexts.iter().skip(1) {
        acc = concat_cols(&acc, c)?;
    }
    Ok(acc)
}

/// Streaming sensitivity estimator (ADR-020 iter-10d).
///
/// Processes a sequence of independent batches one at a time, accumulating
/// per-batch sensitivity scalars into a running mean.  Each batch builds
/// a FRESH `GpuTape` + does its own forward+backward; the tape is dropped
/// at the end of the iteration so per-batch forward intermediates +
/// backward grads are reclaimed before the next batch starts.  Memory
/// stays bounded to a single batch's tape footprint regardless of the
/// total batch count.
///
/// This mirrors mlx-lm `dynamic_quant.py:75-86`'s streaming rhythm:
///
/// ```python
/// for batch in batches:
///     targets = teacher(batch)           # teacher forward
///     _, grads = nn.value_and_grad(student, loss_fn)(batch, targets)
///     grad_accum = tree_map(lambda x, y: x + y, grad_accum, grads)
///     del grads                          # drop per-batch grads
///     mx.eval(grad_accum)                # force GPU sync
/// ```
///
/// Mathematically equivalent to running `estimate_sensitivities` on
/// each batch separately and averaging the scalar results — proven by
/// the identity:
///
/// ```text
///   mean_b s_b = Σ_b (grad_b · (W_low − W_high)).sum() / (numel · n_batches)
///              = (Σ_b grad_b · (W_low − W_high)).sum() / (numel · n_batches)
///              = (grad_accum / n_batches · (W_low − W_high)).sum() / numel
/// ```
///
/// Both formulations yield the same scalar within float-sum-order
/// tolerance, so we use the per-batch-mean formulation here (smaller
/// memory footprint: stores 4 × f64 scalars instead of 4 × full-weight
/// gradient buffers).
///
/// `batch_inputs` is an iterator over the per-batch input `[batch, hidden]`
/// row-major flat fp32 buffers.  Each batch must match `cfg.batch *
/// cfg.hidden` in length.
///
/// Returns the per-quantizable mean sensitivity scalar.
pub fn estimate_attention_block_sensitivities_streaming<I, B>(
    cfg: &AttentionBlockConfig,
    weights: &AttentionBlockWeights,
    qdq_fn: impl Fn(&[f32]) -> Result<Vec<f32>>,
    batch_inputs: I,
) -> Result<std::collections::BTreeMap<String, f64>>
where
    I: IntoIterator<Item = B>,
    B: AsRef<[f32]>,
{
    cfg.validate()?;
    let mut scalar_accum: std::collections::BTreeMap<String, f64> =
        std::collections::BTreeMap::new();
    let mut n_batches: usize = 0;

    // Single shared device + tape across all batches.  The tape is
    // RESET (nodes cleared, MlxBuffer Arcs dropped) between batches
    // — that's the memory-bound mechanism.  Reusing one device
    // avoids the Metal residency-set contention that would otherwise
    // flake the streaming-vs-bulk parity test (each `MlxDevice::new`
    // re-registers a residency set with the system; rapid churn on
    // macOS produces intermittent corrupted state).
    let device = mlx_native::MlxDevice::new()
        .map_err(|e| anyhow!("streaming: device: {e}"))?;
    let tape = GpuTape::new(device);

    for batch_buf in batch_inputs {
        let batch = batch_buf.as_ref();
        if batch.len() != cfg.batch * cfg.hidden {
            return Err(anyhow!(
                "streaming: batch length {} != batch*hidden = {}*{} = {}",
                batch.len(),
                cfg.batch,
                cfg.hidden,
                cfg.batch * cfg.hidden
            ));
        }

        let xt = GpuTensor::from_vec(&tape, batch, vec![cfg.batch, cfg.hidden])?;
        let teacher_leaves = AttentionBlockLeaves::from_weights(&tape, cfg, weights)?;
        let teacher_out = forward(cfg, &xt, &teacher_leaves)?;
        let student_leaves =
            AttentionBlockLeaves::from_weights_qdq(&tape, cfg, weights, &qdq_fn)?;
        let student_out = forward(cfg, &xt, &student_leaves)?;
        let per_batch = estimate_attention_block_sensitivities(
            &tape,
            &student_out,
            &teacher_out,
            &student_leaves,
            weights,
        )?;
        for (k, v) in per_batch {
            *scalar_accum.entry(k).or_insert(0.0) += v;
        }
        n_batches += 1;
        // Drop all per-batch GpuTensors (xt, teacher_leaves, student_leaves,
        // teacher_out, student_out) by resetting the tape — clears
        // its nodes vec, dropping the MlxBuffer Arcs.  Metal reclaims
        // unified-memory pages before next iteration.
        tape.reset();
    }

    if n_batches == 0 {
        return Err(anyhow!("streaming: zero batches provided"));
    }
    let n = n_batches as f64;
    for v in scalar_accum.values_mut() {
        *v /= n;
    }
    Ok(scalar_accum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::autograd_gpu_tape::{backward, ones_like};
    use crate::calibrate::qdq_gpu::{qdq_q4_0_gpu, qdq_q8_0_gpu};
    use mlx_native::MlxDevice;

    fn deterministic_weights(cfg: &AttentionBlockConfig, seed: u64) -> AttentionBlockWeights {
        // Lightweight deterministic PRNG for reproducible test data.
        let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
        let mut next = || {
            state ^= state >> 33;
            state = state.wrapping_mul(0xff51_afd7_ed55_8ccd);
            state ^= state >> 33;
            ((state as i64) as f32) / (i64::MAX as f32)
        };
        let w_n: Vec<f32> = (0..cfg.hidden).map(|_| 1.0 + next() * 0.1).collect();
        let w_q: Vec<f32> = (0..cfg.hidden * cfg.head_dim)
            .map(|_| next() * 0.5)
            .collect();
        let w_k: Vec<f32> = (0..cfg.hidden * cfg.head_dim)
            .map(|_| next() * 0.5)
            .collect();
        let w_v: Vec<f32> = (0..cfg.hidden * cfg.head_dim)
            .map(|_| next() * 0.5)
            .collect();
        let w_o: Vec<f32> = (0..cfg.head_dim * cfg.hidden)
            .map(|_| next() * 0.5)
            .collect();
        AttentionBlockWeights::new(cfg, w_n, w_q, w_k, w_v, w_o).unwrap()
    }

    #[test]
    fn attention_block_forward_shape_correct() {
        let cfg = AttentionBlockConfig::smallest();
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let x: Vec<f32> = (0..cfg.batch * cfg.hidden)
            .map(|i| (i as f32 * 0.013).sin() * 0.4)
            .collect();
        let xt = GpuTensor::from_vec(&tape, &x, vec![cfg.batch, cfg.hidden]).unwrap();
        let weights = deterministic_weights(&cfg, 7919);
        let leaves = AttentionBlockLeaves::from_weights(&tape, &cfg, &weights).unwrap();
        let out = forward(&cfg, &xt, &leaves).unwrap();
        assert_eq!(out.shape(), [cfg.batch, cfg.hidden]);
        // Output values must be finite — the softmax + matmul chain
        // can produce NaN/Inf if shape contracts are violated.
        let out_vec: Vec<f32> = out.to_vec().unwrap();
        for (i, v) in out_vec.iter().enumerate() {
            assert!(v.is_finite(), "out[{i}] = {v} not finite");
        }
    }

    #[test]
    fn attention_block_backward_flows_to_all_four_weight_leaves() {
        // Forward + backward through a real attention block; verify
        // gradients land on EACH of the 4 quantizable weight leaves
        // (W_q, W_k, W_v, W_o) and on the input + RMSNorm weight.
        // The grad shapes must match the leaf shapes exactly.
        let cfg = AttentionBlockConfig::smallest();
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let x: Vec<f32> = (0..cfg.batch * cfg.hidden)
            .map(|i| (i as f32 * 0.011).cos() * 0.3)
            .collect();
        let xt = GpuTensor::from_vec(&tape, &x, vec![cfg.batch, cfg.hidden]).unwrap();
        let weights = deterministic_weights(&cfg, 12347);
        let leaves = AttentionBlockLeaves::from_weights(&tape, &cfg, &weights).unwrap();

        let out = forward(&cfg, &xt, &leaves).unwrap();
        let dy = ones_like(&tape, &[cfg.batch, cfg.hidden]).unwrap();
        let grads = backward(&out, dy).unwrap();

        // Every leaf MUST have a gradient.
        for (label, leaf) in &[
            ("x", &xt),
            ("w_n", &leaves.w_n),
            ("w_q", &leaves.w_q),
            ("w_k", &leaves.w_k),
            ("w_v", &leaves.w_v),
            ("w_o", &leaves.w_o),
        ] {
            let g = grads[leaf.node_idx()]
                .as_ref()
                .unwrap_or_else(|| panic!("{label} grad missing"));
            // Shape match (element count is the load-bearing check).
            let leaf_numel: usize = leaf.shape().iter().product();
            assert_eq!(
                g.element_count(),
                leaf_numel,
                "{label}: grad numel {} != leaf numel {leaf_numel}",
                g.element_count()
            );
            // Grads must be finite — softmax + matmul chain stability
            // implicitly verified.
            let g_vec: Vec<f32> = g.as_slice::<f32>().unwrap().to_vec();
            for (i, v) in g_vec.iter().enumerate() {
                let val: f32 = *v;
                assert!(val.is_finite(), "{label} grad[{i}] = {val} not finite");
            }
        }
    }

    #[test]
    fn attention_block_w_low_q4_0_w_high_q8_0_diff_nonzero() {
        // Sanity for sensitivity scoring: when the same source weights
        // are qdq'd via Q4_0 (W_low) vs Q8_0 (W_high), every
        // projection's W_low - W_high diff must be non-zero on real
        // weight magnitudes.
        let cfg = AttentionBlockConfig::smallest();
        let weights = deterministic_weights(&cfg, 31337);
        let device = MlxDevice::new().expect("device sanity");
        let max_diff = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f32, f32::max)
        };
        for (label, ws) in &[
            ("w_q", &weights.w_q),
            ("w_k", &weights.w_k),
            ("w_v", &weights.w_v),
            ("w_o", &weights.w_o),
        ] {
            let low = qdq_q4_0_gpu(&device, ws).unwrap();
            let high = qdq_q8_0_gpu(&device, ws).unwrap();
            let d = max_diff(&low, &high);
            assert!(
                d > 1e-4,
                "{label}: qdq Q4_0 vs Q8_0 max-abs-diff {d} too small — sensitivity formula will be degenerate"
            );
        }
    }

    #[test]
    fn attention_block_streaming_vs_per_batch_mean_byte_close() {
        // Streaming pattern test: process N batches one at a time
        // (each on a fresh tape that's dropped after the iteration);
        // compare against running estimate_sensitivities on each
        // batch separately and averaging the scalars by hand.
        //
        // Per the identity in `estimate_attention_block_sensitivities_streaming`,
        // both formulations are mathematically equivalent — any
        // discrepancy is float-sum-order at the f64 accumulation step.
        // Tight rel_tol = 1e-12 (f64 precision).
        let cfg = AttentionBlockConfig::smallest();
        let weights = deterministic_weights(&cfg, 555);

        // Three independent batches with different input distributions.
        let n_batches = 3;
        let mut batches: Vec<Vec<f32>> = Vec::with_capacity(n_batches);
        for b in 0..n_batches {
            let batch: Vec<f32> = (0..cfg.batch * cfg.hidden)
                .map(|i| {
                    let v = (i as f32 * 0.0091 + b as f32 * 0.7).sin();
                    v * (0.3 + 0.1 * b as f32)
                })
                .collect();
            batches.push(batch);
        }

        // Run streaming (fresh tape per batch, drops after each).
        let device_for_qdq = MlxDevice::new().expect("device for qdq");
        let stream_result = estimate_attention_block_sensitivities_streaming(
            &cfg,
            &weights,
            |w| qdq_q4_0_gpu(&device_for_qdq, w),
            batches.iter().map(|b| b.as_slice()),
        )
        .unwrap();

        // Manual per-batch + average reference — uses the SAME
        // device + tape-reset pattern as streaming for apples-to-apples
        // comparison (avoids per-batch MlxDevice::new flakes).
        let mut manual_accum: std::collections::BTreeMap<String, f64> =
            std::collections::BTreeMap::new();
        let manual_device = MlxDevice::new().expect("manual device");
        let manual_tape = GpuTape::new(manual_device);
        for batch in &batches {
            let xt = GpuTensor::from_vec(&manual_tape, batch, vec![cfg.batch, cfg.hidden])
                .unwrap();
            let teacher_leaves =
                AttentionBlockLeaves::from_weights(&manual_tape, &cfg, &weights).unwrap();
            let teacher_out = forward(&cfg, &xt, &teacher_leaves).unwrap();
            let student_leaves = AttentionBlockLeaves::from_weights_qdq(
                &manual_tape,
                &cfg,
                &weights,
                |w| qdq_q4_0_gpu(manual_tape.device(), w),
            )
            .unwrap();
            let student_out = forward(&cfg, &xt, &student_leaves).unwrap();
            let per_batch = estimate_attention_block_sensitivities(
                &manual_tape,
                &student_out,
                &teacher_out,
                &student_leaves,
                &weights,
            )
            .unwrap();
            for (k, v) in per_batch {
                *manual_accum.entry(k).or_insert(0.0) += v;
            }
            manual_tape.reset();
        }
        for v in manual_accum.values_mut() {
            *v /= n_batches as f64;
        }

        // Compare keys + scalars.
        assert_eq!(
            stream_result.keys().collect::<Vec<_>>(),
            manual_accum.keys().collect::<Vec<_>>(),
            "streaming and manual key sets differ"
        );
        for k in stream_result.keys() {
            let s = stream_result[k];
            let m = manual_accum[k];
            let diff = (s - m).abs();
            let scale = s.abs().max(m.abs()).max(1e-12);
            assert!(
                diff <= 1e-12 || diff / scale <= 1e-10,
                "{k}: streaming={s} manual={m} diff={diff}"
            );
        }
    }

    /// ADR-020 iter-11h-e3a regression sentinel: each invocation gets
    /// a FRESH tape (no reset reused).  Pre-fix this would diverge
    /// at iter ≥ ~10 due to the missing matmul memory_barrier; post-fix
    /// it is byte-identical across all iterations.  Catches the
    /// matmul-RAW-barrier bug class even if tape lifecycle is bypassed.
    #[test]
    fn matmul_raw_barrier_fresh_tape_regression_sentinel() {
        let cfg = AttentionBlockConfig::smallest();
        let weights = deterministic_weights(&cfg, 555);
        let batch: Vec<f32> = (0..cfg.batch * cfg.hidden)
            .map(|i| (i as f32 * 0.0091).sin() * 0.3)
            .collect();

        let n_repeat = 50;
        let mut wk_values: Vec<f64> = Vec::with_capacity(n_repeat);
        for r in 0..n_repeat {
            let device = MlxDevice::new().expect("device");
            let tape = GpuTape::new(device);
            let xt = GpuTensor::from_vec(&tape, &batch, vec![cfg.batch, cfg.hidden]).unwrap();
            let teacher_leaves =
                AttentionBlockLeaves::from_weights(&tape, &cfg, &weights).unwrap();
            let teacher_out = forward(&cfg, &xt, &teacher_leaves).unwrap();
            let student_leaves = AttentionBlockLeaves::from_weights_qdq(
                &tape, &cfg, &weights,
                |w| qdq_q4_0_gpu(tape.device(), w),
            ).unwrap();
            let student_out = forward(&cfg, &xt, &student_leaves).unwrap();
            let per_batch = estimate_attention_block_sensitivities(
                &tape, &student_out, &teacher_out, &student_leaves, &weights,
            ).unwrap();
            let wk = per_batch["W_k"];
            wk_values.push(wk);
            if r > 0 && wk != wk_values[0] {
                eprintln!("[matmul-raw-barrier-sentinel] regression at r={r}: W_k={wk} != canonical={}", wk_values[0]);
            }
        }
        let canonical = wk_values[0];
        for (i, v) in wk_values.iter().enumerate() {
            assert_eq!(v.to_bits(), canonical.to_bits(),
                "iter {i}: W_k={v} != canonical={canonical}");
        }
    }

    /// ADR-020 iter-11h-e3a regression sentinel — heavy variant.  Runs
    /// `estimate_attention_block_sensitivities` 200× on a single shared
    /// tape (with reset() between iterations) and asserts ALL 200 W_k
    /// scores are byte-identical.  Pre-fix this would diverge at iter ≥
    /// ~30 with one of a small set of repeated wrong values
    /// (-16550.30, -20572.34, +21826.42, +39993.52, 0) due to the
    /// matmul transpose→matmul RAW dependency racing without
    /// `encoder.memory_barrier()`.
    #[test]
    fn matmul_raw_barrier_repetition_regression_sentinel() {
        let cfg = AttentionBlockConfig::smallest();
        let weights = deterministic_weights(&cfg, 555);
        let batch: Vec<f32> = (0..cfg.batch * cfg.hidden)
            .map(|i| (i as f32 * 0.0091).sin() * 0.3)
            .collect();

        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);

        // 200-iter sentinel: pre-fix this would diverge ~10-15% of the
        // time at iter ≥ ~30; post-fix it must be byte-identical across
        // all 200 iters.  Lower than the 1000-iter probe used during
        // root-cause investigation but enough to catch any RAW-barrier
        // regression.
        let n_repeat = 200;
        let mut wk_values: Vec<f64> = Vec::with_capacity(n_repeat);

        for r in 0..n_repeat {
            let xt = GpuTensor::from_vec(&tape, &batch, vec![cfg.batch, cfg.hidden]).unwrap();
            let teacher_leaves =
                AttentionBlockLeaves::from_weights(&tape, &cfg, &weights).unwrap();
            let teacher_out = forward(&cfg, &xt, &teacher_leaves).unwrap();
            let student_leaves = AttentionBlockLeaves::from_weights_qdq(
                &tape, &cfg, &weights,
                |w| qdq_q4_0_gpu(tape.device(), w),
            ).unwrap();
            let student_out = forward(&cfg, &xt, &student_leaves).unwrap();
            let per_batch = estimate_attention_block_sensitivities(
                &tape, &student_out, &teacher_out, &student_leaves, &weights,
            ).unwrap();
            let wk = per_batch["W_k"];
            wk_values.push(wk);
            if r > 0 && wk != wk_values[0] {
                eprintln!("[matmul-raw-barrier-probe] regression at r={r}: W_k={wk} != canonical={}", wk_values[0]);
            }
            tape.reset();
        }

        let canonical = wk_values[0];
        for (i, v) in wk_values.iter().enumerate() {
            assert_eq!(
                v.to_bits(), canonical.to_bits(),
                "iter {i}: W_k={v} != canonical={canonical} \
                 — likely missing memory_barrier between transpose+matmul \
                 (see autograd_gpu_tape.rs forward+backward matmul)"
            );
        }
    }

    /// CPU oracle: multi-head SDPA without the √d scale.
    /// Mirrors `multi_head_sdpa` exactly — same per-head loop +
    /// concat ordering, computed on the host in fp32.
    fn multi_head_sdpa_cpu_oracle(
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
            // scores[b, b'] = Σ_d q[b, start+d] * k[b', start+d]
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
            // softmax per row.
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
            // context[b, d] = Σ_b' attn[b, b'] * v[b', start+d]
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

    fn assert_close_inline(label: &str, gpu: &[f32], cpu: &[f32], rel_tol: f32, abs_tol: f32) {
        assert_eq!(gpu.len(), cpu.len(), "{label}: length mismatch");
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let diff = (g - c).abs();
            let scale = g.abs().max(c.abs()).max(1.0);
            assert!(
                diff <= abs_tol || diff / scale <= rel_tol,
                "{label}: i={i}: gpu={g} cpu={c} diff={diff}"
            );
        }
    }

    #[test]
    fn multi_head_sdpa_n_heads_2_parity_with_cpu_oracle() {
        // n_heads=2 × head_dim=32 → hidden=64; batch=32.
        let batch = 32usize;
        let n_heads = 2usize;
        let head_dim = 32usize;
        let hidden = n_heads * head_dim;
        let q: Vec<f32> = (0..batch * hidden)
            .map(|i| (i as f32 * 0.0091).sin() * 0.4)
            .collect();
        let k: Vec<f32> = (0..batch * hidden)
            .map(|i| ((i as f32) * 0.0073 + 1.5).cos() * 0.3)
            .collect();
        let v: Vec<f32> = (0..batch * hidden)
            .map(|i| ((i as f32) * 0.011 + 2.7).sin() * 0.5)
            .collect();
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let qt = GpuTensor::from_vec(&tape, &q, vec![batch, hidden]).unwrap();
        let kt = GpuTensor::from_vec(&tape, &k, vec![batch, hidden]).unwrap();
        let vt = GpuTensor::from_vec(&tape, &v, vec![batch, hidden]).unwrap();
        let context = multi_head_sdpa(&qt, &kt, &vt, n_heads, head_dim).unwrap();
        let gpu: Vec<f32> = context.to_vec().unwrap();
        let cpu = multi_head_sdpa_cpu_oracle(&q, &k, &v, batch, n_heads, head_dim);
        assert_close_inline("multi-head sdpa n_heads=2", &gpu, &cpu, 1e-4, 1e-5);
    }

    #[test]
    fn multi_head_sdpa_n_heads_1_matches_cpu_oracle() {
        // n_heads=1 special-case: the function takes the single-slice
        // path (slice covers full hidden) + no concat fold (only one
        // context).  Validates the n_heads=1 base case against the
        // CPU oracle.
        let batch = 32usize;
        let head_dim = 32usize;
        let q: Vec<f32> = (0..batch * head_dim)
            .map(|i| (i as f32 * 0.013).sin() * 0.4)
            .collect();
        let k: Vec<f32> = (0..batch * head_dim)
            .map(|i| ((i as f32) * 0.011 + 1.0).cos() * 0.3)
            .collect();
        let v: Vec<f32> = (0..batch * head_dim)
            .map(|i| ((i as f32) * 0.017 + 2.0).sin() * 0.5)
            .collect();
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let qt = GpuTensor::from_vec(&tape, &q, vec![batch, head_dim]).unwrap();
        let kt = GpuTensor::from_vec(&tape, &k, vec![batch, head_dim]).unwrap();
        let vt = GpuTensor::from_vec(&tape, &v, vec![batch, head_dim]).unwrap();
        let context = multi_head_sdpa(&qt, &kt, &vt, 1, head_dim).unwrap();
        let gpu: Vec<f32> = context.to_vec().unwrap();
        let cpu = multi_head_sdpa_cpu_oracle(&q, &k, &v, batch, 1, head_dim);
        assert_close_inline("multi-head sdpa n_heads=1", &gpu, &cpu, 1e-4, 1e-5);
    }

    #[test]
    fn multi_head_sdpa_backward_flows_to_qkv() {
        // Backward through multi-head SDPA must produce gradients
        // on q, k, v leaves with correct shapes + finite values.
        use crate::calibrate::autograd_gpu_tape::{backward, ones_like};
        let batch = 32usize;
        let n_heads = 4usize;
        let head_dim = 32usize;
        let hidden = n_heads * head_dim;
        let q: Vec<f32> = (0..batch * hidden).map(|i| (i as f32 * 0.0091).sin() * 0.4).collect();
        let k: Vec<f32> = (0..batch * hidden).map(|i| (i as f32 * 0.0073).cos() * 0.3).collect();
        let v: Vec<f32> = (0..batch * hidden).map(|i| (i as f32 * 0.011).sin() * 0.5).collect();
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let qt = GpuTensor::from_vec(&tape, &q, vec![batch, hidden]).unwrap();
        let kt = GpuTensor::from_vec(&tape, &k, vec![batch, hidden]).unwrap();
        let vt = GpuTensor::from_vec(&tape, &v, vec![batch, hidden]).unwrap();
        let ctx = multi_head_sdpa(&qt, &kt, &vt, n_heads, head_dim).unwrap();
        let dy = ones_like(&tape, &[batch, hidden]).unwrap();
        let grads = backward(&ctx, dy).unwrap();
        for (label, leaf) in &[("q", &qt), ("k", &kt), ("v", &vt)] {
            let g = grads[leaf.node_idx()]
                .as_ref()
                .unwrap_or_else(|| panic!("{label} grad missing"));
            assert_eq!(g.element_count(), batch * hidden);
            let g_vec: Vec<f32> = g.as_slice::<f32>().unwrap().to_vec();
            for (i, v) in g_vec.iter().enumerate() {
                let val: f32 = *v;
                assert!(val.is_finite(), "{label} grad[{i}] = {val} not finite");
            }
        }
    }

    #[test]
    fn multi_head_sdpa_rejects_invalid_inputs() {
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let q = GpuTensor::from_vec(&tape, &vec![0.5; 32 * 64], vec![32, 64]).unwrap();
        let k = q.clone();
        let v = q.clone();
        // n_heads=0
        match multi_head_sdpa(&q, &k, &v, 0, 32) {
            Err(e) => assert!(format!("{e}").contains("n_heads must be > 0")),
            Ok(_) => panic!("expected n_heads=0 error"),
        }
        // head_dim < 32
        match multi_head_sdpa(&q, &k, &v, 4, 16) {
            Err(e) => assert!(format!("{e}").contains("head_dim ≥ 32")),
            Ok(_) => panic!("expected head_dim<32 error"),
        }
        // mismatched hidden vs n_heads * head_dim
        match multi_head_sdpa(&q, &k, &v, 3, 32) {
            // 3 * 32 = 96 ≠ 64
            Err(e) => assert!(format!("{e}").contains("n_heads*head_dim")),
            Ok(_) => panic!("expected hidden mismatch error"),
        }
    }

    #[test]
    fn attention_block_streaming_zero_batches_errors() {
        let cfg = AttentionBlockConfig::smallest();
        let weights = deterministic_weights(&cfg, 1);
        let device = MlxDevice::new().expect("device");
        let empty: Vec<&[f32]> = Vec::new();
        let err = estimate_attention_block_sensitivities_streaming(
            &cfg,
            &weights,
            |w| qdq_q4_0_gpu(&device, w),
            empty,
        )
        .expect_err("zero batches must error");
        assert!(
            format!("{err}").contains("zero batches"),
            "wrong error: {err}"
        );
    }

    #[test]
    fn attention_block_estimate_sensitivities_runs_and_produces_finite_scores() {
        // End-to-end iter-10c integration test:
        //   1. Build full-precision teacher forward
        //   2. Build qdq-Q4_0 student forward (W_low)
        //   3. Run estimate_sensitivities — expect 4 scalar scores
        //      (one per quantizable projection) all finite + non-zero.
        //   4. Q4_0 (coarser) student should produce LARGER sensitivity
        //      than a Q8_0 (finer) student — pin this monotonicity.
        let cfg = AttentionBlockConfig::smallest();
        let weights = deterministic_weights(&cfg, 4242);

        // Single shared device + tape for both Q4_0 and Q8_0 sub-passes.
        // tape.reset() between passes drops per-sub-pass node Arcs;
        // device persists to avoid Metal residency-set contention
        // under parallel test load (same fix pattern as iter-11b
        // streaming test).
        let device = MlxDevice::new().expect("device shared");
        let tape_t = GpuTape::new(device);
        let x_vec: Vec<f32> = (0..cfg.batch * cfg.hidden)
            .map(|i| (i as f32 * 0.0091).sin() * 0.35)
            .collect();

        // Sub-pass 1: Q4_0 student.
        let x_t = GpuTensor::from_vec(&tape_t, &x_vec, vec![cfg.batch, cfg.hidden]).unwrap();
        let teacher_leaves = AttentionBlockLeaves::from_weights(&tape_t, &cfg, &weights).unwrap();
        let teacher_out = forward(&cfg, &x_t, &teacher_leaves).unwrap();
        let student4_leaves = AttentionBlockLeaves::from_weights_qdq(
            &tape_t,
            &cfg,
            &weights,
            |w| qdq_q4_0_gpu(tape_t.device(), w),
        )
        .unwrap();
        let student4_out = forward(&cfg, &x_t, &student4_leaves).unwrap();
        let sens_q4 = estimate_attention_block_sensitivities(
            &tape_t,
            &student4_out,
            &teacher_out,
            &student4_leaves,
            &weights,
        )
        .unwrap();
        // Drop sub-pass 1 nodes (release MlxBuffer Arcs) before
        // sub-pass 2 builds its forward graph.  Local GpuTensors
        // (x_t, teacher_leaves, student4_leaves, teacher_out,
        // student4_out) drop at end of this block.
        tape_t.reset();

        for k in &["W_q", "W_k", "W_v", "W_o"] {
            let s = sens_q4
                .get(*k)
                .copied()
                .unwrap_or_else(|| panic!("{k} sensitivity missing"));
            assert!(s.is_finite(), "{k} sensitivity {s} not finite");
            // Sensitivity is signed (gradient-alignment with the qdq
            // error vector); we require non-zero magnitude — degenerate
            // 0.0 indicates broken composition.
            assert!(
                s.abs() > 1e-12,
                "{k} sensitivity {s} too small — autograd chain may be degenerate"
            );
        }

        // Sub-pass 2: Q8_0 student on the same tape (post-reset).
        let x_t2 = GpuTensor::from_vec(&tape_t, &x_vec, vec![cfg.batch, cfg.hidden]).unwrap();
        let teacher_leaves2 =
            AttentionBlockLeaves::from_weights(&tape_t, &cfg, &weights).unwrap();
        let teacher_out2 = forward(&cfg, &x_t2, &teacher_leaves2).unwrap();
        let student8_leaves = AttentionBlockLeaves::from_weights_qdq(
            &tape_t,
            &cfg,
            &weights,
            |w| qdq_q8_0_gpu(tape_t.device(), w),
        )
        .unwrap();
        let student8_out = forward(&cfg, &x_t2, &student8_leaves).unwrap();
        let sens_q8 = estimate_attention_block_sensitivities(
            &tape_t,
            &student8_out,
            &teacher_out2,
            &student8_leaves,
            &weights,
        )
        .unwrap();

        // Monotonicity: |sensitivity| with Q4_0 student >= |sensitivity|
        // with Q8_0 student for at least 3 of 4 projections.  (One
        // outlier is acceptable due to the gradient-alignment formula
        // being signed — a single projection can flip sign across
        // bit-widths if its qdq error vector aligns/misaligns with the
        // gradient differently.  Three of four agree on magnitude is
        // the meaningful pin.)
        let mut q4_larger = 0;
        for k in &["W_q", "W_k", "W_v", "W_o"] {
            let s4 = sens_q4[*k].abs();
            let s8 = sens_q8[*k].abs();
            // Use 0.95 slack — Q8_0 errors are ~16x smaller per-element
            // than Q4_0 (255-step vs 15-step quantization grid), so
            // |Q4_0 sens| should dominate with substantial margin.
            if s4 >= 0.95 * s8 {
                q4_larger += 1;
            }
        }
        assert!(
            q4_larger >= 3,
            "Q4_0-vs-Q8_0 sensitivity monotonicity failed: only {q4_larger}/4 projections \
             have |sens(Q4_0)| ≥ 0.95·|sens(Q8_0)|.  sens_q4 = {sens_q4:?}, sens_q8 = {sens_q8:?}"
        );
    }
}
