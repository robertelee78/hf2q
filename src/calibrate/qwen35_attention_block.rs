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
    matmul, rms_norm, softmax, transpose, GpuTape, GpuTensor,
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

        // Fresh device + tape per batch — this is the load-bearing
        // memory bound: when this scope exits the entire tape is
        // dropped (Drop on GpuTapeInner releases all MlxBuffer Arcs;
        // Metal reclaims the unified-memory pages).
        let device = mlx_native::MlxDevice::new()
            .map_err(|e| anyhow!("streaming batch: device: {e}"))?;
        let tape = GpuTape::new(device);

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
        // tape, teacher_leaves, student_leaves, teacher_out, student_out
        // all drop here — per-batch state reclaimed before next iter.
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

        // Manual per-batch + average reference.
        let mut manual_accum: std::collections::BTreeMap<String, f64> =
            std::collections::BTreeMap::new();
        for batch in &batches {
            let device = MlxDevice::new().expect("device per-batch");
            let tape = GpuTape::new(device);
            let xt = GpuTensor::from_vec(&tape, batch, vec![cfg.batch, cfg.hidden]).unwrap();
            let teacher_leaves =
                AttentionBlockLeaves::from_weights(&tape, &cfg, &weights).unwrap();
            let teacher_out = forward(&cfg, &xt, &teacher_leaves).unwrap();
            let student_leaves = AttentionBlockLeaves::from_weights_qdq(
                &tape,
                &cfg,
                &weights,
                |w| qdq_q4_0_gpu(tape.device(), w),
            )
            .unwrap();
            let student_out = forward(&cfg, &xt, &student_leaves).unwrap();
            let per_batch = estimate_attention_block_sensitivities(
                &tape,
                &student_out,
                &teacher_out,
                &student_leaves,
                &weights,
            )
            .unwrap();
            for (k, v) in per_batch {
                *manual_accum.entry(k).or_insert(0.0) += v;
            }
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

        // Sub-pass 1: Q4_0 student.  Fresh device + tape — independent
        // of sub-pass 2.
        let device_q4 = MlxDevice::new().expect("device q4");
        let tape_t = GpuTape::new(device_q4);
        let x_vec: Vec<f32> = (0..cfg.batch * cfg.hidden)
            .map(|i| (i as f32 * 0.0091).sin() * 0.35)
            .collect();
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

        // Sub-pass 2: Q8_0 student.  Fresh device + tape (no leakage).
        let device_q8 = MlxDevice::new().expect("device q8");
        let tape_t2 = GpuTape::new(device_q8);
        let x_t2 = GpuTensor::from_vec(&tape_t2, &x_vec, vec![cfg.batch, cfg.hidden]).unwrap();
        let teacher_leaves2 =
            AttentionBlockLeaves::from_weights(&tape_t2, &cfg, &weights).unwrap();
        let teacher_out2 = forward(&cfg, &x_t2, &teacher_leaves2).unwrap();
        let student8_leaves = AttentionBlockLeaves::from_weights_qdq(
            &tape_t2,
            &cfg,
            &weights,
            |w| qdq_q8_0_gpu(tape_t2.device(), w),
        )
        .unwrap();
        let student8_out = forward(&cfg, &x_t2, &student8_leaves).unwrap();
        let sens_q8 = estimate_attention_block_sensitivities(
            &tape_t2,
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
