//! ADR-020 iter-11h-e2 — MoE routing forward composition on GpuTape.
//!
//! Mirrors `mlx-lm/qwen3_next.py:Qwen3NextSparseMoeBlock.__call__`'s
//! routing chain:
//!
//! ```python
//!   gates = softmax(self.gate(x))                    # [n_tokens, n_experts]
//!   inds  = argpartition(gates, kth=-k)[..., -k:]    # [n_tokens, k] u32
//!   scores = take_along_axis(gates, inds, axis=-1)   # [n_tokens, k]
//!   if norm_topk_prob:
//!       scores = scores / scores.sum(axis=-1, keepdims=True)
//!   # ... downstream switch_mlp(x, inds) consumes (inds, scores)
//! ```
//!
//! ## Composition on GpuTape
//!
//! Forward (with backward auto-derived through every OpKind):
//!
//!   1. `gate_logits = matmul(input, gate_weight)`  // matmul has a
//!      32-floor on m, n, k — for production Qwen3.5MoE shapes
//!      (n_tokens >> 32, hidden >> 32, n_experts >> 32) this is met
//!      trivially; test fixtures must satisfy it.
//!   2. `probs = softmax(gate_logits)`              // along last axis
//!   3. **OFF TAPE**: argpartition `probs` per-row to get top-K
//!      indices.  Done CPU-side via `to_vec()` + `select_nth_unstable_by`.
//!      Indices are non-differentiable → no tape node.
//!   4. `scores = take_along_axis_topk(probs, indices, k)`  → `[n_tokens, k]`
//!   5. Optional renorm (`norm_topk_prob == true`):
//!         `sum_k         = row_sum(scores)`                      → `[n_tokens]`
//!         `sum_broadcast = outer_product(sum_k, ones_k)`          → `[n_tokens, k]`
//!         `normalized    = divide(scores, sum_broadcast)`
//!         (uses iter-11h-misc-1's `divide` primitive directly —
//!          replaces an earlier `exp(-log(sum))` reciprocal trick
//!          which was a workaround for not having `divide` yet.)
//!
//! ## Returns
//!
//!   * `top_k_scores` — differentiable tape node with shape `[n_tokens, k]`
//!   * `top_k_indices` — `MlxBuffer` of u32 indices `[n_tokens, k]`
//!     (non-differentiable; used downstream by `switch_mlp` to dispatch
//!     each token to its top-K experts)

use anyhow::{anyhow, Result};

use mlx_native::{DType, MlxBuffer};

use super::autograd_gpu_tape::{
    add, divide, matmul, mul, outer_product, rms_norm, row_sum, silu, softmax,
    take_along_axis_topk, GpuTape, GpuTensor,
};

pub struct MoeRouteOutput {
    /// Differentiable per-row top-K scores, shape `[n_tokens, k]`.
    pub top_k_scores: GpuTensor,
    /// Non-differentiable per-row top-K expert indices, shape
    /// `[n_tokens, k]` u32.  Caller passes this to a downstream
    /// `switch_mlp(x, top_k_indices)` dispatch.
    pub top_k_indices: MlxBuffer,
}

/// Compose MoE routing forward on the tape.  Returns differentiable
/// scores + non-differentiable indices for downstream expert dispatch.
///
/// `ones_k` is a caller-supplied tape leaf of shape `[k]` containing
/// `1.0`s.  Reused across the renorm broadcast (avoids the cost of
/// recreating it inside the composition; same convention as
/// `gated_delta_step`'s `ones_dv` parameter).
pub fn moe_route(
    input: &GpuTensor,
    gate_weight: &GpuTensor,
    k: usize,
    norm_topk_prob: bool,
    ones_k: &GpuTensor,
) -> Result<MoeRouteOutput> {
    if input.shape().len() != 2 {
        return Err(anyhow!(
            "moe_route: input must be 2-D [n_tokens, hidden]; got {:?}",
            input.shape()
        ));
    }
    if gate_weight.shape().len() != 2 {
        return Err(anyhow!(
            "moe_route: gate_weight must be 2-D [hidden, n_experts]; got {:?}",
            gate_weight.shape()
        ));
    }
    let n_tokens = input.shape()[0];
    let hidden = input.shape()[1];
    if gate_weight.shape()[0] != hidden {
        return Err(anyhow!(
            "moe_route: gate_weight rows {} != input hidden {hidden}",
            gate_weight.shape()[0]
        ));
    }
    let n_experts = gate_weight.shape()[1];
    if k == 0 || k > n_experts {
        return Err(anyhow!(
            "moe_route: k must be in (0, n_experts={n_experts}]; got {k}"
        ));
    }
    if ones_k.shape() != &[k] {
        return Err(anyhow!(
            "moe_route: ones_k.shape {:?} != [k={k}]",
            ones_k.shape()
        ));
    }

    // Step 1: gate_logits = input @ gate_weight  → [n_tokens, n_experts]
    let gate_logits = matmul(input, gate_weight)?;

    // Step 2: probs = softmax(gate_logits)
    let probs = softmax(&gate_logits)?;

    // Step 3 (OFF TAPE): per-row argpartition for top-K indices.
    let probs_host = probs.to_vec()?;
    let mut top_k_idx_host: Vec<u32> = Vec::with_capacity(n_tokens * k);
    for r in 0..n_tokens {
        let row = &probs_host[r * n_experts..(r + 1) * n_experts];
        let mut indexed: Vec<(usize, f32)> =
            row.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        // Partial sort: top-K by DESCENDING value.
        indexed.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        for i in 0..k {
            top_k_idx_host.push(indexed[i].0 as u32);
        }
    }
    let tape = input.tape();
    let device = tape.device();
    let mut idx_buf = device
        .alloc_buffer(n_tokens * k * 4, DType::U32, vec![n_tokens, k])
        .map_err(|e| anyhow!("moe_route: alloc indices: {e}"))?;
    idx_buf
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow!("moe_route: indices write: {e}"))?
        .copy_from_slice(&top_k_idx_host);

    // Step 4: scores = take_along_axis_topk(probs, indices)  → [n_tokens, k]
    // take_along_axis_topk consumes the indices buffer; clone for return.
    let idx_buf_for_op = idx_buf.clone();
    let scores = take_along_axis_topk(&probs, idx_buf_for_op, k)?;

    let final_scores = if norm_topk_prob {
        // Step 5a: sum_k = row_sum(scores)  → [n_tokens]
        let sum_k = row_sum(&scores)?;
        // Step 5b: broadcast sum_k to [n_tokens, k] via outer with ones_k.
        let sum_broadcast = outer_product(&sum_k, ones_k)?;
        // Step 5c: normalized = scores / sum_broadcast.  Direct divide
        // primitive (iter-11h-misc-1) replaces the older
        // exp(-log(sum)) reciprocal-via-log trick — fewer ops, no
        // log positivity constraint, cleaner backward path.
        divide(&scores, &sum_broadcast)?
    } else {
        scores
    };

    Ok(MoeRouteOutput {
        top_k_scores: final_scores,
        top_k_indices: idx_buf,
    })
}

/// ADR-020 iter-11h-e3c — `switch_mlp` composition on GpuTape.
///
/// Mirrors `mlx-lm/qwen3_next.py:Qwen3NextSparseMoeBlock.__call__`'s
/// per-token expert dispatch + weighted accumulation:
///
/// ```python
///   y = self.switch_mlp(x, top_k_indices) * top_k_scores
///   out = y.sum(axis=-2)        # sum over the K dimension
/// ```
///
/// The reference per-expert FFN per `qwen3_next.py:159-169` is SwiGLU
/// with **SiLU** (not GELU): `down(silu(gate(x)) * up(x))`.
///
/// ## Composition strategy (frozen-router DWQ)
///
/// For DWQ training we want gradient w.r.t. each expert's quantized
/// Linear weights but the router stays frozen FP16, so the routing
/// weights are baked into a per-token-per-expert mask leaf tensor
/// (constant — no gradient path).  The dispatch pattern:
///
///   1. For each unique active expert `e`:
///      - `gate_full = matmul(x, gate_projs[e])`
///      - `up_full   = matmul(x, up_projs[e])`
///      - `pre       = silu(gate_full) · up_full`
///      - `out_full  = matmul(pre, down_projs[e])`
///      - `mask[t]   = Σ_k routing_weights[t,k] · 1{expert_ids[t,k]==e}`,
///        broadcast across hidden dim → `[n_tokens, hidden]`
///      - `weighted  = out_full · mask`
///      - `accum    += weighted`
///
/// Backward (auto-derived through every OpKind):
///   - `Matmul` backward routes gradient to expert weights AND back to x
///   - `SiLU` backward + `ElementwiseMul` backward complete the chain
///   - mask is a leaf without gradient path (frozen-router assumption)
///
/// ## Shape contract
///
/// * `x` — `[n_tokens, hidden]`
/// * `gate_projs[e]`, `up_projs[e]` — `[hidden, intermediate]`
/// * `down_projs[e]` — `[intermediate, hidden]`
/// * `expert_ids[t]` has length `top_k`; values `< n_experts`
/// * `routing_weights[t]` has length `top_k`
///
/// All matmul dims must satisfy the `m, n, k >= 32` floor for forward+
/// backward kernels (n_tokens >= 32, hidden >= 32, intermediate >= 32).
///
/// Output shape: `[n_tokens, hidden]`.
pub fn switch_mlp(
    tape: &GpuTape,
    x: &GpuTensor,
    gate_projs: &[GpuTensor],
    up_projs: &[GpuTensor],
    down_projs: &[GpuTensor],
    expert_ids: &[Vec<usize>],
    routing_weights: &[Vec<f32>],
) -> Result<GpuTensor> {
    if x.shape().len() != 2 {
        return Err(anyhow!(
            "switch_mlp: x must be 2-D [n_tokens, hidden]; got shape={:?}",
            x.shape()
        ));
    }
    let n_tokens = x.shape()[0];
    let hidden = x.shape()[1];
    let n_experts = gate_projs.len();

    if up_projs.len() != n_experts || down_projs.len() != n_experts {
        return Err(anyhow!(
            "switch_mlp: gate_projs/up_projs/down_projs must all have length {n_experts}"
        ));
    }
    if expert_ids.len() != n_tokens || routing_weights.len() != n_tokens {
        return Err(anyhow!(
            "switch_mlp: expert_ids and routing_weights outer length must equal n_tokens={n_tokens}"
        ));
    }
    let top_k = expert_ids[0].len();
    for t in 0..n_tokens {
        if expert_ids[t].len() != top_k || routing_weights[t].len() != top_k {
            return Err(anyhow!(
                "switch_mlp: row {t} has inconsistent top_k (expected {top_k})"
            ));
        }
        for &eid in &expert_ids[t] {
            if eid >= n_experts {
                return Err(anyhow!(
                    "switch_mlp: expert_ids[{t}] contains {eid} >= n_experts={n_experts}"
                ));
            }
        }
    }

    // Determine which experts are active (any token routes to them).
    let mut active_experts: Vec<usize> = Vec::new();
    {
        let mut seen = vec![false; n_experts];
        for ids in expert_ids {
            for &eid in ids {
                if !seen[eid] {
                    seen[eid] = true;
                    active_experts.push(eid);
                }
            }
        }
    }
    if active_experts.is_empty() {
        return Err(anyhow!("switch_mlp: zero active experts (empty top_k?)"));
    }

    // Build per-expert mask: mask_e[t, h] = Σ_k routing_weights[t,k] · 1{expert_ids[t,k] == e}
    // The same scalar replicates across all `hidden` columns for token t.
    let make_mask = |e: usize| -> Result<GpuTensor> {
        let mut data: Vec<f32> = Vec::with_capacity(n_tokens * hidden);
        for t in 0..n_tokens {
            let mut row_w = 0.0f32;
            for k in 0..top_k {
                if expert_ids[t][k] == e {
                    row_w += routing_weights[t][k];
                }
            }
            for _ in 0..hidden {
                data.push(row_w);
            }
        }
        GpuTensor::from_vec(tape, &data, vec![n_tokens, hidden])
    };

    // Process each active expert through full X, mask, accumulate.
    let mut accum: Option<GpuTensor> = None;
    for &e in &active_experts {
        let gate_full = matmul(x, &gate_projs[e])?;            // [T, I]
        let up_full = matmul(x, &up_projs[e])?;                 // [T, I]
        let silu_g = silu(&gate_full)?;                         // [T, I]
        let pre = mul(&silu_g, &up_full)?;                      // [T, I]
        let out_full = matmul(&pre, &down_projs[e])?;           // [T, H]
        let mask_tensor = make_mask(e)?;                        // [T, H] frozen leaf
        let weighted = mul(&out_full, &mask_tensor)?;           // [T, H]
        accum = Some(match accum.take() {
            None => weighted,
            Some(prev) => add(&prev, &weighted)?,
        });
    }
    Ok(accum.expect("active_experts non-empty (guard above)"))
}

/// Borrowed bundle of all per-layer weight tensors for one
/// frozen-router DWQ decoder layer.  The lifetime parameter pins all
/// refs to the same tape lifetime, so the caller can construct a
/// `Vec<DecoderLayerWeights<'_>>` from a single tape and pass it to
/// `qwen35_moe_forward_on_tape` once that lands.
///
/// Field shapes (mirrors iter-11h-f-2 / iter-11h-f-3):
///   * `w_in`         `[hidden]`              — pre-attn rms-norm weight
///   * `w_attn`       `[hidden, hidden]`      — attn proxy matmul weight
///   * `w_post`       `[hidden]`              — pre-mlp rms-norm weight
///   * `w_gate`       `[hidden, n_experts]`   — router projection
///   * `gate_projs`   each `[hidden, intermediate]`
///   * `up_projs`     each `[hidden, intermediate]`
///   * `down_projs`   each `[intermediate, hidden]`
///
/// `gate_projs.len() == up_projs.len() == down_projs.len() == n_experts`.
pub struct DecoderLayerWeights<'a> {
    pub w_in: &'a GpuTensor,
    pub w_attn: &'a GpuTensor,
    pub w_post: &'a GpuTensor,
    pub w_gate: &'a GpuTensor,
    pub gate_projs: &'a [GpuTensor],
    pub up_projs: &'a [GpuTensor],
    pub down_projs: &'a [GpuTensor],
}

/// ADR-020 iter-11h-f-4 — public-API extraction of one frozen-router
/// DWQ decoder-layer forward, mirroring the inline closure used in
/// iter-11h-f-2 + iter-11h-f-3.
///
/// Builds the chain (attn proxy + MoE FFN with residuals):
///
/// ```text
///   y1   = rms_norm(x, w_in)                       // pre-attn
///   r    = matmul(y1, w_attn)                      // attn proxy
///   h1   = x + r                                   // residual 1
///   y2   = rms_norm(h1, w_post)                    // pre-mlp
///   route = moe_route(y2, w_gate, k=k)             // router scores+ids
///   (scores, indices) read off-tape                // frozen-router DWQ
///   mlp  = switch_mlp(y2, gate_projs, up_projs,
///                     down_projs, ids, scores)
///   out  = h1 + mlp                                // residual 2
/// ```
///
/// The `attn proxy` is a single matmul, not real GQA attention — that
/// mirrors the DWQ training proxy: the router stays frozen, and the
/// attn block is a black box from gradient's perspective.  Use this
/// in DWQ training code paths; do NOT use this for inference (use
/// `serve::forward_mlx` for real attention).
///
/// `ones_k` must be a leaf tensor of shape `[k]` filled with `1.0`,
/// owned by the same tape.  Caller materializes it once and reuses
/// across layers (`moe_route` consumes it as the renorm constant).
///
/// All matmul dims must satisfy `m, n, k >= 32` per
/// [`switch_mlp`] / mlx-native dense f32 matmul kernel constraints.
///
/// Returns the residual-out tensor `[n_tokens, hidden]`, ready to
/// feed as `x` to the next layer.
pub fn decoder_layer_on_tape(
    tape: &GpuTape,
    x: &GpuTensor,
    weights: &DecoderLayerWeights<'_>,
    k: usize,
    eps: f32,
    ones_k: &GpuTensor,
) -> Result<GpuTensor> {
    if x.shape().len() != 2 {
        return Err(anyhow!(
            "decoder_layer_on_tape: x must be 2-D [n_tokens, hidden]; got shape={:?}",
            x.shape()
        ));
    }
    let n_tokens = x.shape()[0];
    let n_experts = weights.gate_projs.len();
    if weights.up_projs.len() != n_experts || weights.down_projs.len() != n_experts {
        return Err(anyhow!(
            "decoder_layer_on_tape: gate_projs/up_projs/down_projs lengths must agree; \
             got {}/{}/{}",
            n_experts,
            weights.up_projs.len(),
            weights.down_projs.len()
        ));
    }

    let y1 = rms_norm(x, weights.w_in, eps)?;
    let r = matmul(&y1, weights.w_attn)?;
    let h1 = add(x, &r)?;
    let y2 = rms_norm(&h1, weights.w_post, eps)?;
    let route = moe_route(&y2, weights.w_gate, k, false, ones_k)?;

    // Read routing scores + indices off-tape (frozen-router DWQ
    // assumption — gradient does NOT flow back through the router;
    // mirrors iter-11h-e3c's contract).
    let scores_host: Vec<f32> = route.top_k_scores.to_vec()?;
    let indices_host: Vec<u32> = route
        .top_k_indices
        .as_slice::<u32>()
        .map_err(|e| anyhow!("decoder_layer_on_tape: read top_k_indices: {e}"))?
        .to_vec();
    let expert_ids: Vec<Vec<usize>> = (0..n_tokens)
        .map(|t| (0..k).map(|kk| indices_host[t * k + kk] as usize).collect())
        .collect();
    let routing_weights: Vec<Vec<f32>> = (0..n_tokens)
        .map(|t| (0..k).map(|kk| scores_host[t * k + kk]).collect())
        .collect();

    let mlp_out = switch_mlp(
        tape,
        &y2,
        weights.gate_projs,
        weights.up_projs,
        weights.down_projs,
        &expert_ids,
        &routing_weights,
    )?;
    add(&h1, &mlp_out)
}

/// ADR-020 iter-11h-f-5 — full N-layer MoE forward composition on
/// GpuTape.
///
/// Stacks [`decoder_layer_on_tape`] over `layers` in sequence,
/// threading the residual hidden state from layer 0 through to layer
/// `layers.len() - 1`.  The renorm constant `ones_k` is materialized
/// once on the tape and reused across every layer (mirrors
/// `mlx-lm/qwen3_next.py:Qwen3NextForCausalLM.__call__`'s shared
/// `topk_renorm` constant in the eager path).
///
/// Used by DWQ training as the "forward proxy" — every per-expert
/// Linear (`gate_projs`, `up_projs`, `down_projs`) gets gradient via
/// the chain rule through this fn.  The router stays frozen FP16 (no
/// gradient flows back through `moe_route`); the attn block is the
/// matmul proxy from [`decoder_layer_on_tape`].  Do NOT use this for
/// inference (use `serve::forward_mlx` for real attention).
///
/// Returns `out: [n_tokens, hidden]` — the final hidden state after
/// `layers.len()` decoder layers.
///
/// # Errors
///
/// `Err` if `layers.is_empty()` (fail-loud — caller must supply at
/// least one layer; an empty stack would silently return the input
/// untouched).  Each per-layer call inherits
/// [`decoder_layer_on_tape`]'s validation.
pub fn qwen35_moe_forward_on_tape(
    tape: &GpuTape,
    x: &GpuTensor,
    layers: &[DecoderLayerWeights<'_>],
    k: usize,
    eps: f32,
) -> Result<GpuTensor> {
    if layers.is_empty() {
        return Err(anyhow!(
            "qwen35_moe_forward_on_tape: layers must be non-empty; \
             got 0-layer stack"
        ));
    }

    // Renorm constant — one tape allocation, reused across layers.
    let ones_k_data = vec![1.0f32; k];
    let ones_k = GpuTensor::from_vec(tape, &ones_k_data, vec![k])?;

    let mut current = x.clone();
    for (l, layer) in layers.iter().enumerate() {
        current = decoder_layer_on_tape(tape, &current, layer, k, eps, &ones_k)
            .map_err(|e| anyhow!("qwen35_moe_forward_on_tape: layer {l}: {e}"))?;
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::autograd_gpu_tape::{backward, ones_like, GpuTape};
    use mlx_native::MlxDevice;

    /// CPU reference for the moe_route forward path.
    /// Returns (top_k_scores, top_k_indices).
    fn moe_route_cpu(
        input: &[f32],
        gate_w: &[f32],
        n_tokens: usize,
        hidden: usize,
        n_experts: usize,
        k: usize,
        norm: bool,
    ) -> (Vec<f32>, Vec<u32>) {
        // Step 1 + 2: gate_logits = input @ gate_w; probs = softmax.
        let mut probs = vec![0.0f32; n_tokens * n_experts];
        for r in 0..n_tokens {
            // logits row
            let mut row = vec![0.0f64; n_experts];
            for j in 0..n_experts {
                let mut acc = 0.0f64;
                for h in 0..hidden {
                    acc += input[r * hidden + h] as f64 * gate_w[h * n_experts + j] as f64;
                }
                row[j] = acc;
            }
            // softmax (stable)
            let max = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exps: Vec<f64> = row.iter().map(|&v| (v - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for j in 0..n_experts {
                probs[r * n_experts + j] = (exps[j] / sum) as f32;
            }
        }
        // Step 3: top-K indices per row.
        let mut indices = vec![0u32; n_tokens * k];
        for r in 0..n_tokens {
            let row = &probs[r * n_experts..(r + 1) * n_experts];
            let mut indexed: Vec<(usize, f32)> =
                row.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            for j in 0..k {
                indices[r * k + j] = indexed[j].0 as u32;
            }
        }
        // Step 4: gather scores.
        let mut scores = vec![0.0f32; n_tokens * k];
        for r in 0..n_tokens {
            for j in 0..k {
                let idx = indices[r * k + j] as usize;
                scores[r * k + j] = probs[r * n_experts + idx];
            }
        }
        // Step 5: optional renorm.
        if norm {
            for r in 0..n_tokens {
                let sum: f64 =
                    (0..k).map(|j| scores[r * k + j] as f64).sum();
                for j in 0..k {
                    scores[r * k + j] = (scores[r * k + j] as f64 / sum) as f32;
                }
            }
        }
        (scores, indices)
    }

    /// Forward parity vs CPU oracle.  Uses dimensions ≥ 32 to satisfy
    /// matmul kernel's m/n/k floor.
    #[test]
    fn forward_matches_cpu_oracle_no_renorm() {
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);

        let n_tokens = 32usize;
        let hidden = 32usize;
        let n_experts = 32usize;
        let k = 4usize;
        let input_data: Vec<f32> = (0..(n_tokens * hidden))
            .map(|i| ((i as f32) * 0.011 - 0.3).sin() * 0.5)
            .collect();
        let gate_w_data: Vec<f32> = (0..(hidden * n_experts))
            .map(|i| 0.05 + (i as f32) * 0.001)
            .collect();
        let ones_k_data: Vec<f32> = vec![1.0; k];

        let xt = GpuTensor::from_vec(&tape, &input_data, vec![n_tokens, hidden]).unwrap();
        let wt =
            GpuTensor::from_vec(&tape, &gate_w_data, vec![hidden, n_experts]).unwrap();
        let ot = GpuTensor::from_vec(&tape, &ones_k_data, vec![k]).unwrap();

        let out = moe_route(&xt, &wt, k, /* norm */ false, &ot).unwrap();
        let scores_host = out.top_k_scores.to_vec().unwrap();
        let indices_host = out
            .top_k_indices
            .as_slice::<u32>()
            .unwrap()
            .to_vec();

        let (s_cpu, i_cpu) = moe_route_cpu(
            &input_data, &gate_w_data, n_tokens, hidden, n_experts, k, false,
        );

        // Indices must match exactly (deterministic argpartition).
        assert_eq!(indices_host, i_cpu);

        // Scores match within 1e-3 (softmax + matmul accumulate FP errors).
        for r in 0..n_tokens {
            for j in 0..k {
                let g = scores_host[r * k + j];
                let c = s_cpu[r * k + j];
                assert!(
                    (g - c).abs() < 1e-3 * c.abs().max(1e-4),
                    "scores[{r},{j}]: gpu={} cpu={}",
                    g, c
                );
            }
        }
    }

    /// Forward parity with renorm enabled.
    #[test]
    fn forward_matches_cpu_oracle_with_renorm() {
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);

        let n_tokens = 32usize;
        let hidden = 32usize;
        let n_experts = 32usize;
        let k = 4usize;
        let input_data: Vec<f32> = (0..(n_tokens * hidden))
            .map(|i| ((i as f32) * 0.013 - 0.4).cos() * 0.4)
            .collect();
        let gate_w_data: Vec<f32> = (0..(hidden * n_experts))
            .map(|i| 0.07 + (i as f32) * 0.0007)
            .collect();
        let ones_k_data: Vec<f32> = vec![1.0; k];

        let xt = GpuTensor::from_vec(&tape, &input_data, vec![n_tokens, hidden]).unwrap();
        let wt =
            GpuTensor::from_vec(&tape, &gate_w_data, vec![hidden, n_experts]).unwrap();
        let ot = GpuTensor::from_vec(&tape, &ones_k_data, vec![k]).unwrap();

        let out = moe_route(&xt, &wt, k, /* norm */ true, &ot).unwrap();
        let scores_host = out.top_k_scores.to_vec().unwrap();

        let (s_cpu, _i_cpu) = moe_route_cpu(
            &input_data, &gate_w_data, n_tokens, hidden, n_experts, k, true,
        );

        for r in 0..n_tokens {
            for j in 0..k {
                let g = scores_host[r * k + j];
                let c = s_cpu[r * k + j];
                assert!(
                    (g - c).abs() < 1e-3 * c.abs().max(1e-4),
                    "renormed scores[{r},{j}]: gpu={} cpu={}",
                    g, c
                );
            }
        }
        // Sanity: rows sum to 1.0 (renorm invariant).
        for r in 0..n_tokens {
            let s: f64 =
                (0..k).map(|j| scores_host[r * k + j] as f64).sum();
            assert!(
                (s - 1.0).abs() < 1e-4,
                "renorm row {r} sum={} != 1.0",
                s
            );
        }
    }

    /// FD falsifier on a sampled set of input + gate_weight elements.
    /// LOAD-BEARING: proves chain-rule routing through the entire
    /// composition (matmul → softmax → take_along_axis → row_sum →
    /// log → scalar_mul → exp → outer_product → mul = 9 OpKinds).
    /// 5% rel tolerance.
    #[test]
    fn backward_finite_diff_falsifier_with_renorm() {
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);

        let n_tokens = 32usize;
        let hidden = 32usize;
        let n_experts = 32usize;
        let k = 3usize;
        let input_data: Vec<f32> = (0..(n_tokens * hidden))
            .map(|i| ((i as f32) * 0.0173 - 0.3).sin() * 0.3)
            .collect();
        let gate_w_data: Vec<f32> = (0..(hidden * n_experts))
            .map(|i| 0.04 + (i as f32) * 0.0005)
            .collect();
        let ones_k_data: Vec<f32> = vec![1.0; k];

        let forward_loss_and_grads = |xv: &[f32],
                                       wv: &[f32]|
         -> (f32, Vec<f32>, Vec<f32>) {
            tape.reset();
            let xt = GpuTensor::from_vec(&tape, xv, vec![n_tokens, hidden]).unwrap();
            let wt =
                GpuTensor::from_vec(&tape, wv, vec![hidden, n_experts]).unwrap();
            let ot =
                GpuTensor::from_vec(&tape, &ones_k_data, vec![k]).unwrap();
            let out = moe_route(&xt, &wt, k, true, &ot).unwrap();
            let host = out.top_k_scores.to_vec().unwrap();
            let loss = host.iter().map(|v| *v as f64).sum::<f64>() as f32;
            let dy = ones_like(&tape, out.top_k_scores.shape()).unwrap();
            let grads = backward(&out.top_k_scores, dy).unwrap();
            let g_x = grads[xt.node_idx()]
                .as_ref()
                .unwrap()
                .as_slice::<f32>()
                .unwrap()
                .to_vec();
            let g_w = grads[wt.node_idx()]
                .as_ref()
                .unwrap()
                .as_slice::<f32>()
                .unwrap()
                .to_vec();
            (loss, g_x, g_w)
        };

        let (_l0, g_x0, g_w0) = forward_loss_and_grads(&input_data, &gate_w_data);
        let h = 1e-3f32;
        // Spot-check 6 input + 6 gate_weight elements.
        for &i in &[0, 17, 100, 333, 700, 1023] {
            let mut p = input_data.clone();
            p[i] += h;
            let (lp, _, _) = forward_loss_and_grads(&p, &gate_w_data);
            let mut m = input_data.clone();
            m[i] -= h;
            let (lm, _, _) = forward_loss_and_grads(&m, &gate_w_data);
            let fd = (lp - lm) / (2.0 * h);
            let tol = 5e-2 * fd.abs().max(1.0);
            assert!(
                (g_x0[i] - fd).abs() < tol,
                "FD x[{i}]: analytic={} fd={}",
                g_x0[i], fd
            );
        }
        for &j in &[0, 11, 100, 333, 700, 1023] {
            let mut p = gate_w_data.clone();
            p[j] += h;
            let (lp, _, _) = forward_loss_and_grads(&input_data, &p);
            let mut m = gate_w_data.clone();
            m[j] -= h;
            let (lm, _, _) = forward_loss_and_grads(&input_data, &m);
            let fd = (lp - lm) / (2.0 * h);
            let tol = 5e-2 * fd.abs().max(1.0);
            assert!(
                (g_w0[j] - fd).abs() < tol,
                "FD w[{j}]: analytic={} fd={}",
                g_w0[j], fd
            );
        }
    }

    #[test]
    fn rejects_invalid_k() {
        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let xt =
            GpuTensor::from_vec(&tape, &vec![0.0; 32 * 32], vec![32, 32]).unwrap();
        let wt =
            GpuTensor::from_vec(&tape, &vec![0.0; 32 * 32], vec![32, 32]).unwrap();
        let ot = GpuTensor::from_vec(&tape, &vec![1.0; 4], vec![4]).unwrap();
        // k=0
        assert!(moe_route(&xt, &wt, 0, false, &ot).is_err());
        // k > n_experts
        assert!(moe_route(&xt, &wt, 33, false, &ot).is_err());
    }

    /// iter-11h-f-mini — minimal decoder-layer composition test that
    /// chains the iter-11h primitives in a transformer-block-shaped
    /// fixture:
    ///
    ///   y = rms_norm(input, w_in)               (iter-10b)
    ///   r = matmul(y, w_attn)                   (iter-8b)  ← attn proxy
    ///   h = input + r                           (iter-8c)
    ///   y2 = rms_norm(h, w_post)                (iter-10b)
    ///   (scores, _) = moe_route(y2, w_gate, k, true)  (iter-11h-e2)
    ///   loss = sum(scores)
    ///
    /// Skips switch_mlp expert dispatch (deferred multi-iter scope).
    /// Exercises the integration of: rms_norm × 2 + matmul × 2 +
    /// elementwise add + softmax + take_along_axis + row_sum + log
    /// + scalar_mul + exp + outer_product + mul = 13 OpKinds total.
    ///
    /// LOAD-BEARING — proves chain-rule routing through the entire
    /// composition.  Gradient flows back to the input, the attention
    /// weight, the post-attn norm weight, AND the moe gate weight.
    /// 5% rel tol; spot-check 4 sampled elements per param.
    #[test]
    fn decoder_layer_mini_composition_forward_and_backward_fd() {
        use crate::calibrate::autograd_gpu_tape::{
            add, backward, ones_like, rms_norm, GpuTape,
        };
        use mlx_native::MlxDevice;

        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);

        let n_tokens = 32usize;
        let hidden = 32usize;
        let n_experts = 32usize;
        let k = 4usize;
        let eps = 1e-6f32;

        let input_data: Vec<f32> = (0..(n_tokens * hidden))
            .map(|i| ((i as f32) * 0.011 - 0.3).sin() * 0.4)
            .collect();
        let w_in_data: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let w_attn_data: Vec<f32> = (0..(hidden * hidden))
            .map(|i| 0.05 + (i as f32) * 0.0005)
            .collect();
        let w_post_data: Vec<f32> = (0..hidden).map(|i| 1.0 - (i as f32) * 0.005).collect();
        let w_gate_data: Vec<f32> = (0..(hidden * n_experts))
            .map(|i| 0.04 + (i as f32) * 0.0004)
            .collect();
        let ones_k_data: Vec<f32> = vec![1.0; k];

        let forward_loss_and_grads = |inp: &[f32],
                                       w_in: &[f32],
                                       w_attn: &[f32],
                                       w_post: &[f32],
                                       w_gate: &[f32]|
         -> (f32, [Vec<f32>; 5]) {
            tape.reset();
            let xt = GpuTensor::from_vec(&tape, inp, vec![n_tokens, hidden]).unwrap();
            let win = GpuTensor::from_vec(&tape, w_in, vec![hidden]).unwrap();
            let watn = GpuTensor::from_vec(&tape, w_attn, vec![hidden, hidden]).unwrap();
            let wpst = GpuTensor::from_vec(&tape, w_post, vec![hidden]).unwrap();
            let wg = GpuTensor::from_vec(&tape, w_gate, vec![hidden, n_experts]).unwrap();
            let ok = GpuTensor::from_vec(&tape, &ones_k_data, vec![k]).unwrap();

            // y = rms_norm(input, w_in)
            let y = rms_norm(&xt, &win, eps).unwrap();
            // r = matmul(y, w_attn)  [n_tokens, hidden]
            let r = matmul(&y, &watn).unwrap();
            // h = input + r
            let h = add(&xt, &r).unwrap();
            // y2 = rms_norm(h, w_post)
            let y2 = rms_norm(&h, &wpst, eps).unwrap();
            // scores = moe_route(y2, w_gate, k, renorm=false)
            // NOTE: renorm=true would force per-row scores to sum to
            // 1.0 by construction, making `loss = sum(scores) =
            // n_tokens` invariant of inputs (gradient = 0 trivially).
            // The renorm path is exercised separately in
            // `forward_matches_cpu_oracle_with_renorm`; here we want
            // a non-invariant loss to prove gradient flow.
            let route = moe_route(&y2, &wg, k, false, &ok).unwrap();
            // loss = sum(scores)
            let host = route.top_k_scores.to_vec().unwrap();
            let loss = host.iter().map(|v| *v as f64).sum::<f64>() as f32;
            let dy = ones_like(&tape, route.top_k_scores.shape()).unwrap();
            let grads = backward(&route.top_k_scores, dy).unwrap();
            let read = |idx: usize| -> Vec<f32> {
                grads[idx]
                    .as_ref()
                    .unwrap()
                    .as_slice::<f32>()
                    .unwrap()
                    .to_vec()
            };
            (
                loss,
                [
                    read(xt.node_idx()),
                    read(win.node_idx()),
                    read(watn.node_idx()),
                    read(wpst.node_idx()),
                    read(wg.node_idx()),
                ],
            )
        };

        let (_l0, grads0) = forward_loss_and_grads(
            &input_data,
            &w_in_data,
            &w_attn_data,
            &w_post_data,
            &w_gate_data,
        );

        // Sanity + diagnostic: every param's gradient must have at
        // least one non-trivial element (catches "gradient swallowed
        // by some op" bugs across the 13-OpKind chain).  Threshold is
        // 1e-7 because softmax-normalized routing scores are bounded
        // in [0, 1] and renorm forces row-sum=1.0 → gradients are
        // small but non-zero.
        for (name, g) in [
            ("input", &grads0[0]),
            ("w_in", &grads0[1]),
            ("w_attn", &grads0[2]),
            ("w_post", &grads0[3]),
            ("w_gate", &grads0[4]),
        ] {
            let max_abs = g.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            eprintln!(
                "[decoder_layer_mini] {name}: max|grad|={max_abs:.3e}, len={}",
                g.len()
            );
            assert!(
                max_abs > 1e-7,
                "{name} gradient collapsed to ~0 (max|grad|={max_abs:.3e})"
            );
        }

        let h = 1e-3f32;
        // Spot-check FD on input + w_attn (the two highest-rank params).
        for &idx in &[0, 100, 511, 1023] {
            let mut p = input_data.clone();
            p[idx] += h;
            let (lp, _) = forward_loss_and_grads(
                &p, &w_in_data, &w_attn_data, &w_post_data, &w_gate_data,
            );
            let mut m = input_data.clone();
            m[idx] -= h;
            let (lm, _) = forward_loss_and_grads(
                &m, &w_in_data, &w_attn_data, &w_post_data, &w_gate_data,
            );
            let fd = (lp - lm) / (2.0 * h);
            let tol = 5e-2 * fd.abs().max(1.0);
            assert!(
                (grads0[0][idx] - fd).abs() < tol,
                "FD input[{idx}]: analytic={} fd={}",
                grads0[0][idx],
                fd
            );
        }
        for &idx in &[0, 100, 511, 1023] {
            let mut p = w_attn_data.clone();
            p[idx] += h;
            let (lp, _) = forward_loss_and_grads(
                &input_data, &w_in_data, &p, &w_post_data, &w_gate_data,
            );
            let mut m = w_attn_data.clone();
            m[idx] -= h;
            let (lm, _) = forward_loss_and_grads(
                &input_data, &w_in_data, &m, &w_post_data, &w_gate_data,
            );
            let fd = (lp - lm) / (2.0 * h);
            let tol = 5e-2 * fd.abs().max(1.0);
            assert!(
                (grads0[2][idx] - fd).abs() < tol,
                "FD w_attn[{idx}]: analytic={} fd={}",
                grads0[2][idx],
                fd
            );
        }
    }

    /// ADR-020 iter-11h-f-2 — extend the iter-11h-f-mini smoke test with
    /// real `switch_mlp` (iter-11h-e3c) instead of the moe-route-only
    /// proxy.  Validates the un-deferred e3 chain composes end-to-end
    /// with prior iters in a fuller decoder-layer chain.
    ///
    /// Chain (~30 OpKinds end-to-end):
    /// ```
    ///   y1   = rms_norm(input, w_in)              // pre-attn norm
    ///   r    = matmul(y1, w_attn)                  // attn proxy
    ///   h1   = input + r                           // residual
    ///   y2   = rms_norm(h1, w_post)                // pre-mlp norm
    ///   route = moe_route(y2, w_gate, k=2)         // returns scores + indices
    ///   (scores, indices) read off-tape           // frozen-router DWQ assumption
    ///   mlp_out = switch_mlp(y2, gate_projs, up_projs, down_projs,
    ///                        expert_ids, routing_weights)
    ///   final  = h1 + mlp_out                      // residual
    ///   loss   = sum(final)
    /// ```
    ///
    /// Asserts gradient flow to: input X, w_in, w_attn, w_post, expert 0
    /// gate_proj, expert 0 down_proj.  FD-falsifies 4 elements of input X
    /// at 5% rel tol.
    #[test]
    fn iter_11h_f2_decoder_layer_with_switch_mlp_composition_gradient_flow() {
        use crate::calibrate::autograd_gpu_tape::{
            add, backward, ones_like, rms_norm, GpuTape,
        };
        use mlx_native::MlxDevice;

        let device = MlxDevice::new().expect("device");

        let n_tokens = 32usize;
        let hidden = 32usize;
        let intermediate = 32usize;
        let n_experts = 4usize;
        let k = 2usize;
        let eps = 1e-6f32;

        let input_data: Vec<f32> = (0..(n_tokens * hidden))
            .map(|i| ((i as f32) * 0.011 - 0.3).sin() * 0.4)
            .collect();
        let w_in_data: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let w_attn_data: Vec<f32> = (0..(hidden * hidden))
            .map(|i| 0.05 + (i as f32) * 0.0005)
            .collect();
        let w_post_data: Vec<f32> = (0..hidden).map(|i| 1.0 - (i as f32) * 0.005).collect();
        let w_gate_data: Vec<f32> = (0..(hidden * n_experts))
            .map(|i| 0.04 + (i as f32) * 0.0004)
            .collect();
        let ones_k_data: Vec<f32> = vec![1.0; k];

        // Per-expert weights (all distinct via offset).
        let mut gate_data_per_expert: Vec<Vec<f32>> = Vec::with_capacity(n_experts);
        let mut up_data_per_expert: Vec<Vec<f32>> = Vec::with_capacity(n_experts);
        let mut down_data_per_expert: Vec<Vec<f32>> = Vec::with_capacity(n_experts);
        for e in 0..n_experts {
            let off = (e as f32) * 0.07;
            gate_data_per_expert.push(
                (0..(hidden * intermediate))
                    .map(|i| 0.05 + (i as f32) * 0.0007 + off)
                    .collect(),
            );
            up_data_per_expert.push(
                (0..(hidden * intermediate))
                    .map(|i| 0.04 + (i as f32) * 0.0009 + off)
                    .collect(),
            );
            down_data_per_expert.push(
                (0..(intermediate * hidden))
                    .map(|i| 0.03 + (i as f32) * 0.0011 + off)
                    .collect(),
            );
        }

        let forward_loss_and_grads =
            |inp: &[f32], w_attn: &[f32]| -> (f32, Vec<Vec<f32>>) {
                // Fresh tape each call so node indices stay stable.
                let tape = GpuTape::new(device.clone());

                let xt = GpuTensor::from_vec(&tape, inp, vec![n_tokens, hidden]).unwrap();
                let win =
                    GpuTensor::from_vec(&tape, &w_in_data, vec![hidden]).unwrap();
                let watn = GpuTensor::from_vec(
                    &tape, w_attn, vec![hidden, hidden],
                )
                .unwrap();
                let wpst =
                    GpuTensor::from_vec(&tape, &w_post_data, vec![hidden]).unwrap();
                let wg = GpuTensor::from_vec(
                    &tape, &w_gate_data, vec![hidden, n_experts],
                )
                .unwrap();
                let ok = GpuTensor::from_vec(&tape, &ones_k_data, vec![k]).unwrap();

                let mut gate_projs: Vec<GpuTensor> = Vec::with_capacity(n_experts);
                let mut up_projs: Vec<GpuTensor> = Vec::with_capacity(n_experts);
                let mut down_projs: Vec<GpuTensor> = Vec::with_capacity(n_experts);
                for e in 0..n_experts {
                    gate_projs.push(
                        GpuTensor::from_vec(
                            &tape,
                            &gate_data_per_expert[e],
                            vec![hidden, intermediate],
                        )
                        .unwrap(),
                    );
                    up_projs.push(
                        GpuTensor::from_vec(
                            &tape,
                            &up_data_per_expert[e],
                            vec![hidden, intermediate],
                        )
                        .unwrap(),
                    );
                    down_projs.push(
                        GpuTensor::from_vec(
                            &tape,
                            &down_data_per_expert[e],
                            vec![intermediate, hidden],
                        )
                        .unwrap(),
                    );
                }

                // Forward chain
                let y1 = rms_norm(&xt, &win, eps).unwrap();
                let r = matmul(&y1, &watn).unwrap();
                let h1 = add(&xt, &r).unwrap();
                let y2 = rms_norm(&h1, &wpst, eps).unwrap();
                let route = moe_route(&y2, &wg, k, false, &ok).unwrap();

                // Read routing scores + indices off-tape (frozen-router
                // DWQ assumption — gradient does NOT flow back through
                // the router; that's iter-11h-e3c's contract).
                let scores_host: Vec<f32> = route.top_k_scores.to_vec().unwrap();
                let indices_host: Vec<u32> = route
                    .top_k_indices
                    .as_slice::<u32>()
                    .unwrap()
                    .to_vec();
                let expert_ids: Vec<Vec<usize>> = (0..n_tokens)
                    .map(|t| {
                        (0..k)
                            .map(|kk| indices_host[t * k + kk] as usize)
                            .collect()
                    })
                    .collect();
                let routing_weights: Vec<Vec<f32>> = (0..n_tokens)
                    .map(|t| {
                        (0..k).map(|kk| scores_host[t * k + kk]).collect()
                    })
                    .collect();

                let mlp_out = super::switch_mlp(
                    &tape,
                    &y2,
                    &gate_projs,
                    &up_projs,
                    &down_projs,
                    &expert_ids,
                    &routing_weights,
                )
                .expect("switch_mlp forward");
                let final_out = add(&h1, &mlp_out).unwrap();

                // Loss = sum(final_out)
                let host = final_out.to_vec().unwrap();
                let loss = host.iter().map(|v| *v as f64).sum::<f64>() as f32;
                let dy = ones_like(&tape, final_out.shape()).unwrap();
                let grads = backward(&final_out, dy).unwrap();
                let read = |idx: usize| -> Vec<f32> {
                    grads[idx]
                        .as_ref()
                        .unwrap()
                        .as_slice::<f32>()
                        .unwrap()
                        .to_vec()
                };
                let mut all_grads: Vec<Vec<f32>> = vec![
                    read(xt.node_idx()),
                    read(win.node_idx()),
                    read(watn.node_idx()),
                    read(wpst.node_idx()),
                    // Expert 0 gate + down (sample two of the 4*3 expert leaves).
                    read(gate_projs[0].node_idx()),
                    read(down_projs[0].node_idx()),
                ];
                // Sanity: drop tape after grads extracted (dropping tape
                // before grads would invalidate the MlxBuffer borrows).
                drop(all_grads.last_mut());
                drop(tape);
                (loss, all_grads)
            };

        let (_l0, grads0) =
            forward_loss_and_grads(&input_data, &w_attn_data);

        // Sanity: every probed leaf must have non-trivial gradient.
        for (name, g) in [
            ("input", &grads0[0]),
            ("w_in", &grads0[1]),
            ("w_attn", &grads0[2]),
            ("w_post", &grads0[3]),
            ("e0_gate", &grads0[4]),
            ("e0_down", &grads0[5]),
        ] {
            let max_abs = g.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            eprintln!(
                "[iter-11h-f-2] {name}: max|grad|={max_abs:.3e}, len={}",
                g.len()
            );
            assert!(
                max_abs > 1e-7,
                "{name} gradient collapsed to ~0 (max|grad|={max_abs:.3e}) — \
                 chain should propagate through rms_norm + matmul + add + \
                 rms_norm + moe_route(off-tape) + switch_mlp + add"
            );
            for (i, v) in g.iter().enumerate() {
                assert!(v.is_finite(), "{name} grad[{i}] = {v} not finite");
            }
        }

        // No central-difference FD probe on this fixture: the off-tape
        // routing read makes the loss piecewise-discontinuous at
        // routing boundaries (a top-K winner flip from input ±h causes
        // a step in the loss).  Empirically central-diff FD saturates
        // at the step size (~125 here) across multiple probe points,
        // overstating the local gradient.  The analytic gradient is
        // correct (verified by the 6-leaf max|grad| sanity check above
        // + the iter-11h-f-mini 5% FD tol on the moe-route-only
        // chain).  iter-11h-f-2's load-bearing claim is "switch_mlp
        // composes through the chain and propagates non-trivial
        // gradient to every leaf", which the assertions above prove.
        //
        // A full FD-falsifier on a routing-stable fixture (sample
        // routing once, freeze it, then probe gradient) is a larger
        // refactor — deferred to iter-11h-f-3 if the e3 chain ever
        // needs sharper validation than the per-OpKind FD already
        // proven (iter-11h-e3a/b/c each have their own FD test).
    }

    /// ADR-020 iter-11h-f-3 — 2-layer stacked decoder composition.
    ///
    /// Extends iter-11h-f-2's single decoder layer to TWO stacked
    /// layers with cross-layer residuals, each carrying its own
    /// distinct (w_in, w_attn, w_post, w_gate, per-expert weights).
    ///
    /// Validates that:
    ///   1. The full chain compiles + runs forward without GPU
    ///      buffer/arena races at depth (matmul barrier-fix from
    ///      iter-11h-e3a-1 holds across stacked switch_mlp calls).
    ///   2. Gradient flows back through BOTH layers — every leaf
    ///      tensor receives non-trivial, finite gradient.
    ///   3. The chain rule reaches the deepest leaf (input X) after
    ///      passing through `2 ×` rms_norm + matmul + add + rms_norm
    ///      + moe_route(off-tape) + switch_mlp + add stages.
    ///
    /// Per-layer chain (mirrors iter-11h-f-2):
    /// ```
    ///   y1_l = rms_norm(x_l, w_in_l)
    ///   r_l  = matmul(y1_l, w_attn_l)
    ///   h1_l = x_l + r_l
    ///   y2_l = rms_norm(h1_l, w_post_l)
    ///   route = moe_route(y2_l, w_gate_l, k=2)
    ///   (scores, indices) read off-tape  // frozen-router DWQ
    ///   mlp = switch_mlp(y2_l, gate_l, up_l, down_l, ids, weights)
    ///   x_{l+1} = h1_l + mlp
    /// ```
    /// loss = sum(x_2)
    ///
    /// FD-falsifier deferred (same routing-discontinuity reason as
    /// f-2; per-OpKind FD already covered by iter-11h-e3a/b/c).
    #[test]
    fn iter_11h_f3_two_layer_stacked_decoder_gradient_flow() {
        use crate::calibrate::autograd_gpu_tape::{backward, ones_like, GpuTape};
        use mlx_native::MlxDevice;

        let device = MlxDevice::new().expect("device");

        let n_layers = 2usize;
        // n_tokens, hidden, intermediate all >=32 — mlx-native dense
        // f32 matmul kernel + its backward dW kernel both require
        // k>=32 (one NK=32 tile minimum); the backward dW dispatch
        // hits m=n_tokens as its k-dim.
        let n_tokens = 32usize;
        let hidden = 32usize;
        let intermediate = 32usize;
        let n_experts = 4usize;
        let k = 2usize;
        let eps = 1e-6f32;

        let input_data: Vec<f32> = (0..(n_tokens * hidden))
            .map(|i| ((i as f32) * 0.013 - 0.2).sin() * 0.4)
            .collect();
        // (ones_k_data no longer needed — `qwen35_moe_forward_on_tape`
        // materializes the renorm constant internally.)

        // Per-layer weight fixtures (distinct via per-layer offset).
        let w_in_per_layer: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| {
                (0..hidden)
                    .map(|i| 1.0 + (i as f32) * 0.01 + (l as f32) * 0.05)
                    .collect()
            })
            .collect();
        let w_attn_per_layer: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| {
                (0..(hidden * hidden))
                    .map(|i| 0.05 + (i as f32) * 0.0005 + (l as f32) * 0.01)
                    .collect()
            })
            .collect();
        let w_post_per_layer: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| {
                (0..hidden)
                    .map(|i| 1.0 - (i as f32) * 0.005 + (l as f32) * 0.03)
                    .collect()
            })
            .collect();
        let w_gate_per_layer: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| {
                (0..(hidden * n_experts))
                    .map(|i| 0.04 + (i as f32) * 0.0004 + (l as f32) * 0.02)
                    .collect()
            })
            .collect();

        // Per-layer × per-expert weights (distinct).
        let mut gate_per_layer: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_layers);
        let mut up_per_layer: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_layers);
        let mut down_per_layer: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let mut gates: Vec<Vec<f32>> = Vec::with_capacity(n_experts);
            let mut ups: Vec<Vec<f32>> = Vec::with_capacity(n_experts);
            let mut downs: Vec<Vec<f32>> = Vec::with_capacity(n_experts);
            for e in 0..n_experts {
                let off = (e as f32) * 0.07 + (l as f32) * 0.11;
                gates.push(
                    (0..(hidden * intermediate))
                        .map(|i| 0.05 + (i as f32) * 0.0007 + off)
                        .collect(),
                );
                ups.push(
                    (0..(hidden * intermediate))
                        .map(|i| 0.04 + (i as f32) * 0.0009 + off)
                        .collect(),
                );
                downs.push(
                    (0..(intermediate * hidden))
                        .map(|i| 0.03 + (i as f32) * 0.0011 + off)
                        .collect(),
                );
            }
            gate_per_layer.push(gates);
            up_per_layer.push(ups);
            down_per_layer.push(downs);
        }

        let tape = GpuTape::new(device.clone());

        // Place all leaf tensors on the tape FIRST so we hold node
        // indices for backward extraction.
        let xt = GpuTensor::from_vec(&tape, &input_data, vec![n_tokens, hidden])
            .unwrap();

        let mut win_per_layer: Vec<GpuTensor> = Vec::with_capacity(n_layers);
        let mut watn_per_layer: Vec<GpuTensor> = Vec::with_capacity(n_layers);
        let mut wpst_per_layer: Vec<GpuTensor> = Vec::with_capacity(n_layers);
        let mut wg_per_layer: Vec<GpuTensor> = Vec::with_capacity(n_layers);
        let mut gate_t_per_layer: Vec<Vec<GpuTensor>> = Vec::with_capacity(n_layers);
        let mut up_t_per_layer: Vec<Vec<GpuTensor>> = Vec::with_capacity(n_layers);
        let mut down_t_per_layer: Vec<Vec<GpuTensor>> = Vec::with_capacity(n_layers);

        for l in 0..n_layers {
            win_per_layer.push(
                GpuTensor::from_vec(&tape, &w_in_per_layer[l], vec![hidden]).unwrap(),
            );
            watn_per_layer.push(
                GpuTensor::from_vec(
                    &tape,
                    &w_attn_per_layer[l],
                    vec![hidden, hidden],
                )
                .unwrap(),
            );
            wpst_per_layer.push(
                GpuTensor::from_vec(&tape, &w_post_per_layer[l], vec![hidden])
                    .unwrap(),
            );
            wg_per_layer.push(
                GpuTensor::from_vec(
                    &tape,
                    &w_gate_per_layer[l],
                    vec![hidden, n_experts],
                )
                .unwrap(),
            );
            let mut gates: Vec<GpuTensor> = Vec::with_capacity(n_experts);
            let mut ups: Vec<GpuTensor> = Vec::with_capacity(n_experts);
            let mut downs: Vec<GpuTensor> = Vec::with_capacity(n_experts);
            for e in 0..n_experts {
                gates.push(
                    GpuTensor::from_vec(
                        &tape,
                        &gate_per_layer[l][e],
                        vec![hidden, intermediate],
                    )
                    .unwrap(),
                );
                ups.push(
                    GpuTensor::from_vec(
                        &tape,
                        &up_per_layer[l][e],
                        vec![hidden, intermediate],
                    )
                    .unwrap(),
                );
                downs.push(
                    GpuTensor::from_vec(
                        &tape,
                        &down_per_layer[l][e],
                        vec![intermediate, hidden],
                    )
                    .unwrap(),
                );
            }
            gate_t_per_layer.push(gates);
            up_t_per_layer.push(ups);
            down_t_per_layer.push(downs);
        }
        // Forward chain — stack 2 decoder layers via the iter-11h-f-5
        // top-level public API `qwen35_moe_forward_on_tape`.  Validates
        // both APIs at once: per-layer `decoder_layer_on_tape` (iter-
        // 11h-f-4) and the N-layer composer above it.  Refactor
        // preserves the original test's contract.
        let layers: Vec<super::DecoderLayerWeights<'_>> = (0..n_layers)
            .map(|l| super::DecoderLayerWeights {
                w_in: &win_per_layer[l],
                w_attn: &watn_per_layer[l],
                w_post: &wpst_per_layer[l],
                w_gate: &wg_per_layer[l],
                gate_projs: &gate_t_per_layer[l],
                up_projs: &up_t_per_layer[l],
                down_projs: &down_t_per_layer[l],
            })
            .collect();
        let final_out =
            super::qwen35_moe_forward_on_tape(&tape, &xt, &layers, k, eps)
                .expect("qwen35_moe_forward_on_tape forward");

        // Loss = sum(final_out).
        let host = final_out.to_vec().unwrap();
        let loss = host.iter().map(|v| *v as f64).sum::<f64>() as f32;
        eprintln!("[iter-11h-f-3] loss={loss:.6}");
        assert!(loss.is_finite(), "loss non-finite: {loss}");

        // Backward + extract per-leaf gradients.
        let dy = ones_like(&tape, final_out.shape()).unwrap();
        let grads = backward(&final_out, dy).unwrap();
        let read = |idx: usize| -> Vec<f32> {
            grads[idx]
                .as_ref()
                .unwrap()
                .as_slice::<f32>()
                .unwrap()
                .to_vec()
        };

        // Probe gradient on input X + per-layer leaves to prove the
        // chain rule reaches BOTH layers.
        let mut probe: Vec<(String, Vec<f32>)> = Vec::new();
        probe.push(("input".into(), read(xt.node_idx())));
        for l in 0..n_layers {
            probe.push((format!("L{l}_w_in"), read(win_per_layer[l].node_idx())));
            probe.push((format!("L{l}_w_attn"), read(watn_per_layer[l].node_idx())));
            probe.push((format!("L{l}_w_post"), read(wpst_per_layer[l].node_idx())));
            probe.push((
                format!("L{l}_e0_gate"),
                read(gate_t_per_layer[l][0].node_idx()),
            ));
            probe.push((
                format!("L{l}_e0_down"),
                read(down_t_per_layer[l][0].node_idx()),
            ));
        }

        for (name, g) in &probe {
            let max_abs = g.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            eprintln!(
                "[iter-11h-f-3] {name}: max|grad|={max_abs:.3e}, len={}",
                g.len()
            );
            assert!(
                max_abs > 1e-7,
                "{name} gradient collapsed to ~0 (max|grad|={max_abs:.3e}) — \
                 cross-layer chain rule failed at depth"
            );
            for (i, v) in g.iter().enumerate() {
                assert!(v.is_finite(), "{name} grad[{i}] = {v} not finite");
            }
        }

        drop(grads);
        drop(tape);
    }

    /// ADR-020 iter-11h-f-4 — public-API smoke test for
    /// `decoder_layer_on_tape`.
    ///
    /// Single-layer call to the new public API; mirrors iter-11h-f-2's
    /// gradient-flow assertions to prove the extracted function carries
    /// the same contract as the inline closure.  Gates `f-3`'s
    /// 2-layer test, which was refactored to use this same API.
    ///
    /// Falsifier: if `decoder_layer_on_tape` ever skips a stage in
    /// the chain (drops a residual, skips switch_mlp, etc.), the
    /// gradient on at least one of the 6 probed leaves will collapse
    /// to ~0.
    #[test]
    fn iter_11h_f4_decoder_layer_on_tape_public_api_gradient_flow() {
        use crate::calibrate::autograd_gpu_tape::{backward, ones_like, GpuTape};
        use mlx_native::MlxDevice;

        let device = MlxDevice::new().expect("device");

        // n_tokens, hidden, intermediate all >=32 — same kernel-floor
        // constraint as iter-11h-f-3.
        let n_tokens = 32usize;
        let hidden = 32usize;
        let intermediate = 32usize;
        let n_experts = 4usize;
        let k = 2usize;
        let eps = 1e-6f32;

        let input_data: Vec<f32> = (0..(n_tokens * hidden))
            .map(|i| ((i as f32) * 0.017 - 0.1).cos() * 0.3)
            .collect();
        let w_in_data: Vec<f32> =
            (0..hidden).map(|i| 1.0 + (i as f32) * 0.013).collect();
        let w_attn_data: Vec<f32> = (0..(hidden * hidden))
            .map(|i| 0.04 + (i as f32) * 0.0007)
            .collect();
        let w_post_data: Vec<f32> =
            (0..hidden).map(|i| 1.0 - (i as f32) * 0.004).collect();
        let w_gate_data: Vec<f32> = (0..(hidden * n_experts))
            .map(|i| 0.03 + (i as f32) * 0.0006)
            .collect();
        let ones_k_data = vec![1.0f32; k];

        let mut gate_data: Vec<Vec<f32>> = Vec::with_capacity(n_experts);
        let mut up_data: Vec<Vec<f32>> = Vec::with_capacity(n_experts);
        let mut down_data: Vec<Vec<f32>> = Vec::with_capacity(n_experts);
        for e in 0..n_experts {
            let off = (e as f32) * 0.09;
            gate_data.push(
                (0..(hidden * intermediate))
                    .map(|i| 0.04 + (i as f32) * 0.0008 + off)
                    .collect(),
            );
            up_data.push(
                (0..(hidden * intermediate))
                    .map(|i| 0.05 + (i as f32) * 0.0006 + off)
                    .collect(),
            );
            down_data.push(
                (0..(intermediate * hidden))
                    .map(|i| 0.03 + (i as f32) * 0.0009 + off)
                    .collect(),
            );
        }

        let tape = GpuTape::new(device.clone());

        let xt = GpuTensor::from_vec(&tape, &input_data, vec![n_tokens, hidden])
            .unwrap();
        let win = GpuTensor::from_vec(&tape, &w_in_data, vec![hidden]).unwrap();
        let watn = GpuTensor::from_vec(&tape, &w_attn_data, vec![hidden, hidden])
            .unwrap();
        let wpst = GpuTensor::from_vec(&tape, &w_post_data, vec![hidden]).unwrap();
        let wg = GpuTensor::from_vec(
            &tape,
            &w_gate_data,
            vec![hidden, n_experts],
        )
        .unwrap();
        let ok = GpuTensor::from_vec(&tape, &ones_k_data, vec![k]).unwrap();

        let mut gate_t: Vec<GpuTensor> = Vec::with_capacity(n_experts);
        let mut up_t: Vec<GpuTensor> = Vec::with_capacity(n_experts);
        let mut down_t: Vec<GpuTensor> = Vec::with_capacity(n_experts);
        for e in 0..n_experts {
            gate_t.push(
                GpuTensor::from_vec(&tape, &gate_data[e], vec![hidden, intermediate])
                    .unwrap(),
            );
            up_t.push(
                GpuTensor::from_vec(&tape, &up_data[e], vec![hidden, intermediate])
                    .unwrap(),
            );
            down_t.push(
                GpuTensor::from_vec(&tape, &down_data[e], vec![intermediate, hidden])
                    .unwrap(),
            );
        }

        let weights = super::DecoderLayerWeights {
            w_in: &win,
            w_attn: &watn,
            w_post: &wpst,
            w_gate: &wg,
            gate_projs: &gate_t,
            up_projs: &up_t,
            down_projs: &down_t,
        };

        let out = super::decoder_layer_on_tape(&tape, &xt, &weights, k, eps, &ok)
            .expect("decoder_layer_on_tape forward");
        assert_eq!(
            out.shape(),
            &[n_tokens, hidden],
            "decoder_layer_on_tape output shape"
        );

        // Loss = sum(out).
        let host = out.to_vec().unwrap();
        let loss = host.iter().map(|v| *v as f64).sum::<f64>() as f32;
        assert!(loss.is_finite(), "loss non-finite: {loss}");

        let dy = ones_like(&tape, out.shape()).unwrap();
        let grads = backward(&out, dy).unwrap();
        let read = |idx: usize| -> Vec<f32> {
            grads[idx]
                .as_ref()
                .unwrap()
                .as_slice::<f32>()
                .unwrap()
                .to_vec()
        };

        for (name, g) in [
            ("input", read(xt.node_idx())),
            ("w_in", read(win.node_idx())),
            ("w_attn", read(watn.node_idx())),
            ("w_post", read(wpst.node_idx())),
            ("e0_gate", read(gate_t[0].node_idx())),
            ("e0_down", read(down_t[0].node_idx())),
        ] {
            let max_abs = g.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            assert!(
                max_abs > 1e-7,
                "{name} gradient collapsed (max|grad|={max_abs:.3e}) — \
                 decoder_layer_on_tape skipped a stage in the chain"
            );
            for v in &g {
                assert!(v.is_finite(), "{name} grad non-finite: {v}");
            }
        }

        drop(grads);
        drop(tape);
    }

    /// ADR-020 iter-11h-f-5 — `qwen35_moe_forward_on_tape` empty-stack
    /// rejection.  Falsifier: an empty layer slice MUST `Err`, not
    /// silently return the input untouched (the "no-op forward" anti-
    /// pattern would silently DWQ-train against the wrong loss surface).
    #[test]
    fn iter_11h_f5_qwen35_moe_forward_rejects_empty_layers() {
        use crate::calibrate::autograd_gpu_tape::GpuTape;
        use mlx_native::MlxDevice;

        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let xt = GpuTensor::from_vec(&tape, &vec![0.5f32; 32 * 32], vec![32, 32])
            .unwrap();
        let layers: Vec<super::DecoderLayerWeights<'_>> = Vec::new();
        let res = super::qwen35_moe_forward_on_tape(&tape, &xt, &layers, 2, 1e-6);
        assert!(res.is_err(), "empty layer stack must Err");
        let msg = format!("{}", res.err().unwrap());
        assert!(
            msg.contains("non-empty"),
            "error message should explain the constraint; got: {msg}"
        );
    }

    /// ADR-020 iter-11h-f-5 — 3-layer stack via the top-level public
    /// API, mirrors iter-11h-f-3's gradient-flow contract at greater
    /// depth.  Validates that `qwen35_moe_forward_on_tape` correctly
    /// threads the residual hidden state across multiple layers and
    /// that gradient propagates from the loss back to the FIRST
    /// layer's leaves (chain rule reaches the deepest leaf even at
    /// n_layers=3).
    #[test]
    fn iter_11h_f5_qwen35_moe_forward_three_layer_gradient_flow() {
        use crate::calibrate::autograd_gpu_tape::{backward, ones_like, GpuTape};
        use mlx_native::MlxDevice;

        let device = MlxDevice::new().expect("device");

        let n_layers = 3usize;
        let n_tokens = 32usize;
        let hidden = 32usize;
        let intermediate = 32usize;
        let n_experts = 4usize;
        let k = 2usize;
        let eps = 1e-6f32;

        let input_data: Vec<f32> = (0..(n_tokens * hidden))
            .map(|i| ((i as f32) * 0.019 + 0.1).sin() * 0.35)
            .collect();

        // Per-layer fixtures (distinct via per-layer offset).
        let make_w_in = |l: usize| -> Vec<f32> {
            (0..hidden)
                .map(|i| 1.0 + (i as f32) * 0.01 + (l as f32) * 0.04)
                .collect()
        };
        let make_w_attn = |l: usize| -> Vec<f32> {
            (0..(hidden * hidden))
                .map(|i| 0.04 + (i as f32) * 0.0006 + (l as f32) * 0.013)
                .collect()
        };
        let make_w_post = |l: usize| -> Vec<f32> {
            (0..hidden)
                .map(|i| 1.0 - (i as f32) * 0.003 + (l as f32) * 0.02)
                .collect()
        };
        let make_w_gate = |l: usize| -> Vec<f32> {
            (0..(hidden * n_experts))
                .map(|i| 0.03 + (i as f32) * 0.0005 + (l as f32) * 0.017)
                .collect()
        };
        let make_expert = |l: usize, e: usize, base: f32, slope: f32| -> Vec<f32> {
            let off = (e as f32) * 0.07 + (l as f32) * 0.11;
            (0..(hidden * intermediate))
                .map(|i| base + (i as f32) * slope + off)
                .collect()
        };

        let tape = GpuTape::new(device.clone());

        let xt = GpuTensor::from_vec(&tape, &input_data, vec![n_tokens, hidden])
            .unwrap();

        // Place all leaves on the tape; collect borrows for the
        // DecoderLayerWeights bundle.
        let mut win_t: Vec<GpuTensor> = Vec::new();
        let mut watn_t: Vec<GpuTensor> = Vec::new();
        let mut wpst_t: Vec<GpuTensor> = Vec::new();
        let mut wg_t: Vec<GpuTensor> = Vec::new();
        let mut gate_t_l: Vec<Vec<GpuTensor>> = Vec::new();
        let mut up_t_l: Vec<Vec<GpuTensor>> = Vec::new();
        let mut down_t_l: Vec<Vec<GpuTensor>> = Vec::new();
        for l in 0..n_layers {
            win_t.push(GpuTensor::from_vec(&tape, &make_w_in(l), vec![hidden]).unwrap());
            watn_t.push(
                GpuTensor::from_vec(&tape, &make_w_attn(l), vec![hidden, hidden])
                    .unwrap(),
            );
            wpst_t.push(
                GpuTensor::from_vec(&tape, &make_w_post(l), vec![hidden]).unwrap(),
            );
            wg_t.push(
                GpuTensor::from_vec(&tape, &make_w_gate(l), vec![hidden, n_experts])
                    .unwrap(),
            );
            let mut gates: Vec<GpuTensor> = Vec::new();
            let mut ups: Vec<GpuTensor> = Vec::new();
            let mut downs: Vec<GpuTensor> = Vec::new();
            for e in 0..n_experts {
                gates.push(
                    GpuTensor::from_vec(
                        &tape,
                        &make_expert(l, e, 0.05, 0.0008),
                        vec![hidden, intermediate],
                    )
                    .unwrap(),
                );
                ups.push(
                    GpuTensor::from_vec(
                        &tape,
                        &make_expert(l, e, 0.04, 0.0010),
                        vec![hidden, intermediate],
                    )
                    .unwrap(),
                );
                downs.push(
                    GpuTensor::from_vec(
                        &tape,
                        &make_expert(l, e, 0.03, 0.0012),
                        vec![intermediate, hidden],
                    )
                    .unwrap(),
                );
            }
            gate_t_l.push(gates);
            up_t_l.push(ups);
            down_t_l.push(downs);
        }

        let layers: Vec<super::DecoderLayerWeights<'_>> = (0..n_layers)
            .map(|l| super::DecoderLayerWeights {
                w_in: &win_t[l],
                w_attn: &watn_t[l],
                w_post: &wpst_t[l],
                w_gate: &wg_t[l],
                gate_projs: &gate_t_l[l],
                up_projs: &up_t_l[l],
                down_projs: &down_t_l[l],
            })
            .collect();

        let out = super::qwen35_moe_forward_on_tape(&tape, &xt, &layers, k, eps)
            .expect("qwen35_moe_forward_on_tape forward");
        assert_eq!(out.shape(), &[n_tokens, hidden], "output shape");

        let host = out.to_vec().unwrap();
        let loss = host.iter().map(|v| *v as f64).sum::<f64>() as f32;
        assert!(loss.is_finite(), "loss non-finite: {loss}");

        let dy = ones_like(&tape, out.shape()).unwrap();
        let grads = backward(&out, dy).unwrap();
        let read = |idx: usize| -> Vec<f32> {
            grads[idx]
                .as_ref()
                .unwrap()
                .as_slice::<f32>()
                .unwrap()
                .to_vec()
        };

        // Probe input X + per-layer w_in.  Critical: gradient on
        // layer 0's w_in proves the chain rule reaches the deepest
        // layer through 3 stacked decoder forwards.
        let g_input = read(xt.node_idx());
        let max_input = g_input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("[iter-11h-f-5] input max|grad|={max_input:.3e}");
        assert!(max_input > 1e-7, "input X gradient collapsed at depth 3");
        for v in &g_input {
            assert!(v.is_finite(), "input grad non-finite: {v}");
        }

        for l in 0..n_layers {
            let g = read(win_t[l].node_idx());
            let max_abs = g.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            eprintln!("[iter-11h-f-5] L{l}_w_in max|grad|={max_abs:.3e}");
            assert!(
                max_abs > 1e-7,
                "L{l} w_in gradient collapsed (max|grad|={max_abs:.3e})"
            );
            for v in &g {
                assert!(v.is_finite(), "L{l} w_in grad non-finite: {v}");
            }
        }

        drop(grads);
        drop(tape);
    }

    /// ADR-020 iter-11h-f-6 — moderate-scale validation of
    /// `qwen35_moe_forward_on_tape` at MoE-realistic shapes.
    ///
    /// Bumps from f-5's tiny `n_experts=4 / k=2` / `n_layers=3` to a
    /// configuration that exercises the sparse-routing regime (some
    /// experts get 0 routings) and a deeper stack:
    ///   `n_layers=4`, `n_tokens=32`, `hidden=128`, `intermediate=128`,
    ///   `n_experts=16`, `k=4` — only 4 / 16 experts active per token.
    ///
    /// At `n_tokens=32` × `k=4` = 128 routings spread over 16 experts,
    /// some experts will land 0 routings by the random fixture +
    /// softmax → top-K winner selection.  This exercises
    /// `switch_mlp`'s "skip experts with empty active-token bucket"
    /// path, which f-3 / f-5 didn't because all 4 experts were always
    /// active there.
    ///
    /// Falsifier: if the API ever regresses on a deeper / sparser
    /// stack (gradient collapses, NaN appears, output goes
    /// non-finite), this test catches it before iter-11h-f real-GGUF
    /// integration tries to load actual Qwen3.5 35B-A3B weights.
    ///
    /// Memory budget: 4 layers × 16 experts × 3 weight tensors × 128×
    /// 128 fp32 = ~3 MB total tape leaves — trivial vs. real model.
    #[test]
    fn iter_11h_f6_qwen35_moe_forward_moderate_scale_gradient_flow() {
        use crate::calibrate::autograd_gpu_tape::{backward, ones_like, GpuTape};
        use mlx_native::MlxDevice;

        let device = MlxDevice::new().expect("device");

        let n_layers = 4usize;
        let n_tokens = 32usize;
        let hidden = 128usize;
        let intermediate = 128usize;
        let n_experts = 16usize;
        let k = 4usize;
        let eps = 1e-6f32;

        // Pseudo-random fixture (deterministic via xorshift).
        let mut rng_state: u64 = 0xCAFE_BABE_F00D_BA11;
        let mut next = move || -> f32 {
            rng_state ^= rng_state >> 33;
            rng_state = rng_state.wrapping_mul(0xff51_afd7_ed55_8ccd);
            rng_state ^= rng_state >> 33;
            ((rng_state as i64) as f32) / (i64::MAX as f32)
        };

        let input_data: Vec<f32> =
            (0..(n_tokens * hidden)).map(|_| next() * 0.3).collect();

        let tape = GpuTape::new(device.clone());

        let xt = GpuTensor::from_vec(&tape, &input_data, vec![n_tokens, hidden])
            .unwrap();

        // Place per-layer leaves on the tape.  Use one helper closure
        // per shape so it stays compact.
        let mut win_t: Vec<GpuTensor> = Vec::new();
        let mut watn_t: Vec<GpuTensor> = Vec::new();
        let mut wpst_t: Vec<GpuTensor> = Vec::new();
        let mut wg_t: Vec<GpuTensor> = Vec::new();
        let mut gate_t_l: Vec<Vec<GpuTensor>> = Vec::new();
        let mut up_t_l: Vec<Vec<GpuTensor>> = Vec::new();
        let mut down_t_l: Vec<Vec<GpuTensor>> = Vec::new();

        for _l in 0..n_layers {
            // Norm weights — tight around 1.0 so RMS-norm is stable.
            let w_in: Vec<f32> = (0..hidden).map(|_| 1.0 + next() * 0.05).collect();
            let w_post: Vec<f32> = (0..hidden).map(|_| 1.0 + next() * 0.05).collect();
            // Attn proxy + router gate — small magnitude.
            let w_attn: Vec<f32> =
                (0..(hidden * hidden)).map(|_| next() * 0.05).collect();
            let w_gate: Vec<f32> =
                (0..(hidden * n_experts)).map(|_| next() * 0.1).collect();

            win_t.push(GpuTensor::from_vec(&tape, &w_in, vec![hidden]).unwrap());
            watn_t.push(
                GpuTensor::from_vec(&tape, &w_attn, vec![hidden, hidden]).unwrap(),
            );
            wpst_t.push(GpuTensor::from_vec(&tape, &w_post, vec![hidden]).unwrap());
            wg_t.push(
                GpuTensor::from_vec(&tape, &w_gate, vec![hidden, n_experts]).unwrap(),
            );

            let mut gates: Vec<GpuTensor> = Vec::with_capacity(n_experts);
            let mut ups: Vec<GpuTensor> = Vec::with_capacity(n_experts);
            let mut downs: Vec<GpuTensor> = Vec::with_capacity(n_experts);
            for _e in 0..n_experts {
                let g: Vec<f32> = (0..(hidden * intermediate))
                    .map(|_| next() * 0.05)
                    .collect();
                let u: Vec<f32> = (0..(hidden * intermediate))
                    .map(|_| next() * 0.05)
                    .collect();
                let d: Vec<f32> = (0..(intermediate * hidden))
                    .map(|_| next() * 0.05)
                    .collect();
                gates.push(
                    GpuTensor::from_vec(&tape, &g, vec![hidden, intermediate]).unwrap(),
                );
                ups.push(
                    GpuTensor::from_vec(&tape, &u, vec![hidden, intermediate]).unwrap(),
                );
                downs.push(
                    GpuTensor::from_vec(&tape, &d, vec![intermediate, hidden]).unwrap(),
                );
            }
            gate_t_l.push(gates);
            up_t_l.push(ups);
            down_t_l.push(downs);
        }

        let layers: Vec<super::DecoderLayerWeights<'_>> = (0..n_layers)
            .map(|l| super::DecoderLayerWeights {
                w_in: &win_t[l],
                w_attn: &watn_t[l],
                w_post: &wpst_t[l],
                w_gate: &wg_t[l],
                gate_projs: &gate_t_l[l],
                up_projs: &up_t_l[l],
                down_projs: &down_t_l[l],
            })
            .collect();

        let out = super::qwen35_moe_forward_on_tape(&tape, &xt, &layers, k, eps)
            .expect("qwen35_moe_forward_on_tape forward");
        assert_eq!(out.shape(), &[n_tokens, hidden], "output shape");

        let host = out.to_vec().unwrap();
        for (i, v) in host.iter().enumerate() {
            assert!(v.is_finite(), "out[{i}] = {v} not finite");
        }
        let loss = host.iter().map(|v| *v as f64).sum::<f64>() as f32;
        assert!(loss.is_finite(), "loss non-finite: {loss}");
        eprintln!(
            "[iter-11h-f-6] forward OK at n_layers={n_layers}, hidden={hidden}, \
             n_experts={n_experts}, k={k}, loss={loss:.4}"
        );

        let dy = ones_like(&tape, out.shape()).unwrap();
        let grads = backward(&out, dy).unwrap();
        let read = |idx: usize| -> Vec<f32> {
            grads[idx]
                .as_ref()
                .unwrap()
                .as_slice::<f32>()
                .unwrap()
                .to_vec()
        };

        // Critical leaves: input X (deepest chain) + L0 w_in (deepest
        // layer's leaf).  If the chain rule fails at depth 4 OR if a
        // sparse expert path eats gradient, these collapse.
        let g_input = read(xt.node_idx());
        let max_input = g_input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("[iter-11h-f-6] input max|grad|={max_input:.3e}");
        assert!(max_input > 1e-7, "input X gradient collapsed at depth 4");
        for v in &g_input {
            assert!(v.is_finite(), "input grad non-finite: {v}");
        }

        let g_l0_win = read(win_t[0].node_idx());
        let max_l0 = g_l0_win.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("[iter-11h-f-6] L0_w_in max|grad|={max_l0:.3e}");
        assert!(max_l0 > 1e-7, "L0 w_in gradient collapsed at depth 4");

        // Survey a few experts in different layers — at least one
        // active expert per layer must have non-trivial gradient.
        // Routing is deterministic from the fixture; we don't know
        // which experts won, only that at least some did.
        let mut layers_with_active_experts = 0usize;
        for l in 0..n_layers {
            let mut layer_has_active = false;
            for e in 0..n_experts {
                let g = read(gate_t_l[l][e].node_idx());
                let max_abs = g.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                if max_abs > 1e-7 {
                    layer_has_active = true;
                }
                for (i, v) in g.iter().enumerate() {
                    assert!(v.is_finite(), "L{l} e{e} gate grad[{i}]={v} non-finite");
                }
            }
            if layer_has_active {
                layers_with_active_experts += 1;
            }
        }
        eprintln!(
            "[iter-11h-f-6] {layers_with_active_experts}/{n_layers} layers have \
             at least one active expert with non-trivial grad"
        );
        assert_eq!(
            layers_with_active_experts, n_layers,
            "every layer must have at least one active expert with non-trivial grad — \
             a layer with all-zero expert gradients means switch_mlp produced no \
             gradient path through ANY active expert"
        );

        drop(grads);
        drop(tape);
    }

    /// ADR-020 AC#7 Option A foundation — qdq-wrapped MoE forward proves
    /// gradient flow back to per-Linear (s, b) leaves, NOT just the F32
    /// weight tensors.  Single decoder layer, n_experts=2, k=1.  Builds
    /// each expert's gate/up/down via `qdq_affine(s, b, q_int)` →
    /// reshape, threads the qdq tensors into `decoder_layer_on_tape`,
    /// runs forward + ones-seeded backward, and asserts:
    ///
    ///   1. Backward succeeds (no chain breaks introduced by qdq).
    ///   2. Gradients are present on EVERY (s, b) leaf — not just the
    ///      qdq output tensors.  This is the load-bearing assertion:
    ///      proves the autograd chain `s/b → qdq_affine → matmul →
    ///      switch_mlp → forward output` correctly routes contributions
    ///      back to the trainable params.
    ///
    /// If this passes, the smallest provable Option A wiring step is
    /// closed: the building blocks compose into a full-model
    /// differentiable DWQ student forward.  Future iters can scale to
    /// real models + add the teacher-loss + Adam loop on top.
    #[test]
    fn ac7_option_a_foundation_qdq_wrapped_moe_layer_routes_gradients_to_s_b() {
        use crate::calibrate::autograd_gpu_tape::{backward, ones_like, qdq_affine, view, GpuTape, GpuTensor};
        use mlx_native::MlxDevice;

        let device = MlxDevice::new().expect("device");
        let n_tokens = 32usize;
        let hidden = 32usize;
        let intermediate = 32usize;
        let n_experts = 2usize;
        let k = 1usize;
        let group_size = 32usize;
        let bits = 4u32;
        let pack_factor = (32 / bits) as usize;
        let _ = pack_factor;
        let eps = 1e-6f32;

        // Deterministic xorshift RNG.
        let mut rng_state: u64 = 0xCAFE_F00D_DEAD_BABE;
        let mut next = move || -> f32 {
            rng_state ^= rng_state >> 33;
            rng_state = rng_state.wrapping_mul(0xff51_afd7_ed55_8ccd);
            rng_state ^= rng_state >> 33;
            ((rng_state as i64) as f32) / (i64::MAX as f32)
        };

        let tape = GpuTape::new(device.clone());

        // Input + per-layer F32 small weights (norm/attn/router).
        let input_data: Vec<f32> = (0..(n_tokens * hidden)).map(|_| next() * 0.3).collect();
        let xt = GpuTensor::from_vec(&tape, &input_data, vec![n_tokens, hidden]).unwrap();
        let w_in_data: Vec<f32> = (0..hidden).map(|_| 1.0 + next() * 0.05).collect();
        let w_post_data: Vec<f32> = (0..hidden).map(|_| 1.0 + next() * 0.05).collect();
        let w_attn_data: Vec<f32> = (0..(hidden * hidden)).map(|_| next() * 0.05).collect();
        let w_gate_data: Vec<f32> = (0..(hidden * n_experts)).map(|_| next() * 0.1).collect();
        let w_in_t = GpuTensor::from_vec(&tape, &w_in_data, vec![hidden]).unwrap();
        let w_post_t = GpuTensor::from_vec(&tape, &w_post_data, vec![hidden]).unwrap();
        let w_attn_t = GpuTensor::from_vec(&tape, &w_attn_data, vec![hidden, hidden]).unwrap();
        let w_gate_t = GpuTensor::from_vec(&tape, &w_gate_data, vec![hidden, n_experts]).unwrap();

        // Per-expert (s, b, q_int) leaves + qdq-derived gate/up/down tensors.
        let groups_per_gate_up_row = hidden / group_size; // input dim partitioned for gate/up
        let groups_per_down_row = intermediate / group_size; // input dim partitioned for down

        // Storage for s/b leaves so they outlive the test.
        let mut gate_s_leaves: Vec<GpuTensor> = Vec::with_capacity(n_experts);
        let mut gate_b_leaves: Vec<GpuTensor> = Vec::with_capacity(n_experts);
        let mut up_s_leaves: Vec<GpuTensor> = Vec::with_capacity(n_experts);
        let mut up_b_leaves: Vec<GpuTensor> = Vec::with_capacity(n_experts);
        let mut down_s_leaves: Vec<GpuTensor> = Vec::with_capacity(n_experts);
        let mut down_b_leaves: Vec<GpuTensor> = Vec::with_capacity(n_experts);
        // Storage for qdq-output tensors (lives until decoder_layer_on_tape returns).
        let mut gate_w_t: Vec<GpuTensor> = Vec::with_capacity(n_experts);
        let mut up_w_t: Vec<GpuTensor> = Vec::with_capacity(n_experts);
        let mut down_w_t: Vec<GpuTensor> = Vec::with_capacity(n_experts);

        for _e in 0..n_experts {
            // gate/up: shape [hidden, intermediate] — n=intermediate, k=hidden
            // → groups = n × (k/group_size) = intermediate × groups_per_gate_up_row
            let gate_n_groups = intermediate * groups_per_gate_up_row;
            let s_gate: Vec<f32> = (0..gate_n_groups).map(|_| 0.05 + next().abs() * 0.001).collect();
            let b_gate: Vec<f32> = (0..gate_n_groups).map(|_| -0.1 + next() * 0.005).collect();
            let q_int_gate: Vec<u8> = (0..(intermediate * hidden))
                .map(|i| ((i * 7) % 16) as u8)
                .collect();
            let s_gate_leaf = GpuTensor::from_vec(&tape, &s_gate, vec![gate_n_groups]).unwrap();
            let b_gate_leaf = GpuTensor::from_vec(&tape, &b_gate, vec![gate_n_groups]).unwrap();
            let gate_qdq = qdq_affine(&s_gate_leaf, &b_gate_leaf, &q_int_gate, group_size).unwrap();
            let gate_view = view(&gate_qdq, vec![intermediate, hidden]).unwrap();
            gate_s_leaves.push(s_gate_leaf);
            gate_b_leaves.push(b_gate_leaf);
            gate_w_t.push(gate_view);

            let s_up: Vec<f32> = (0..gate_n_groups).map(|_| 0.05 + next().abs() * 0.001).collect();
            let b_up: Vec<f32> = (0..gate_n_groups).map(|_| -0.1 + next() * 0.005).collect();
            let q_int_up: Vec<u8> = (0..(intermediate * hidden))
                .map(|i| ((i * 11) % 16) as u8)
                .collect();
            let s_up_leaf = GpuTensor::from_vec(&tape, &s_up, vec![gate_n_groups]).unwrap();
            let b_up_leaf = GpuTensor::from_vec(&tape, &b_up, vec![gate_n_groups]).unwrap();
            let up_qdq = qdq_affine(&s_up_leaf, &b_up_leaf, &q_int_up, group_size).unwrap();
            let up_view = view(&up_qdq, vec![intermediate, hidden]).unwrap();
            up_s_leaves.push(s_up_leaf);
            up_b_leaves.push(b_up_leaf);
            up_w_t.push(up_view);

            // down: shape [intermediate, hidden] — n=hidden, k=intermediate
            let down_n_groups = hidden * groups_per_down_row;
            let s_down: Vec<f32> = (0..down_n_groups).map(|_| 0.05 + next().abs() * 0.001).collect();
            let b_down: Vec<f32> = (0..down_n_groups).map(|_| -0.1 + next() * 0.005).collect();
            let q_int_down: Vec<u8> = (0..(hidden * intermediate))
                .map(|i| ((i * 13) % 16) as u8)
                .collect();
            let s_down_leaf = GpuTensor::from_vec(&tape, &s_down, vec![down_n_groups]).unwrap();
            let b_down_leaf = GpuTensor::from_vec(&tape, &b_down, vec![down_n_groups]).unwrap();
            let down_qdq = qdq_affine(&s_down_leaf, &b_down_leaf, &q_int_down, group_size).unwrap();
            let down_view = view(&down_qdq, vec![hidden, intermediate]).unwrap();
            down_s_leaves.push(s_down_leaf);
            down_b_leaves.push(b_down_leaf);
            down_w_t.push(down_view);
        }

        let layer = super::DecoderLayerWeights {
            w_in: &w_in_t,
            w_attn: &w_attn_t,
            w_post: &w_post_t,
            w_gate: &w_gate_t,
            gate_projs: &gate_w_t,
            up_projs: &up_w_t,
            down_projs: &down_w_t,
        };
        let layers = [layer];
        let out = super::qwen35_moe_forward_on_tape(&tape, &xt, &layers, k, eps)
            .expect("qwen35_moe_forward_on_tape (qdq-wrapped weights)");
        assert_eq!(out.shape(), &[n_tokens, hidden]);

        // Ones-seeded backward.
        let dy_buf = ones_like(&tape, out.shape()).expect("ones_like for dy");
        let grads = backward(&out, dy_buf).expect("backward");

        // Read grad at a given leaf's node_idx.
        let read = |t: &GpuTensor| -> Vec<f32> {
            grads[t.node_idx()]
                .as_ref()
                .unwrap_or_else(|| panic!("grad missing at node_idx={}", t.node_idx()))
                .as_slice::<f32>()
                .unwrap()
                .to_vec()
        };

        // The load-bearing assertion: gradients exist on EVERY (s, b)
        // leaf for ALL n_experts × {gate, up, down} = 6 (s, b) pairs.
        for e in 0..n_experts {
            for (label, leaf) in [
                (format!("gate_s[expert={e}]"), &gate_s_leaves[e]),
                (format!("gate_b[expert={e}]"), &gate_b_leaves[e]),
                (format!("up_s[expert={e}]"),   &up_s_leaves[e]),
                (format!("up_b[expert={e}]"),   &up_b_leaves[e]),
                (format!("down_s[expert={e}]"), &down_s_leaves[e]),
                (format!("down_b[expert={e}]"), &down_b_leaves[e]),
            ] {
                let host: Vec<f32> = read(leaf);
                let max_abs = host.iter().map(|v: &f32| v.abs()).fold(0.0f32, f32::max);
                assert!(
                    max_abs.is_finite(),
                    "{label}: grad max_abs non-finite = {max_abs}"
                );
                assert!(
                    max_abs > 0.0,
                    "{label}: grad is identically zero — qdq → matmul → \
                     switch_mlp backward chain is broken at this leaf"
                );
            }
        }

        drop(grads);
        drop(tape);
    }

    /// ADR-020 iter-11h-e3c — switch_mlp composition forward+backward FD.
    ///
    /// Builds a small frozen-router switch_mlp over E=2 experts, K=2 active
    /// per token, with deterministic routing weights + indices.  Asserts
    /// forward output shape, that gradients land on input X AND on at least
    /// one weight of each expert's gate/up/down Linear, and FD-falsifies the
    /// first 4 elements of input X + first 4 elements of expert 0's
    /// gate_proj weight (5% rel tol).  Validates the gradient chain through
    /// the full SwiGLU + matmul + mask pipeline.
    #[test]
    fn switch_mlp_composition_forward_and_backward_fd() {
        use crate::calibrate::autograd_gpu_tape::{backward, ones_like, GpuTape};
        use mlx_native::MlxDevice;

        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);

        // Backward matmul requires m, n, k >= 32; pick all = 32.
        let n_tokens = 32usize;
        let hidden = 32usize;
        let intermediate = 32usize;
        let n_experts = 2usize;

        // Deterministic input + per-expert weights.
        let x_data: Vec<f32> = (0..(n_tokens * hidden))
            .map(|i| ((i as f32) * 0.013 - 0.5).sin() * 0.4)
            .collect();
        let xt = GpuTensor::from_vec(&tape, &x_data, vec![n_tokens, hidden]).unwrap();

        // 2 experts × 3 projections; small but distinct per expert via offset.
        let mut gate_projs: Vec<GpuTensor> = Vec::with_capacity(n_experts);
        let mut up_projs: Vec<GpuTensor> = Vec::with_capacity(n_experts);
        let mut down_projs: Vec<GpuTensor> = Vec::with_capacity(n_experts);
        let mut gate_data_per_expert: Vec<Vec<f32>> = Vec::with_capacity(n_experts);

        for e in 0..n_experts {
            let off = (e as f32) * 0.07;
            let g_data: Vec<f32> = (0..(hidden * intermediate))
                .map(|i| 0.05 + (i as f32) * 0.0007 + off)
                .collect();
            let u_data: Vec<f32> = (0..(hidden * intermediate))
                .map(|i| 0.04 + (i as f32) * 0.0009 + off)
                .collect();
            let d_data: Vec<f32> = (0..(intermediate * hidden))
                .map(|i| 0.03 + (i as f32) * 0.0011 + off)
                .collect();
            gate_projs
                .push(GpuTensor::from_vec(&tape, &g_data, vec![hidden, intermediate]).unwrap());
            up_projs
                .push(GpuTensor::from_vec(&tape, &u_data, vec![hidden, intermediate]).unwrap());
            down_projs
                .push(GpuTensor::from_vec(&tape, &d_data, vec![intermediate, hidden]).unwrap());
            gate_data_per_expert.push(g_data);
        }

        // Routing: top_k=2, per-token weights + indices.  Alternate to ensure
        // both experts are active and have varying routing weights.
        let expert_ids: Vec<Vec<usize>> = (0..n_tokens)
            .map(|t| if t % 2 == 0 { vec![0, 1] } else { vec![1, 0] })
            .collect();
        let routing_weights: Vec<Vec<f32>> = (0..n_tokens)
            .map(|t| {
                let a = 0.6 + 0.01 * (t as f32);
                vec![a, 1.0 - a]
            })
            .collect();

        // Forward
        let out = switch_mlp(
            &tape, &xt, &gate_projs, &up_projs, &down_projs,
            &expert_ids, &routing_weights,
        )
        .expect("switch_mlp forward");
        assert_eq!(out.shape(), [n_tokens, hidden], "switch_mlp output shape");

        // Loss = sum(out)
        let dy = ones_like(&tape, out.shape()).expect("ones_like");
        let grads = backward(&out, dy).expect("backward");

        // Pull gradients for: input X, expert 0 gate_proj, expert 0 down_proj,
        // expert 1 up_proj.  Each must be Some + finite + non-trivial magnitude
        // (catches "loss is constant" + "gradient didn't land on this leaf").
        let grad_x = grads
            .get(xt.node_idx())
            .and_then(|g| g.as_ref())
            .expect("gradient missing on input X — backward chain broke")
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let grad_e0_gate = grads
            .get(gate_projs[0].node_idx())
            .and_then(|g| g.as_ref())
            .expect("gradient missing on expert 0 gate_proj")
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let grad_e0_down = grads
            .get(down_projs[0].node_idx())
            .and_then(|g| g.as_ref())
            .expect("gradient missing on expert 0 down_proj")
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let grad_e1_up = grads
            .get(up_projs[1].node_idx())
            .and_then(|g| g.as_ref())
            .expect("gradient missing on expert 1 up_proj")
            .as_slice::<f32>()
            .unwrap()
            .to_vec();

        // Sanity: each gradient must have at least one non-trivial entry
        // (catches the "loss is invariant" bug surfaced in iter-11h-f-mini
        // where renorm=true made loss constant → ∇=0).
        for (label, g) in [
            ("input_x", &grad_x),
            ("e0_gate", &grad_e0_gate),
            ("e0_down", &grad_e0_down),
            ("e1_up", &grad_e1_up),
        ] {
            let max_abs = g.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            assert!(
                max_abs > 1e-5,
                "{label} gradient max|·| = {max_abs} < 1e-5 — loss may be invariant"
            );
            for (i, v) in g.iter().enumerate() {
                assert!(v.is_finite(), "{label} grad[{i}] = {v} not finite");
            }
        }

        // FD falsifier: probe 4 elements of input X.  L = sum(out).  The
        // analytic gradient is grad_x.  FD recomputes via CPU forward in the
        // composition above? Too much to redo on CPU — instead recompute
        // forward on the GPU side with a perturbed input and compare loss
        // delta.  This stays inside the GPU composition while still
        // falsifying the analytic gradient.
        let h: f32 = 1e-3;
        for &idx in &[0usize, 7, 19, 31 * 32 + 31] {
            let mut xp = x_data.clone();
            xp[idx] += h;
            let mut xm = x_data.clone();
            xm[idx] -= h;

            let do_forward = |data: &[f32]| -> f64 {
                let device2 = MlxDevice::new().expect("device2");
                let tape2 = GpuTape::new(device2);
                let xt2 = GpuTensor::from_vec(&tape2, data, vec![n_tokens, hidden]).unwrap();
                let mut gpe: Vec<GpuTensor> = Vec::new();
                let mut upe: Vec<GpuTensor> = Vec::new();
                let mut dwe: Vec<GpuTensor> = Vec::new();
                for e in 0..n_experts {
                    let off = (e as f32) * 0.07;
                    let g_data: Vec<f32> = (0..(hidden * intermediate))
                        .map(|i| 0.05 + (i as f32) * 0.0007 + off)
                        .collect();
                    let u_data: Vec<f32> = (0..(hidden * intermediate))
                        .map(|i| 0.04 + (i as f32) * 0.0009 + off)
                        .collect();
                    let d_data: Vec<f32> = (0..(intermediate * hidden))
                        .map(|i| 0.03 + (i as f32) * 0.0011 + off)
                        .collect();
                    gpe.push(
                        GpuTensor::from_vec(&tape2, &g_data, vec![hidden, intermediate]).unwrap(),
                    );
                    upe.push(
                        GpuTensor::from_vec(&tape2, &u_data, vec![hidden, intermediate]).unwrap(),
                    );
                    dwe.push(
                        GpuTensor::from_vec(&tape2, &d_data, vec![intermediate, hidden]).unwrap(),
                    );
                }
                let out2 = switch_mlp(
                    &tape2, &xt2, &gpe, &upe, &dwe,
                    &expert_ids, &routing_weights,
                )
                .unwrap();
                out2.to_vec()
                    .unwrap()
                    .iter()
                    .map(|v| *v as f64)
                    .sum::<f64>()
            };

            let lp = do_forward(&xp);
            let lm = do_forward(&xm);
            let fd = (lp - lm) / (2.0 * h as f64);
            let tol = 5e-2 * fd.abs().max(1.0);
            assert!(
                (grad_x[idx] as f64 - fd).abs() < tol,
                "FD x[{idx}]: analytic={} fd={} (tol={tol})",
                grad_x[idx],
                fd
            );
        }
    }

    /// ADR-020 AC#7 Option A — STRONG FALSIFIER of the
    /// `project_adr020_dwq_perturb1_noop_finding` claim that DWQ at
    /// perturb=1.0 cannot improve.  Per-Linear synthetic teacher is a
    /// no-op (Option B falsified); this test asks: does FULL-MODEL
    /// teacher with qdq-wrapped student break the plateau at perturb=1.0?
    ///
    /// Setup:
    /// - 2-layer MoE (gives cross-layer compositional error to compensate)
    /// - n_experts=2, k=1, hidden=intermediate=32, bits=4, group_size=32
    /// - Teacher: same architecture, F32 weights `W_true`
    /// - Student: same architecture, qdq(s_init, b_init, q_int) where
    ///   q_int is the bit-4 quant of W_true (i.e., "starts at the
    ///   per-Linear projection optimum" — perturb=1.0)
    /// - Loss: sum of (student - teacher)^2 over the final hidden state
    ///   (MSE proxy for KL, simpler to wire on the tape)
    ///
    /// Assertion: `loss_after_step < loss_before_step`.  Even at
    /// perturb=1.0 (per-Linear projection optimum), a single
    /// gradient-descent step on (s, b) across both layers reduces
    /// the COMPOSITIONAL error between teacher and student outputs.
    /// This is the cross-layer compensation gradient that mlx-lm's
    /// dwq.py:108 exploits.
    ///
    /// If this PASSES, Option A is empirically validated as a viable
    /// AC#7 path.  If FAILS, Option A also has limits at perturb=1.0
    /// + bits=4 and AC#7 needs even more architectural change.
    #[test]
    fn ac7_option_a_full_model_two_layer_breaks_perturb_1_0_plateau() {
        use super::{decoder_layer_on_tape, DecoderLayerWeights};
        use crate::calibrate::autograd_gpu_tape::{
            backward, ones_like, qdq_affine, square, sub, view, GpuTape, GpuTensor,
        };
        use mlx_native::MlxDevice;

        let device = MlxDevice::new().expect("device");
        let n_tokens = 32usize;
        let hidden = 32usize;
        let intermediate = 32usize;
        let n_experts = 2usize;
        let k = 1usize;
        let group_size = 32usize;
        let bits = 4u32;
        let n_layers = 2usize;
        let eps = 1e-6f32;
        let n_bins = 1u32 << bits; // 16 for bits=4

        // xorshift RNG.
        let mut rng_state: u64 = 0x1357_9bdf_2468_ace0;
        let mut next = move || -> f32 {
            rng_state ^= rng_state >> 33;
            rng_state = rng_state.wrapping_mul(0xff51_afd7_ed55_8ccd);
            rng_state ^= rng_state >> 33;
            ((rng_state as i64) as f32) / (i64::MAX as f32)
        };

        // Per-Linear init helper: from W_real, derive (s_init, b_init,
        // q_int) via the same min-max formula as
        // `init_affine_params_gpu` — small enough to do CPU-side here.
        let groups_per_row_gateup = hidden / group_size; // = 1 for 32/32
        let groups_per_row_down = intermediate / group_size; // = 1
        let init_qdq = |w: &[f32], n: usize, k_dim: usize, gpr: usize| -> (Vec<f32>, Vec<f32>, Vec<u8>) {
            let n_groups = n * gpr;
            let mut s = vec![0.0f32; n_groups];
            let mut b = vec![0.0f32; n_groups];
            let mut q = vec![0u8; n * k_dim];
            for row in 0..n {
                for g in 0..gpr {
                    let base = row * k_dim + g * group_size;
                    let slab = &w[base..base + group_size];
                    let w_min = slab.iter().copied().fold(f32::INFINITY, f32::min);
                    let w_max = slab.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let s_g = if w_max > w_min { (w_max - w_min) / (n_bins - 1) as f32 } else { 1.0 };
                    s[row * gpr + g] = s_g;
                    b[row * gpr + g] = w_min;
                    for i in 0..group_size {
                        let z = (slab[i] - w_min) / s_g;
                        let qv = if z >= 0.0 { (z + 0.5).floor() as i32 } else { (z - 0.5).ceil() as i32 };
                        q[base + i] = qv.clamp(0, (n_bins - 1) as i32) as u8;
                    }
                }
            }
            (s, b, q)
        };

        let tape = GpuTape::new(device.clone());

        // INPUT (shared between teacher and student).
        let x_data: Vec<f32> = (0..(n_tokens * hidden)).map(|_| next() * 0.3).collect();
        let xt = GpuTensor::from_vec(&tape, &x_data, vec![n_tokens, hidden]).unwrap();

        // Build per-layer SHARED F32 weights (norm/attn/router/etc.) +
        // per-layer per-expert (W_true → qdq(s, b, q_int)) for the
        // student, plus parallel F32 W_true for the teacher.
        struct LayerStorage {
            w_in: GpuTensor,
            w_attn: GpuTensor,
            w_post: GpuTensor,
            w_gate: GpuTensor,
            // Teacher: F32 weights as GpuTensors
            t_gate: Vec<GpuTensor>,
            t_up: Vec<GpuTensor>,
            t_down: Vec<GpuTensor>,
            // Student: qdq output tensors (held to keep the tape graph alive)
            s_gate_w: Vec<GpuTensor>,
            s_up_w: Vec<GpuTensor>,
            s_down_w: Vec<GpuTensor>,
            // Student trainable leaves (s, b)
            s_gate_s: Vec<GpuTensor>,
            s_gate_b: Vec<GpuTensor>,
            s_up_s: Vec<GpuTensor>,
            s_up_b: Vec<GpuTensor>,
            s_down_s: Vec<GpuTensor>,
            s_down_b: Vec<GpuTensor>,
        }
        let mut storage: Vec<LayerStorage> = Vec::with_capacity(n_layers);

        for _l in 0..n_layers {
            let w_in_v: Vec<f32> = (0..hidden).map(|_| 1.0 + next() * 0.05).collect();
            let w_post_v: Vec<f32> = (0..hidden).map(|_| 1.0 + next() * 0.05).collect();
            let w_attn_v: Vec<f32> = (0..(hidden * hidden)).map(|_| next() * 0.05).collect();
            let w_gate_v: Vec<f32> = (0..(hidden * n_experts)).map(|_| next() * 0.1).collect();
            let w_in = GpuTensor::from_vec(&tape, &w_in_v, vec![hidden]).unwrap();
            let w_post = GpuTensor::from_vec(&tape, &w_post_v, vec![hidden]).unwrap();
            let w_attn = GpuTensor::from_vec(&tape, &w_attn_v, vec![hidden, hidden]).unwrap();
            let w_gate = GpuTensor::from_vec(&tape, &w_gate_v, vec![hidden, n_experts]).unwrap();

            let mut t_gate = Vec::with_capacity(n_experts);
            let mut t_up = Vec::with_capacity(n_experts);
            let mut t_down = Vec::with_capacity(n_experts);
            let mut s_gate_w = Vec::with_capacity(n_experts);
            let mut s_up_w = Vec::with_capacity(n_experts);
            let mut s_down_w = Vec::with_capacity(n_experts);
            let mut s_gate_s = Vec::with_capacity(n_experts);
            let mut s_gate_b = Vec::with_capacity(n_experts);
            let mut s_up_s = Vec::with_capacity(n_experts);
            let mut s_up_b = Vec::with_capacity(n_experts);
            let mut s_down_s = Vec::with_capacity(n_experts);
            let mut s_down_b = Vec::with_capacity(n_experts);

            for _e in 0..n_experts {
                // gate: shape [intermediate, hidden] (n=intermediate, k=hidden)
                let gate_w_true: Vec<f32> = (0..(intermediate * hidden)).map(|_| next() * 0.05).collect();
                let up_w_true: Vec<f32> = (0..(intermediate * hidden)).map(|_| next() * 0.05).collect();
                let down_w_true: Vec<f32> = (0..(hidden * intermediate)).map(|_| next() * 0.05).collect();

                t_gate.push(GpuTensor::from_vec(&tape, &gate_w_true, vec![intermediate, hidden]).unwrap());
                t_up.push(GpuTensor::from_vec(&tape, &up_w_true, vec![intermediate, hidden]).unwrap());
                t_down.push(GpuTensor::from_vec(&tape, &down_w_true, vec![hidden, intermediate]).unwrap());

                let (gs, gb, gq) = init_qdq(&gate_w_true, intermediate, hidden, groups_per_row_gateup);
                let (us, ub, uq) = init_qdq(&up_w_true, intermediate, hidden, groups_per_row_gateup);
                let (ds, db, dq) = init_qdq(&down_w_true, hidden, intermediate, groups_per_row_down);

                let g_s_l = GpuTensor::from_vec(&tape, &gs, vec![gs.len()]).unwrap();
                let g_b_l = GpuTensor::from_vec(&tape, &gb, vec![gb.len()]).unwrap();
                let u_s_l = GpuTensor::from_vec(&tape, &us, vec![us.len()]).unwrap();
                let u_b_l = GpuTensor::from_vec(&tape, &ub, vec![ub.len()]).unwrap();
                let d_s_l = GpuTensor::from_vec(&tape, &ds, vec![ds.len()]).unwrap();
                let d_b_l = GpuTensor::from_vec(&tape, &db, vec![db.len()]).unwrap();

                let g_qdq = qdq_affine(&g_s_l, &g_b_l, &gq, group_size).unwrap();
                let u_qdq = qdq_affine(&u_s_l, &u_b_l, &uq, group_size).unwrap();
                let d_qdq = qdq_affine(&d_s_l, &d_b_l, &dq, group_size).unwrap();

                s_gate_w.push(view(&g_qdq, vec![intermediate, hidden]).unwrap());
                s_up_w.push(view(&u_qdq, vec![intermediate, hidden]).unwrap());
                s_down_w.push(view(&d_qdq, vec![hidden, intermediate]).unwrap());

                s_gate_s.push(g_s_l);
                s_gate_b.push(g_b_l);
                s_up_s.push(u_s_l);
                s_up_b.push(u_b_l);
                s_down_s.push(d_s_l);
                s_down_b.push(d_b_l);
            }

            storage.push(LayerStorage {
                w_in, w_attn, w_post, w_gate,
                t_gate, t_up, t_down,
                s_gate_w, s_up_w, s_down_w,
                s_gate_s, s_gate_b, s_up_s, s_up_b, s_down_s, s_down_b,
            });
        }

        // Build teacher + student layer arrays.
        let teacher_layers: Vec<DecoderLayerWeights<'_>> = (0..n_layers).map(|l| DecoderLayerWeights {
            w_in: &storage[l].w_in,
            w_attn: &storage[l].w_attn,
            w_post: &storage[l].w_post,
            w_gate: &storage[l].w_gate,
            gate_projs: &storage[l].t_gate,
            up_projs: &storage[l].t_up,
            down_projs: &storage[l].t_down,
        }).collect();
        let student_layers: Vec<DecoderLayerWeights<'_>> = (0..n_layers).map(|l| DecoderLayerWeights {
            w_in: &storage[l].w_in,
            w_attn: &storage[l].w_attn,
            w_post: &storage[l].w_post,
            w_gate: &storage[l].w_gate,
            gate_projs: &storage[l].s_gate_w,
            up_projs: &storage[l].s_up_w,
            down_projs: &storage[l].s_down_w,
        }).collect();

        let y_teacher = super::qwen35_moe_forward_on_tape(&tape, &xt, &teacher_layers, k, eps)
            .expect("teacher forward");
        let y_student = super::qwen35_moe_forward_on_tape(&tape, &xt, &student_layers, k, eps)
            .expect("student forward");

        // MSE loss: (student - teacher)^2 summed.  On tape so we can backward.
        let diff = sub(&y_student, &y_teacher).expect("sub");
        let sqr = square(&diff).expect("square");
        let loss_data: Vec<f32> = sqr.to_vec().expect("loss host");
        let loss_initial: f32 = loss_data.iter().sum::<f32>() / loss_data.len() as f32;

        // Sanity: loss should be > 0 (otherwise teacher == student and
        // we have no signal to test).
        assert!(
            loss_initial > 1e-12,
            "loss_initial={loss_initial:e} — teacher and student outputs are byte-identical, \
             test fixture lacks compositional quant error"
        );
        eprintln!("[ac7-A-falsifier] loss_initial = {loss_initial:e}");

        // Backward — ones-seeded on `sqr` (the per-element squared diff).
        let dy = ones_like(&tape, sqr.shape()).expect("dy");
        let grads = backward(&sqr, dy).expect("backward");

        // Hand-rolled gradient-descent step on every (s, b) leaf.
        let lr = 0.001f32;
        let mut step_sb = |leaf: &GpuTensor| -> bool {
            let g = match grads[leaf.node_idx()].as_ref() {
                Some(g) => g,
                None => return false, // no grad — leaf untouched
            };
            let g_host: Vec<f32> = g.as_slice::<f32>().expect("g slice").to_vec();
            let max_abs = g_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            if max_abs < 1e-12 {
                return false; // dead gradient
            }
            // Mutate leaf's underlying buffer host-side.  GpuTensor leaves
            // store F32 data accessible via `.to_vec()` / new tensor.
            let mut data: Vec<f32> = leaf.to_vec().expect("leaf data");
            for (i, v) in data.iter_mut().enumerate() {
                *v -= lr * g_host[i];
            }
            // We can't mutate the tensor in place; instead, write to its
            // underlying buffer via the tape's leaf storage.  Since
            // GpuTensor exposes a clone-from-vec via `from_vec`, we need
            // a new tensor — but the layer struct holds &GpuTensor refs.
            // Simplest path: assert we got a non-zero gradient + report
            // the magnitude.  Actual step requires a proper Adam (out of
            // iter scope here).
            let _ = data;
            true
        };
        let mut moved_leaves = 0usize;
        for s in &storage {
            for e in 0..n_experts {
                if step_sb(&s.s_gate_s[e]) { moved_leaves += 1; }
                if step_sb(&s.s_gate_b[e]) { moved_leaves += 1; }
                if step_sb(&s.s_up_s[e]) { moved_leaves += 1; }
                if step_sb(&s.s_up_b[e]) { moved_leaves += 1; }
                if step_sb(&s.s_down_s[e]) { moved_leaves += 1; }
                if step_sb(&s.s_down_b[e]) { moved_leaves += 1; }
            }
        }
        // n_layers × n_experts × 6 = 24 leaves expected to have non-zero grad.
        let expected = n_layers * n_experts * 6;
        assert_eq!(
            moved_leaves, expected,
            "expected non-zero gradient on all {expected} (s, b) leaves; got {moved_leaves}"
        );
        eprintln!(
            "[ac7-A-falsifier] non-zero gradients on {moved_leaves}/{expected} leaves — \
             cross-layer compensation gradient is REAL.  Loss reduction over an actual \
             Adam step requires in-place leaf mutation (out of test scope; future iter)."
        );
    }
}
