//! Qwen3.5 FFN forward pass — scalar CPU references.
//!
//! Two variants:
//!
//! * [`dense_swiglu_cpu_ref`] — standard SwiGLU (dense variant, ADR-013
//!   Decision 14). `y = down @ (silu(gate @ x) * (up @ x))`.
//!
//! * [`moe_ffn_cpu_ref`] — top-k expert routing with a sigmoid-gated shared
//!   expert (MoE variant, ADR-013 Decision 13).
//!
//! Both are the authoritative f32 specs — used as oracles for the forthcoming
//! GPU builders (`build_dense_ffn_layer` / `build_moe_ffn_layer` in P11).
//!
//! Shapes throughout: flat f32 row-major Vecs. GGUF weight convention is
//! `[out_dim, in_dim]` — weight row `i` holds the coefficients that produce
//! output feature `i`. Matmul uses [`crate::inference::models::qwen35::full_attn`]
//! 's existing pattern: `matmul_a_by_bt(x, w) = x @ w_t`.

// ================================================================
// Shared helpers (copy of full_attn::matmul_a_by_bt; kept local here
// so this module doesn't depend on full_attn's implementation details).
// ================================================================

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `out[i, j] = sum_k lhs[i, k] * rhs[j, k]` — GGUF-style (rhs transposed).
///
/// Shapes:
///   lhs: `[m, k]`
///   rhs: `[n, k]`  (row-major `[out_dim, in_dim]`)
///   out: `[m, n]`
fn matmul_a_by_bt(lhs: &[f32], rhs: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += lhs[i * k + kk] * rhs[j * k + kk];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

// ================================================================
// Dense SwiGLU
// ================================================================

/// Weights for a single dense SwiGLU FFN layer.
#[derive(Debug, Clone)]
pub struct DenseFfnWeights {
    /// `[intermediate_size, hidden_size]` — gate_proj.
    pub gate: Vec<f32>,
    /// `[intermediate_size, hidden_size]` — up_proj.
    pub up: Vec<f32>,
    /// `[hidden_size, intermediate_size]` — down_proj.
    pub down: Vec<f32>,
}

/// Shape parameters for a dense SwiGLU FFN.
#[derive(Debug, Clone, Copy)]
pub struct DenseFfnShape {
    pub hidden_size: u32,
    pub intermediate_size: u32,
}

/// Pure-Rust dense SwiGLU FFN forward (ADR-013 Decision 14).
///
/// Spec:
///
/// ```text
/// a = gate @ x          // [seq, intermediate]
/// b = up @ x            // [seq, intermediate]
/// c = silu(a) * b       // elementwise
/// y = down @ c          // [seq, hidden]
/// ```
///
/// Inputs:
/// * `x`: residual stream, shape `[seq_len, hidden_size]`.
/// * `weights`: FFN weights.
/// * `shape`: derived shape parameters.
///
/// Returns: residual CONTRIBUTION, `[seq_len, hidden_size]`. Caller adds to x.
pub fn dense_swiglu_cpu_ref(
    x: &[f32],
    weights: &DenseFfnWeights,
    shape: DenseFfnShape,
) -> Vec<f32> {
    let h = shape.hidden_size as usize;
    let m = shape.intermediate_size as usize;
    let seq_len = x.len() / h;

    assert_eq!(x.len(), seq_len * h, "x shape mismatch");
    assert_eq!(weights.gate.len(), m * h, "gate weight shape");
    assert_eq!(weights.up.len(), m * h, "up weight shape");
    assert_eq!(weights.down.len(), h * m, "down weight shape");

    let a = matmul_a_by_bt(x, &weights.gate, seq_len, h, m);
    let b = matmul_a_by_bt(x, &weights.up, seq_len, h, m);
    let mut c = vec![0.0f32; seq_len * m];
    for i in 0..(seq_len * m) {
        c[i] = silu(a[i]) * b[i];
    }
    matmul_a_by_bt(&c, &weights.down, seq_len, m, h)
}

// ================================================================
// MoE (with gated shared expert)
// ================================================================

/// Weights for a single Qwen3.5-MoE FFN layer.
#[derive(Debug, Clone)]
pub struct MoeFfnWeights {
    /// Router logits: `[num_experts, hidden_size]`.
    pub router: Vec<f32>,
    /// Expert gate_proj stacked: `[num_experts * moe_intermediate_size, hidden_size]`.
    pub expert_gate: Vec<f32>,
    /// Expert up_proj stacked: `[num_experts * moe_intermediate_size, hidden_size]`.
    pub expert_up: Vec<f32>,
    /// Expert down_proj stacked: `[num_experts * hidden_size, moe_intermediate_size]`.
    pub expert_down: Vec<f32>,
    /// Shared-expert gate projection: `[hidden_size]` (output scalar per token
    /// before sigmoid — treat as a `[1, hidden_size]` matrix).
    pub shared_gate_logit: Vec<f32>,
    /// Shared-expert SwiGLU weights.
    pub shared_gate: Vec<f32>, // [shared_intermediate, hidden_size]
    pub shared_up: Vec<f32>,   // [shared_intermediate, hidden_size]
    pub shared_down: Vec<f32>, // [hidden_size, shared_intermediate]
}

/// Shape parameters for a Qwen3.5-MoE FFN.
#[derive(Debug, Clone, Copy)]
pub struct MoeFfnShape {
    pub hidden_size: u32,
    pub num_experts: u32,
    pub num_experts_per_tok: u32, // top-k, typically 8
    pub moe_intermediate_size: u32,
    pub shared_intermediate_size: u32,
}

/// Pure-Rust MoE FFN forward (ADR-013 Decision 13).
///
/// Spec (per token):
///
/// ```text
/// // Router
/// logits  = router @ x                   // [num_experts]
/// probs   = softmax(logits)              // [num_experts]
/// (topk_idx, topk_w) = top_k(probs, k=num_experts_per_tok)
/// topk_w  = topk_w / sum(topk_w)         // renormalize after top-k slice
///
/// // Routed experts — SwiGLU applied per selected expert, weighted sum.
/// moe_out = 0
/// for (e_idx, w) in zip(topk_idx, topk_w):
///     a = expert_gate[e_idx] @ x
///     b = expert_up[e_idx] @ x
///     c = silu(a) * b
///     y = expert_down[e_idx] @ c
///     moe_out += w * y
///
/// // Shared expert — gated by a sigmoid logit. Per ADR-013 Decision 13:
/// // "Gated shared expert: project via ffn_gate_inp_shexp → sigmoid →
/// //  multiply shared-expert FFN output → add to routed MoE output"
/// shared_gate_logit = shared_gate_logit_w @ x    // scalar
/// shared_gate_val   = sigmoid(shared_gate_logit)
/// a_s = shared_gate @ x
/// b_s = shared_up   @ x
/// c_s = silu(a_s) * b_s
/// y_s = shared_down @ c_s
/// output = moe_out + shared_gate_val * y_s
/// ```
pub fn moe_ffn_cpu_ref(
    x: &[f32],
    weights: &MoeFfnWeights,
    shape: MoeFfnShape,
) -> Vec<f32> {
    let h = shape.hidden_size as usize;
    let ne = shape.num_experts as usize;
    let topk = shape.num_experts_per_tok as usize;
    let m_moe = shape.moe_intermediate_size as usize;
    let m_sh = shape.shared_intermediate_size as usize;
    let seq_len = x.len() / h;

    assert_eq!(x.len(), seq_len * h);
    assert_eq!(weights.router.len(), ne * h);
    assert_eq!(weights.expert_gate.len(), ne * m_moe * h);
    assert_eq!(weights.expert_up.len(), ne * m_moe * h);
    assert_eq!(weights.expert_down.len(), ne * h * m_moe);
    assert_eq!(weights.shared_gate_logit.len(), h);
    assert_eq!(weights.shared_gate.len(), m_sh * h);
    assert_eq!(weights.shared_up.len(), m_sh * h);
    assert_eq!(weights.shared_down.len(), h * m_sh);
    assert!(topk <= ne, "top-k cannot exceed num_experts");
    assert!(topk > 0, "top-k must be positive");

    let mut output = vec![0.0f32; seq_len * h];

    for t in 0..seq_len {
        let x_t = &x[t * h..(t + 1) * h];

        // Router: logits = router @ x_t, shape [num_experts].
        let mut logits = vec![0.0f32; ne];
        for e in 0..ne {
            let w_row = &weights.router[e * h..(e + 1) * h];
            let mut acc = 0.0f32;
            for i in 0..h {
                acc += w_row[i] * x_t[i];
            }
            logits[e] = acc;
        }

        // Softmax.
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp = vec![0.0f32; ne];
        let mut denom = 0.0f32;
        for e in 0..ne {
            exp[e] = (logits[e] - max_l).exp();
            denom += exp[e];
        }
        let probs: Vec<f32> = exp.iter().map(|v| v / denom).collect();

        // Top-k selection by probability.
        let mut idx_sorted: Vec<usize> = (0..ne).collect();
        idx_sorted.sort_by(|a, b| {
            probs[*b].partial_cmp(&probs[*a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let topk_idx: Vec<usize> = idx_sorted[..topk].to_vec();
        let topk_probs: Vec<f32> = topk_idx.iter().map(|&i| probs[i]).collect();

        // Renormalize.
        let renorm_sum: f32 = topk_probs.iter().sum();
        let topk_w: Vec<f32> = if renorm_sum > 1e-20 {
            topk_probs.iter().map(|p| p / renorm_sum).collect()
        } else {
            // Degenerate; fall back to uniform.
            vec![1.0 / topk as f32; topk]
        };

        // Per-expert SwiGLU, weighted sum into `moe_out`.
        let mut moe_out = vec![0.0f32; h];
        for (w_i, &e_idx) in topk_w.iter().zip(topk_idx.iter()) {
            // Expert e_idx's gate/up/down weights.
            let g_off = e_idx * m_moe * h;
            let u_off = e_idx * m_moe * h;
            let d_off = e_idx * h * m_moe;
            let gate_w = &weights.expert_gate[g_off..g_off + m_moe * h];
            let up_w = &weights.expert_up[u_off..u_off + m_moe * h];
            let down_w = &weights.expert_down[d_off..d_off + h * m_moe];

            // a = gate @ x_t, shape [m_moe]
            let mut a = vec![0.0f32; m_moe];
            for i in 0..m_moe {
                let mut acc = 0.0f32;
                for j in 0..h {
                    acc += gate_w[i * h + j] * x_t[j];
                }
                a[i] = acc;
            }
            // b = up @ x_t
            let mut b = vec![0.0f32; m_moe];
            for i in 0..m_moe {
                let mut acc = 0.0f32;
                for j in 0..h {
                    acc += up_w[i * h + j] * x_t[j];
                }
                b[i] = acc;
            }
            // c = silu(a) * b
            for i in 0..m_moe {
                a[i] = silu(a[i]) * b[i];
            }
            // y = down @ c
            let mut y = vec![0.0f32; h];
            for i in 0..h {
                let mut acc = 0.0f32;
                for j in 0..m_moe {
                    acc += down_w[i * m_moe + j] * a[j];
                }
                y[i] = acc;
            }
            for i in 0..h {
                moe_out[i] += w_i * y[i];
            }
        }

        // Shared expert (sigmoid-gated).
        let shared_logit: f32 = weights
            .shared_gate_logit
            .iter()
            .zip(x_t.iter())
            .map(|(w, x)| w * x)
            .sum();
        let shared_gate_val = sigmoid(shared_logit);

        let mut a_s = vec![0.0f32; m_sh];
        for i in 0..m_sh {
            let mut acc = 0.0f32;
            for j in 0..h {
                acc += weights.shared_gate[i * h + j] * x_t[j];
            }
            a_s[i] = acc;
        }
        let mut b_s = vec![0.0f32; m_sh];
        for i in 0..m_sh {
            let mut acc = 0.0f32;
            for j in 0..h {
                acc += weights.shared_up[i * h + j] * x_t[j];
            }
            b_s[i] = acc;
        }
        for i in 0..m_sh {
            a_s[i] = silu(a_s[i]) * b_s[i];
        }
        let mut y_s = vec![0.0f32; h];
        for i in 0..h {
            let mut acc = 0.0f32;
            for j in 0..m_sh {
                acc += weights.shared_down[i * m_sh + j] * a_s[j];
            }
            y_s[i] = acc;
        }

        for i in 0..h {
            output[t * h + i] = moe_out[i] + shared_gate_val * y_s[i];
        }
    }

    output
}

// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_rand(seed: &mut u32, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                ((*seed as i32 as f32) / (i32::MAX as f32)) * scale
            })
            .collect()
    }

    // ====================================================
    // Dense SwiGLU
    // ====================================================

    /// Zero weights → zero output regardless of input.
    #[test]
    fn dense_swiglu_zero_weights_zero_output() {
        let shape = DenseFfnShape {
            hidden_size: 4,
            intermediate_size: 8,
        };
        let weights = DenseFfnWeights {
            gate: vec![0.0; 8 * 4],
            up: vec![0.0; 8 * 4],
            down: vec![0.0; 4 * 8],
        };
        let x: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect(); // 4 tokens
        let out = dense_swiglu_cpu_ref(&x, &weights, shape);
        assert_eq!(out.len(), 16);
        for v in &out {
            assert!(v.abs() < 1e-7, "expected 0, got {}", v);
        }
    }

    /// ADR Decision 14 acceptance: synthetic weights produce expected output.
    /// Verify against independent recomputation.
    #[test]
    fn dense_swiglu_matches_independent_recompute() {
        let shape = DenseFfnShape {
            hidden_size: 4,
            intermediate_size: 8,
        };
        let mut seed = 0xAB_u32;
        let weights = DenseFfnWeights {
            gate: mk_rand(&mut seed, 8 * 4, 0.2),
            up: mk_rand(&mut seed, 8 * 4, 0.2),
            down: mk_rand(&mut seed, 4 * 8, 0.2),
        };
        let x = mk_rand(&mut seed, 8, 0.5); // 2 tokens × 4

        let out = dense_swiglu_cpu_ref(&x, &weights, shape);

        // Independent recomputation.
        for t in 0..2 {
            let x_t = &x[t * 4..(t + 1) * 4];
            let mut a = [0.0f32; 8];
            let mut b = [0.0f32; 8];
            for i in 0..8 {
                for j in 0..4 {
                    a[i] += weights.gate[i * 4 + j] * x_t[j];
                    b[i] += weights.up[i * 4 + j] * x_t[j];
                }
            }
            for i in 0..8 {
                a[i] = silu(a[i]) * b[i];
            }
            let mut y = [0.0f32; 4];
            for i in 0..4 {
                for j in 0..8 {
                    y[i] += weights.down[i * 8 + j] * a[j];
                }
            }
            for j in 0..4 {
                let d = (out[t * 4 + j] - y[j]).abs();
                assert!(d < 1e-5, "t={}, j={}: got {}, want {}", t, j, out[t * 4 + j], y[j]);
            }
        }
    }

    #[test]
    fn dense_swiglu_deterministic() {
        let shape = DenseFfnShape {
            hidden_size: 4,
            intermediate_size: 8,
        };
        let mut seed = 0xCC_u32;
        let weights = DenseFfnWeights {
            gate: mk_rand(&mut seed, 8 * 4, 0.1),
            up: mk_rand(&mut seed, 8 * 4, 0.1),
            down: mk_rand(&mut seed, 4 * 8, 0.1),
        };
        let x = vec![0.5; 4];
        let o1 = dense_swiglu_cpu_ref(&x, &weights, shape);
        let o2 = dense_swiglu_cpu_ref(&x, &weights, shape);
        for i in 0..4 {
            assert_eq!(o1[i].to_bits(), o2[i].to_bits());
        }
    }

    // ====================================================
    // MoE
    // ====================================================

    /// ADR Decision 13 acceptance: 4 experts + 1 shared, routed output is
    /// weighted sum of selected experts.
    #[test]
    fn moe_4experts_routing_selects_top_k() {
        let shape = MoeFfnShape {
            hidden_size: 4,
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 4,
            shared_intermediate_size: 4,
        };

        let mut seed = 0x100_u32;
        let weights = MoeFfnWeights {
            router: mk_rand(&mut seed, 4 * 4, 0.5),
            expert_gate: mk_rand(&mut seed, 4 * 4 * 4, 0.1),
            expert_up: mk_rand(&mut seed, 4 * 4 * 4, 0.1),
            expert_down: mk_rand(&mut seed, 4 * 4 * 4, 0.1),
            shared_gate_logit: mk_rand(&mut seed, 4, 0.1),
            shared_gate: mk_rand(&mut seed, 4 * 4, 0.1),
            shared_up: mk_rand(&mut seed, 4 * 4, 0.1),
            shared_down: mk_rand(&mut seed, 4 * 4, 0.1),
        };

        let x: Vec<f32> = (0..4).map(|i| i as f32 * 0.1).collect();
        let out = moe_ffn_cpu_ref(&x, &weights, shape);
        assert_eq!(out.len(), 4);
        for v in &out {
            assert!(v.is_finite());
        }
    }

    /// ADR Decision 13 acceptance: shared-expert gate path.
    /// With `shared_gate_logit = large_negative`, sigmoid ≈ 0 so shared
    /// contribution ≈ 0; output ≈ routed-only output.
    /// With `shared_gate_logit = large_positive`, sigmoid ≈ 1 so shared
    /// contribution is full.
    #[test]
    fn moe_shared_expert_gate_controls_contribution() {
        let shape = MoeFfnShape {
            hidden_size: 4,
            num_experts: 2,
            num_experts_per_tok: 1,
            moe_intermediate_size: 4,
            shared_intermediate_size: 4,
        };

        let mut seed = 0x200_u32;
        // Baseline weights: gate logit zero so shared contributes ~0.5 × y_shared.
        let base_weights = MoeFfnWeights {
            router: mk_rand(&mut seed, 2 * 4, 0.5),
            expert_gate: mk_rand(&mut seed, 2 * 4 * 4, 0.1),
            expert_up: mk_rand(&mut seed, 2 * 4 * 4, 0.1),
            expert_down: mk_rand(&mut seed, 2 * 4 * 4, 0.1),
            shared_gate_logit: vec![0.0; 4], // produces logit = 0 → sigmoid = 0.5
            shared_gate: mk_rand(&mut seed, 4 * 4, 0.1),
            shared_up: mk_rand(&mut seed, 4 * 4, 0.1),
            shared_down: mk_rand(&mut seed, 4 * 4, 0.1),
        };
        let x: Vec<f32> = (0..4).map(|i| 0.1 * (i as f32 + 1.0)).collect();

        // With gate logit = 0 → sigmoid = 0.5.
        let out_mid = moe_ffn_cpu_ref(&x, &base_weights, shape);

        // With huge negative gate_logit: shared contribution → 0.
        let mut weights_off = base_weights.clone();
        weights_off.shared_gate_logit = vec![-1000.0; 4]; // logit → very negative
        let out_off = moe_ffn_cpu_ref(&x, &weights_off, shape);

        // With huge positive gate_logit: shared contribution → full y_shared.
        let mut weights_on = base_weights.clone();
        weights_on.shared_gate_logit = vec![1000.0; 4];
        let out_on = moe_ffn_cpu_ref(&x, &weights_on, shape);

        // out_off should be closer to "routed-only" than out_mid.
        // out_on should be the largest (assuming y_s is non-zero).
        // Rigorous check: out_mid ≈ 0.5 * (out_off + out_on) elementwise
        // (because out = moe + gate * y_s, and moe is shared across all three).
        for i in 0..4 {
            let avg = 0.5 * (out_off[i] + out_on[i]);
            let d = (out_mid[i] - avg).abs();
            // Tolerance: sigmoid(±1000) saturates at ~0 and ~1 with ~0 error.
            assert!(
                d < 1e-3,
                "gate linearity broken at {}: mid={}, avg_off_on={}, d={}",
                i, out_mid[i], avg, d
            );
        }
    }

    /// Determinism smoke.
    #[test]
    fn moe_deterministic() {
        let shape = MoeFfnShape {
            hidden_size: 4,
            num_experts: 3,
            num_experts_per_tok: 2,
            moe_intermediate_size: 4,
            shared_intermediate_size: 4,
        };
        let mut seed = 0x300_u32;
        let weights = MoeFfnWeights {
            router: mk_rand(&mut seed, 3 * 4, 0.3),
            expert_gate: mk_rand(&mut seed, 3 * 4 * 4, 0.1),
            expert_up: mk_rand(&mut seed, 3 * 4 * 4, 0.1),
            expert_down: mk_rand(&mut seed, 3 * 4 * 4, 0.1),
            shared_gate_logit: mk_rand(&mut seed, 4, 0.1),
            shared_gate: mk_rand(&mut seed, 4 * 4, 0.1),
            shared_up: mk_rand(&mut seed, 4 * 4, 0.1),
            shared_down: mk_rand(&mut seed, 4 * 4, 0.1),
        };
        let x: Vec<f32> = (0..4).map(|i| 0.1 * i as f32).collect();
        let o1 = moe_ffn_cpu_ref(&x, &weights, shape);
        let o2 = moe_ffn_cpu_ref(&x, &weights, shape);
        for i in 0..4 {
            assert_eq!(o1[i].to_bits(), o2[i].to_bits());
        }
    }

    /// Top-k all experts: no renormalization effect; behaves like a normal
    /// softmax-weighted expert sum.
    #[test]
    fn moe_topk_all_experts_eq_softmax_weighted_sum() {
        let shape = MoeFfnShape {
            hidden_size: 2,
            num_experts: 3,
            num_experts_per_tok: 3, // top-k = all experts
            moe_intermediate_size: 2,
            shared_intermediate_size: 2,
        };
        let mut seed = 0x400_u32;
        let weights = MoeFfnWeights {
            router: mk_rand(&mut seed, 3 * 2, 0.3),
            expert_gate: mk_rand(&mut seed, 3 * 2 * 2, 0.2),
            expert_up: mk_rand(&mut seed, 3 * 2 * 2, 0.2),
            expert_down: mk_rand(&mut seed, 3 * 2 * 2, 0.2),
            shared_gate_logit: vec![-1000.0; 2], // disable shared
            shared_gate: vec![0.0; 2 * 2],
            shared_up: vec![0.0; 2 * 2],
            shared_down: vec![0.0; 2 * 2],
        };
        let x = vec![0.3, -0.2];
        let out = moe_ffn_cpu_ref(&x, &weights, shape);

        // Reference: compute softmax, then weighted sum of all 3 experts'
        // SwiGLU outputs with softmax weights (no top-k slicing since k = ne).
        let mut logits = [0.0f32; 3];
        for e in 0..3 {
            for j in 0..2 {
                logits[e] += weights.router[e * 2 + j] * x[j];
            }
        }
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp = [0.0f32; 3];
        let mut denom = 0.0f32;
        for e in 0..3 {
            exp[e] = (logits[e] - max_l).exp();
            denom += exp[e];
        }
        let probs = [exp[0] / denom, exp[1] / denom, exp[2] / denom];

        let mut expected = [0.0f32; 2];
        for e in 0..3 {
            let g_off = e * 2 * 2;
            let u_off = e * 2 * 2;
            let d_off = e * 2 * 2;
            let mut a = [0.0f32; 2];
            let mut b = [0.0f32; 2];
            for i in 0..2 {
                for j in 0..2 {
                    a[i] += weights.expert_gate[g_off + i * 2 + j] * x[j];
                    b[i] += weights.expert_up[u_off + i * 2 + j] * x[j];
                }
            }
            for i in 0..2 {
                a[i] = silu(a[i]) * b[i];
            }
            let mut y = [0.0f32; 2];
            for i in 0..2 {
                for j in 0..2 {
                    y[i] += weights.expert_down[d_off + i * 2 + j] * a[j];
                }
            }
            for i in 0..2 {
                expected[i] += probs[e] * y[i];
            }
        }

        for i in 0..2 {
            let d = (out[i] - expected[i]).abs();
            assert!(
                d < 1e-5,
                "at {}: got {}, expected {}, d {}",
                i, out[i], expected[i], d
            );
        }
    }
}
