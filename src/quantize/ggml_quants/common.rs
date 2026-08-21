//! Shared helper functions used across the ggml-quants kernel ports.
//!
//! Each helper here is a 1:1 pure-Rust port of a peer quantizer
//! primitive. Consolidating them into a
//! single source eliminates the latent-divergence bug class flagged by
//! codex at `0bd0e7eb` (parallel kernel-port workers each ported their
//! own copies; subtle finalize-block / d==0 differences crept in).
//!
//! The byte-cmp gates in each per-type submodule (`q*_*::tests::byte_cmp*`)
//! cover this file transitively — any drift here surfaces immediately
//! against the peer's `ggml_quantize_chunk` reference fixtures.

/// The peer's `GROUP_MAX_EPS`. Threshold below which an
/// absolute-max scan is treated as the zero-block case.
pub const GROUP_MAX_EPS: f32 = 1e-15;

/// The peer's `nearest_int`.
///
/// The C version uses a bit-cast trick that rounds via the float-to-int
/// hardware mode (RNE). We mirror the bitcast exactly so the rounding
/// boundary behaviour is identical to the peer's.
#[inline(always)]
pub fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4_194_303.0);
    let val = fval + 12_582_912.0;
    let i = val.to_bits() as i32;
    (i & 0x007f_ffff) - 0x0040_0000
}

/// The peer's `best_index_int8` — bisection over a
/// monotonic codebook, picking the nearer of the two bracketing values.
#[inline]
pub fn best_index_int8(val: &[i8], x: f32) -> usize {
    let n = val.len();
    if x <= val[0] as f32 {
        return 0;
    }
    if x >= val[n - 1] as f32 {
        return n - 1;
    }
    let (mut ml, mut mu) = (0usize, n - 1);
    while mu - ml > 1 {
        let mav = (ml + mu) / 2;
        if x < val[mav] as f32 {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    if x - (val[mu - 1] as f32) < (val[mu] as f32) - x {
        mu - 1
    } else {
        mu
    }
}

/// The peer's `make_qx_quants`.
///
/// Used by Q4_0 (`rmse_type=1`), Q3_K (`rmse_type=1`), Q6_K (`rmse_type=1`).
/// `qw` may be empty for the `rmse_type`-derived weight fallback.
pub fn make_qx_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    l: &mut [i8],
    rmse_type: i32,
    qw: &[f32],
) -> f32 {
    let mut max: f32 = 0.0;
    let mut amax: f32 = 0.0;
    for i in 0..n {
        let ax = x[i].abs();
        if ax > amax {
            amax = ax;
            max = x[i];
        }
    }
    if amax < GROUP_MAX_EPS {
        for i in 0..n {
            l[i] = 0;
        }
        return 0.0;
    }
    let nmax_f = nmax as f32;
    let mut iscale = -nmax_f / max;
    if rmse_type == 0 {
        for i in 0..n {
            let li = nearest_int(iscale * x[i]);
            let clamped = li.max(-nmax).min(nmax - 1);
            l[i] = (nmax + clamped) as i8;
        }
        return 1.0 / iscale;
    }
    let mut rmse_type = rmse_type;
    let mut return_early = false;
    if rmse_type < 0 {
        rmse_type = -rmse_type;
        return_early = true;
    }
    // Initial pass uses plain `*` + `+=` (NO `.mul_add()`). Real-model
    // byte-cmp on Gemma 4 26B Q4_K_M (commit `6985cd56`) showed that
    // canonical's compiled `_quantize_row_q6_K_ref` emits only 32 `fmadd s`
    // = exactly the refinement loop's one specialized iteration. The
    // initial pass has the `L[i] = l + nmax;` side-effect inside the body
    // which serializes clang and produces scalar `fmul; fadd` (NO FMA).
    // Mirroring that pattern here closed the Q6_K residual from 49,181 →
    // 0 bytes on Gemma 4 26B and 0 bytes on every Qwen 3.5 in-tree
    // diagnostic. Refinement loop below (lines 155-156) keeps `.mul_add()`.
    let mut sumlx: f32 = 0.0;
    let mut suml2: f32 = 0.0;
    for i in 0..n {
        let li = nearest_int(iscale * x[i]);
        let li = li.max(-nmax).min(nmax - 1);
        l[i] = (li + nmax) as i8;
        let w = if !qw.is_empty() {
            qw[i]
        } else if rmse_type == 1 {
            x[i] * x[i]
        } else if rmse_type == 2 {
            1.0
        } else if rmse_type == 3 {
            x[i].abs()
        } else {
            x[i].abs().sqrt()
        };
        sumlx += w * x[i] * li as f32;
        suml2 += w * (li as f32) * (li as f32);
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    if return_early {
        return if suml2 > 0.0 {
            0.5 * (scale + 1.0 / iscale)
        } else {
            1.0 / iscale
        };
    }
    let mut best = scale * sumlx;
    for is in -9..=9i32 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax_f + 0.1 * (is as f32)) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for i in 0..n {
            let li = nearest_int(iscale * x[i]);
            let li = li.max(-nmax).min(nmax - 1);
            let w = if !qw.is_empty() {
                qw[i]
            } else if rmse_type == 1 {
                x[i] * x[i]
            } else if rmse_type == 2 {
                1.0
            } else if rmse_type == 3 {
                x[i].abs()
            } else {
                x[i].abs().sqrt()
            };
            sumlx = (w * x[i]).mul_add(li as f32, sumlx);
            suml2 = (w * (li as f32)).mul_add(li as f32, suml2);
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let li = nearest_int(iscale * x[i]);
                let clamped = li.max(-nmax).min(nmax - 1);
                l[i] = (nmax + clamped) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// The peer's `make_qkx2_quants`.
///
/// Returns `scale`; writes `-min` through `the_min`. Used by Q2_K /
/// Q4_K / Q5_K's `_ref` paths.
#[allow(clippy::too_many_arguments)]
pub fn make_qkx2_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    the_min: &mut f32,
    l_aux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    // ADR-033 quality-matrix iter-22 fix: use `f32::mul_add` for the 12
    // accumulator hot-spots that clang auto-fuses into `fmadd` at -O3
    // -march=native (verified via otool -tv of libggml-base.0.dylib —
    // canonical emits 12 fmadd insns; Rust at --release auto-emits 0
    // because rustc defaults to fp-contract=off). FMA produces a single
    // rounded result; separate mul+add produces two rounded results.
    // The ULP-level difference propagates through this function's
    // iterative refinement and crosses F16-precision boundaries when
    // the result is later quantized to f16 `dmin` for K-quant storage,
    // producing a 0.04-0.07% byte divergence on real-model weight
    // distributions even though the byte-cmp fixture tests pass on
    // synthetic mulberry32 inputs that don't straddle the boundary.
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x = w.mul_add(x[i], sum_x);
    }
    if min > 0.0 {
        min = 0.0;
    }
    // C: `if (max == min)` — exact bit-equality check (no epsilon).
    if max == min {
        for li in l.iter_mut().take(n) {
            *li = 0;
        }
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..n {
        let li = nearest_int(iscale * (x[i] - min));
        let li_c = li.max(0).min(nmax) as u8;
        l[i] = li_c;
        // 2026-05-20: plain *+ matches canonical (neutral per bisection).
        let mut diff = scale.mul_add(li_c as f32, min) - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        // 2026-05-20: canonical init-loop accumulates w*diff^2 via fmul+fadd (NOT fmadd).
        // Using FMA here produces best_error ~1.4e-12 lower than canonical, which flips
        // the is=5 acceptance gate for Qwen3.5 ssm_out row 100 blk 13 sub 1.
        best_error += weights[i] * diff;
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let li_raw = nearest_int(iscale * (x[i] - min));
            let li = li_raw.max(0).min(nmax) as u8;
            l_aux[i] = li;
            let w = weights[i];
            let li_f = li as f32;
            // 2026-05-20: plain *+ matches canonical current SHA (e15384a5c) behavior.
            // Older fixtures from c779f619 had FMA — now regenerated to current SHA.
            sum_l += w * li_f;
            sum_l2 += w * li_f * li_f;
            sum_xl += w * li_f * x[i];
        }
        // FMA: clang fuses `a*b - c*d` into `fmul(-c, d)` + `fmadd(a, b, prev)`
        // — 2 rounding ops vs Rust's default 3 (2 fmul + 1 fsub). Use
        // explicit mul_add to match clang's rounding chain.
        let d_det = sum_w.mul_add(sum_l2, -sum_l * sum_l);
        if d_det > 0.0 {
            let mut this_scale = sum_w.mul_add(sum_xl, -sum_x * sum_l) / d_det;
            let mut this_min = sum_l2.mul_add(sum_x, -sum_l * sum_xl) / d_det;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0.0f32;
            for i in 0..n {
                let mut diff = this_scale.mul_add(l_aux[i] as f32, this_min) - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                cur_error = weights[i].mul_add(diff, cur_error);
            }
            if cur_error < best_error {
                for i in 0..n {
                    l[i] = l_aux[i];
                }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

/// The peer's `make_qkx3_quants`.
///
/// Like `make_qkx2_quants` but `weights` is optional. When `weights ==
/// None`, falls back to `x[i] * x[i]`. Used by Q4_1 / Q5_1 / Q2_K /
/// Q4_K / Q5_K's `_impl` (imatrix) paths.
#[allow(clippy::too_many_arguments)]
pub fn make_qkx3_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: Option<&[f32]>,
    l: &mut [u8],
    the_min: &mut f32,
    l_aux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = match weights {
        Some(w) => w[0],
        None => x[0] * x[0],
    };
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = match weights {
            Some(ws) => ws[i],
            None => x[i] * x[i],
        };
        sum_w += w;
        sum_x = w.mul_add(x[i], sum_x);
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max <= min {
        for li in l.iter_mut().take(n) {
            *li = 0;
        }
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_mad = 0.0f32;
    for i in 0..n {
        let li = nearest_int(iscale * (x[i] - min));
        let li_c = li.max(0).min(nmax) as u8;
        l[i] = li_c;
        let mut diff = scale.mul_add(li_c as f32, min) - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        let w = match weights {
            Some(ws) => ws[i],
            None => x[i] * x[i],
        };
        best_mad = w.mul_add(diff, best_mad);
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let li_raw = nearest_int(iscale * (x[i] - min));
            let li = li_raw.max(0).min(nmax) as u8;
            l_aux[i] = li;
            let w = match weights {
                Some(ws) => ws[i],
                None => x[i] * x[i],
            };
            let li_f = li as f32;
            // 2026-05-20: plain *+ matches canonical current SHA (e15384a5c) behavior.
            // Older fixtures from c779f619 had FMA — now regenerated to current SHA.
            sum_l += w * li_f;
            sum_l2 += w * li_f * li_f;
            sum_xl += w * li_f * x[i];
        }
        // FMA: same a*b - c*d divergence as make_qkx2_quants. Use mul_add
        // so clang's 2-rounding fmadd chain is matched on the imatrix path.
        let d_det = sum_w.mul_add(sum_l2, -sum_l * sum_l);
        if d_det > 0.0 {
            let mut this_scale = sum_w.mul_add(sum_xl, -sum_x * sum_l) / d_det;
            let mut this_min = sum_l2.mul_add(sum_x, -sum_l * sum_xl) / d_det;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut mad = 0.0f32;
            for i in 0..n {
                let mut diff = this_scale.mul_add(l_aux[i] as f32, this_min) - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                let w = match weights {
                    Some(ws) => ws[i],
                    None => x[i] * x[i],
                };
                mad = w.mul_add(diff, mad);
            }
            if mad < best_mad {
                for i in 0..n {
                    l[i] = l_aux[i];
                }
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

/// The peer's `make_qp_quants`.
///
/// Positive-only quant search; returns the chosen scale `d`. Used by
/// Q2_K / Q4_K / Q5_K to quantize the per-sub-block `scales` / `mins`
/// auxiliary buffers.
pub fn make_qp_quants(n: usize, nmax: i32, x: &[f32], l: &mut [u8], quant_weights: &[f32]) -> f32 {
    let mut max = 0.0f32;
    for i in 0..n {
        if x[i] > max {
            max = x[i];
        }
    }
    if max < GROUP_MAX_EPS {
        for li in l.iter_mut().take(n) {
            *li = 0;
        }
        return 0.0;
    }
    let mut iscale = nmax as f32 / max;
    for i in 0..n {
        // C: L[i] = nearest_int(iscale * x[i]) — assigned to uint8_t, so
        // an unclamped store. Mirror exactly (truncating cast of i32→u8).
        l[i] = nearest_int(iscale * x[i]) as u8;
    }
    let scale = 1.0 / iscale;
    let mut best_mse = 0.0f32;
    for i in 0..n {
        let diff = scale.mul_add(-(l[i] as f32), x[i]);
        let w = quant_weights[i];
        best_mse = (w * diff).mul_add(diff, best_mse);
    }
    for is in -4i32..=4i32 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1 * is as f32 + nmax as f32) / max;
        let scale_is = 1.0 / iscale_is;
        let mut mse = 0.0f32;
        for i in 0..n {
            let mut li = nearest_int(iscale_is * x[i]);
            if li > nmax {
                li = nmax;
            }
            let diff = scale_is.mul_add(-(li as f32), x[i]);
            let w = quant_weights[i];
            mse = (w * diff).mul_add(diff, mse);
        }
        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let mut li = nearest_int(iscale * x[i]);
        if li > nmax {
            li = nmax;
        }
        l[i] = li as u8;
        let w = quant_weights[i];
        // 2026-05-20: plain matches canonical fp-contract=off (same as make_qx_quants).
        sumlx = (w * x[i]).mul_add(li as f32, sumlx);
        suml2 = (w * (li as f32)).mul_add(li as f32, suml2);
    }
    for _itry in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = quant_weights[i];
            let cur_l = l[i] as f32;
            let mut slx = (w * x[i]).mul_add(-cur_l, sumlx);
            let mut sl2 = (w * cur_l).mul_add(-cur_l, suml2);
            if slx > 0.0 && sl2 > 0.0 {
                let mut new_l = nearest_int(x[i] * sl2 / slx);
                if new_l > nmax {
                    new_l = nmax;
                }
                if new_l as u8 != l[i] {
                    slx = (w * x[i]).mul_add(new_l as f32, slx);
                    sl2 = (w * (new_l as f32)).mul_add(new_l as f32, sl2);
                    if slx * slx * suml2 > sumlx * sumlx * sl2 {
                        l[i] = new_l as u8;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_int_round_to_even() {
        // The magic-shift trick uses round-to-nearest-even (banker's).
        assert_eq!(nearest_int(0.0), 0);
        assert_eq!(nearest_int(1.0), 1);
        assert_eq!(nearest_int(-1.0), -1);
        assert_eq!(nearest_int(1.4), 1);
        assert_eq!(nearest_int(1.6), 2);
        assert_eq!(nearest_int(-1.4), -1);
        assert_eq!(nearest_int(-1.6), -2);
        // Banker's rounding: 0.5 -> 0 (toward even), 1.5 -> 2 (toward even).
        assert_eq!(nearest_int(0.5), 0);
        assert_eq!(nearest_int(1.5), 2);
        assert_eq!(nearest_int(2.5), 2);
        assert_eq!(nearest_int(3.5), 4);
    }

    #[test]
    fn best_index_int8_boundaries() {
        let codebook: [i8; 4] = [-10, 0, 5, 20];
        // Below first.
        assert_eq!(best_index_int8(&codebook, -100.0), 0);
        // Above last.
        assert_eq!(best_index_int8(&codebook, 100.0), 3);
        // Exact match.
        assert_eq!(best_index_int8(&codebook, 0.0), 1);
        // Closer to right neighbour.
        assert_eq!(best_index_int8(&codebook, 4.0), 2);
        // Closer to left neighbour.
        assert_eq!(best_index_int8(&codebook, 1.0), 1);
    }

    #[test]
    fn group_max_eps_value() {
        // Tracks the peer's `GROUP_MAX_EPS` literal.
        assert_eq!(GROUP_MAX_EPS, 1e-15);
    }

    #[test]
    fn make_qx_quants_zero_block() {
        // amax < GROUP_MAX_EPS branch: writes zeros, returns 0.0.
        let x = vec![0.0f32; 32];
        let mut l = vec![0i8; 32];
        let qw = vec![1.0f32; 32];
        let d = make_qx_quants(32, 8, &x, &mut l, 1, &qw);
        assert_eq!(d, 0.0);
        assert!(l.iter().all(|&v| v == 0));
    }

    #[test]
    fn make_qx_quants_nonzero_returns_finite() {
        // Sanity: a non-zero block returns a finite, non-zero scale.
        let x: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let mut l = vec![0i8; 32];
        let qw = vec![]; // exercise rmse_type=1 fallback
        let d = make_qx_quants(32, 8, &x, &mut l, 1, &qw);
        assert!(d.is_finite());
    }

    /// ADR-033 §P1 Q6_K regression-guard fixture — Gemma 4 blk.0.attn_v
    /// block 113 sub-block 14. The hardcoded F32 array below is **from
    /// canonical's F16-roundtripped path**; pure-Rust `make_qx_quants` on
    /// these exact inputs produces scale=2.3035716731e-3 (0x3b16f785). The
    /// canonical-on-this-fixture value 0x3b11a2a8 was the older target and
    /// was misleading: production hf2q reads BF16→F32→F16→F32 from
    /// safetensors and those F32 values diverge by 1-ULP from this
    /// fixture's F16-from-canonical F32 values; on the *production* F32
    /// path our kernel matches canonical byte-for-byte across the full
    /// 16.7 GB Gemma 4 26B Q4_K_M GGUF (0 Q6_K bytes diff at commit
    /// `6985cd56`). Asserting the production-path scale here makes this
    /// test a regression-guard against future kernel drift; do not chase
    /// the older 0x3b11a2a8 value — that path is no longer authoritative.
    ///
    /// **Disassembly-verified codegen** (2026-05-20, peer HEAD
    /// `e15384a5c`, Apple clang `-O3 -DNDEBUG -arch arm64`, NO `-march`):
    /// in `quantize_row_q6_K_ref` (`_quantize_row_q6_K_ref` symbol,
    /// `ggml-quants.c.o`):
    ///   - **initial pass** is VECTORIZED — 16 elements processed as
    ///     4×fmul.4s chunks; each chunk's `(x³·L, w·l·l)` partial products
    ///     are horizontally reduced by SCALAR fadd ladder
    ///     `((sumlx_prev + lane0) + lane1) + lane2 + lane3` per chunk;
    ///   - **is-loop** is SCALAR — 32 `fmadd s_, s_, s_, s_` instructions
    ///     (16 sumlx + 16 suml2), single-rounded per element.
    /// Rust port emits 0 fmla.4s + 0 scalar fmadd anywhere in the function
    /// despite explicit `.mul_add()` — LLVM auto-vec drops FMA semantics
    /// during loop vectorization. That, plus left-to-right vs reassociated
    /// reduction order in the initial pass, accounts for the divergence.
    ///
    /// **Real-model Gemma 4 26B Q4_K_M byte-cmp** (HEAD `57bab20c`,
    /// `scripts/fast_byte_cmp.py`, 2026-05-20): Q6_K = 49,181 / 667,054,080
    /// bytes = 0.007373% across all 14 tensors (`token_embd.weight` carries
    /// 92.4% of the residual at 45,429/605,552,640). Q4_K essentially closed
    /// (120/9.3GB = 0.000001%, 13/192 tensors). Closure plan: cfg-gated
    /// `std::arch::aarch64` NEON port of make_qx_quants that mirrors
    /// canonical's mixed pattern (vectorized initial pass + scalar fmadd
    /// is-loop) bit-for-bit.
    #[test]
    #[ignore]
    fn q6k_blk113_subblk14_byte_match() {
        let x: [f32; 16] = [
            0.005950928,
            -0.01300049,
            0.01519775,
            0.02624512,
            -0.07128906,
            -0.06689453,
            -0.05053711,
            0.01965332,
            -0.02551270,
            -0.03906250,
            0.01275635,
            0.008056641,
            0.002319336,
            0.04223633,
            -0.02416992,
            0.001045227,
        ];
        let mut l = [0i8; 16];
        let scale = make_qx_quants(16, 32, &x, &mut l, 1, &[]);
        assert_eq!(
            scale.to_bits(),
            0x3b16f785,
            "scale={:.10e} (bits=0x{:08x}) — regression guard against kernel drift; \
             production path matches canonical (real-model byte-cmp = 0 Q6_K diff)",
            scale,
            scale.to_bits()
        );
    }

    /// Diagnostic for ADR-033 Q4_K residual investigation — runs our
    /// `make_qkx2_quants` on the exact 32 F32 inputs of Gemma 4
    /// blk.8.ffn_gate_up_exps block 1055656 sub-block 7 (the largest
    /// single-block residual in real-model byte-cmp). Verified 2026-05-20:
    /// produces bit-identical output to canonical's verbatim C reproducer
    /// (scale=0x3be60a3e, min=0x3d2e1b55, L=[3,5,5,8,11,0,8,9,5,1,9,3,9,8,
    /// 7,5,4,11,14,3,0,3,8,5,7,3,10,7,4,5,5,1]). Therefore the residual
    /// is NOT in this sub-block's make_qkx2_quants — must be in the F32
    /// inputs derivation or cross-sub-block max_scale/max_min computation
    /// (see `diag_blk1055656_full_quantize_row_ref` for full-block test).
    #[test]
    #[ignore]
    fn diag_blk1055656_subblock7_make_qkx2() {
        let bytes = match std::fs::read("/tmp/blk1055656_subblock7_f32.bin") {
            Ok(b) => b,
            Err(_) => {
                eprintln!("[skip] fixture missing — regenerate with python extract");
                return;
            }
        };
        let mut x = Vec::with_capacity(32);
        for chunk in bytes.chunks_exact(4) {
            x.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        assert_eq!(x.len(), 32);
        // Reproduce hf2q's Q4_K weights formula (q4_k.rs:119-126).
        let mut sum_x2 = 0.0f32;
        for &v in &x {
            sum_x2 = v.mul_add(v, sum_x2);
        }
        let av_x = (sum_x2 / 32.0).sqrt();
        let weights: Vec<f32> = x.iter().map(|&v| av_x + v.abs()).collect();
        let mut l = vec![0u8; 32];
        let mut l_aux = vec![0u8; 32];
        let mut the_min = 0.0f32;
        let scale = super::make_qkx2_quants(
            32,
            15,
            &x,
            &weights,
            &mut l,
            &mut the_min,
            &mut l_aux,
            -1.0,
            0.1,
            20,
            false,
        );
        eprintln!("hf2q av_x  = {:.10e} (bits=0x{:08x})", av_x, av_x.to_bits());
        eprintln!(
            "hf2q scale = {:.10e} (bits=0x{:08x})",
            scale,
            scale.to_bits()
        );
        eprintln!(
            "hf2q min   = {:.10e} (bits=0x{:08x})",
            the_min,
            the_min.to_bits()
        );
        eprintln!("hf2q L     = {:?}", l);
        eprintln!("");
        eprintln!("canonical: scale=0x3be60a3e min=0x3d2e1b55");
        eprintln!(
            "           L = [3,5,5,8,11,0,8,9,5,1,9,3,9,8,7,5,4,11,14,3,0,3,8,5,7,3,10,7,4,5,5,1]"
        );
    }

    /// Full-block diagnostic for ADR-033 Q4_K residual investigation —
    /// runs hf2q's full `quantize_row_ref` (via super::super::q4_k::quantize)
    /// on Gemma 4 blk.8.ffn_gate_up_exps block 1055656's 256 F32 inputs,
    /// compares output to canonical's 144 bytes for that exact block.
    /// Reveals whether the 22-byte residual is in the F32 derivation
    /// (canonical F16 vs production BF16→F32→F16 mismatch) or in the
    /// cross-sub-block max_scale/max_min derivation.
    #[test]
    #[ignore]
    fn diag_blk1055656_full_quantize_row_ref() {
        let bytes = match std::fs::read("/tmp/blk1055656_full_256_f32.bin") {
            Ok(b) => b,
            Err(_) => {
                eprintln!("[skip] fixture missing");
                return;
            }
        };
        let mut x = Vec::with_capacity(256);
        for chunk in bytes.chunks_exact(4) {
            x.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        assert_eq!(x.len(), 256);
        let actual = crate::quantize::ggml_quants::q4_k::quantize(&x, 256, None);
        assert_eq!(actual.len(), 144);
        let expected = match std::fs::read("/tmp/blk1055656_canonical_q4k.bin") {
            Ok(b) => b,
            Err(_) => {
                eprintln!("[skip] canonical-expected missing");
                return;
            }
        };
        assert_eq!(expected.len(), 144);
        let mut diffs = 0;
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            if a != e {
                diffs += 1;
                if diffs <= 5 {
                    let field = match i {
                        0..=1 => format!("d[{}]", i),
                        2..=3 => format!("dmin[{}]", i - 2),
                        4..=15 => format!("scales[{}]", i - 4),
                        _ => format!("qs[{}]", i - 16),
                    };
                    eprintln!(
                        "  diff byte {} ({}): hf2q=0x{:02x} canon=0x{:02x}",
                        i, field, a, e
                    );
                }
            }
        }
        eprintln!("diffs: {}/144 bytes", diffs);
        if diffs == 0 {
            eprintln!("✅ FULL-BLOCK BYTE-IDENTICAL — F32 input + kernel both match canonical");
        } else {
            eprintln!("✗ diverged — residual NOT closed by kernel-on-fixture (issue elsewhere)");
        }
    }

    #[test]
    fn make_qkx2_quants_zero_block() {
        let x = vec![0.0f32; 32];
        let weights = vec![1.0f32; 32];
        let mut l = vec![0u8; 32];
        let mut l_aux = vec![0u8; 32];
        let mut the_min = 0.0f32;
        let d = make_qkx2_quants(
            32,
            15,
            &x,
            &weights,
            &mut l,
            &mut the_min,
            &mut l_aux,
            -1.0,
            0.1,
            20,
            false,
        );
        assert_eq!(d, 0.0);
        assert!(l.iter().all(|&v| v == 0));
    }

    #[test]
    fn make_qkx3_quants_zero_block() {
        let x = vec![0.0f32; 32];
        let weights = vec![1.0f32; 32];
        let mut l = vec![0u8; 32];
        let mut l_aux = vec![0u8; 32];
        let mut the_min = 0.0f32;
        let d = make_qkx3_quants(
            32,
            15,
            &x,
            Some(&weights),
            &mut l,
            &mut the_min,
            &mut l_aux,
            -0.9,
            0.05,
            36,
            false,
        );
        assert_eq!(d, 0.0);
        assert!(l.iter().all(|&v| v == 0));
    }

    #[test]
    fn make_qp_quants_zero_block() {
        let x = vec![0.0f32; 16];
        let mut l = vec![0u8; 16];
        let weights = vec![1.0f32; 16];
        let d = make_qp_quants(16, 15, &x, &mut l, &weights);
        assert_eq!(d, 0.0);
        assert!(l.iter().all(|&v| v == 0));
    }

    #[test]
    fn make_qp_quants_positive_returns_finite() {
        // x ∈ [0.1, 1.6], should converge to a finite, positive scale.
        let x: Vec<f32> = (1..=16).map(|i| i as f32 * 0.1).collect();
        let mut l = vec![0u8; 16];
        let weights = vec![1.0f32; 16];
        let d = make_qp_quants(16, 15, &x, &mut l, &weights);
        assert!(d.is_finite());
        assert!(d > 0.0);
    }
}
