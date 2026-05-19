//! Shared helper functions used across the ggml-quants kernel ports.
//!
//! Each helper here is a 1:1 pure-Rust port of a primitive from
//! `/opt/llama.cpp/ggml/src/ggml-quants.c`. Consolidating them into a
//! single source eliminates the latent-divergence bug class flagged by
//! codex at `0bd0e7eb` (parallel kernel-port workers each ported their
//! own copies; subtle finalize-block / d==0 differences crept in).
//!
//! The byte-cmp gates in each per-type submodule (`q*_*::tests::byte_cmp*`)
//! cover this file transitively — any drift here surfaces immediately
//! against the llama.cpp `ggml_quantize_chunk` reference fixtures.

/// `GROUP_MAX_EPS` at `ggml-quants.c:16`. Threshold below which an
/// absolute-max scan is treated as the zero-block case.
pub const GROUP_MAX_EPS: f32 = 1e-15;

/// `nearest_int` at `ggml-quants.c:559`.
///
/// The C version uses a bit-cast trick that rounds via the float-to-int
/// hardware mode (RNE). We mirror the bitcast exactly so the rounding
/// boundary behaviour is identical to llama.cpp.
#[inline(always)]
pub fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4_194_303.0);
    let val = fval + 12_582_912.0;
    let i = val.to_bits() as i32;
    (i & 0x007f_ffff) - 0x0040_0000
}

/// `best_index_int8` at `ggml-quants.c:24-33` — bisection over a
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

/// `make_qx_quants` at `ggml-quants.c:566`.
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
        sumlx += w * x[i] * (li as f32);
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
            sumlx += w * x[i] * (li as f32);
            suml2 += w * (li as f32) * (li as f32);
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

/// `make_qkx2_quants` — `ggml-quants.c:737-816`.
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
        sum_x += w * x[i];
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
        let mut diff = scale * li_c as f32 + min - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
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
            sum_l += w * li_f;
            sum_l2 += w * li_f * li_f;
            sum_xl += w * li_f * x[i];
        }
        let d_det = sum_w * sum_l2 - sum_l * sum_l;
        if d_det > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d_det;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d_det;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0.0f32;
            for i in 0..n {
                let mut diff = this_scale * l_aux[i] as f32 + this_min - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                cur_error += weights[i] * diff;
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

/// `make_qkx3_quants` — `ggml-quants.c:931-1012`.
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
        sum_x += w * x[i];
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
        let mut diff = scale * li_c as f32 + min - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        let w = match weights {
            Some(ws) => ws[i],
            None => x[i] * x[i],
        };
        best_mad += w * diff;
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
            sum_l += w * li_f;
            sum_l2 += w * li_f * li_f;
            sum_xl += w * li_f * x[i];
        }
        let d_det = sum_w * sum_l2 - sum_l * sum_l;
        if d_det > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d_det;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d_det;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut mad = 0.0f32;
            for i in 0..n {
                let mut diff = this_scale * l_aux[i] as f32 + this_min - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                let w = match weights {
                    Some(ws) => ws[i],
                    None => x[i] * x[i],
                };
                mad += w * diff;
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

/// `make_qp_quants` — `ggml-quants.c:1014-1085`.
///
/// Positive-only quant search; returns the chosen scale `d`. Used by
/// Q2_K / Q4_K / Q5_K to quantize the per-sub-block `scales` / `mins`
/// auxiliary buffers.
pub fn make_qp_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    l: &mut [u8],
    quant_weights: &[f32],
) -> f32 {
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
        let diff = x[i] - scale * l[i] as f32;
        let w = quant_weights[i];
        best_mse += w * diff * diff;
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
            let diff = x[i] - scale_is * li as f32;
            let w = quant_weights[i];
            mse += w * diff * diff;
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
        sumlx += w * x[i] * li as f32;
        suml2 += w * (li as f32) * (li as f32);
    }
    for _itry in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = quant_weights[i];
            let cur_l = l[i] as f32;
            let mut slx = sumlx - w * x[i] * cur_l;
            let mut sl2 = suml2 - w * cur_l * cur_l;
            if slx > 0.0 && sl2 > 0.0 {
                let mut new_l = nearest_int(x[i] * sl2 / slx);
                if new_l > nmax {
                    new_l = nmax;
                }
                if new_l as u8 != l[i] {
                    slx += w * x[i] * new_l as f32;
                    sl2 += w * (new_l as f32) * (new_l as f32);
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
        // Tracks `ggml-quants.c:16` literal.
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

    #[test]
    fn make_qkx2_quants_zero_block() {
        let x = vec![0.0f32; 32];
        let weights = vec![1.0f32; 32];
        let mut l = vec![0u8; 32];
        let mut l_aux = vec![0u8; 32];
        let mut the_min = 0.0f32;
        let d = make_qkx2_quants(32, 15, &x, &weights, &mut l, &mut the_min, &mut l_aux, -1.0, 0.1, 20, false);
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
            32, 15, &x, Some(&weights), &mut l, &mut the_min, &mut l_aux,
            -0.9, 0.05, 36, false,
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
