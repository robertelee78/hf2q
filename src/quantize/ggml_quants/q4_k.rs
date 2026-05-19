//! Q4_K quantizer — ADR-033 P0 pure-Rust port of
//! `quantize_row_q4_K_ref` (`/opt/llama.cpp/ggml/src/ggml-quants.c:1395`),
//! `quantize_row_q4_K_impl` (`.../ggml-quants.c:1491`), and the dispatcher
//! `quantize_q4_K` (`.../ggml-quants.c:1564`).
//!
//! Block layout from `ggml-common.h:319-328`:
//! ```text
//! #define QK_K 256
//! #define K_SCALE_SIZE 12
//! typedef struct {
//!     union { struct { ggml_half d; ggml_half dmin; }; ggml_half2 dm; };
//!     uint8_t scales[K_SCALE_SIZE]; // 8 sub-block scales + 8 sub-block mins,
//!                                   // each 6-bit, packed via the q3/q4/q5_K
//!                                   // shared encoding (see get_scale_min_k4
//!                                   // at ggml-quants.c:818-825).
//!     uint8_t qs[QK_K/2];           // 128 nibble bytes
//! } block_q4_K;                     // sizeof == 2*2 + 12 + 128 == 144
//! ```
//!
//! Helpers (ported byte-for-byte from llama.cpp):
//! * `nearest_int` — `ggml-quants.c:559`
//! * `make_qkx2_quants` — `ggml-quants.c:737-816` (used by `_ref`)
//! * `make_qkx3_quants` — `ggml-quants.c:931-1012` (used by `_impl`)
//! * `make_qp_quants` — `ggml-quants.c:1014-1085` (used by `_impl` for
//!   the super-block-level d / m positive quantization to 6 bits)
//!
//! 6-bit scale packing (8 sub-block scales `ls[0..8]` + 8 mins `lm[0..8]`)
//! lives in `y[i].scales[0..12]` per the ref/impl loops at C:1427-1439 and
//! C:1526-1536. Bytes 0..3 hold the low 6 bits of `ls[0..4]`; bytes 4..7
//! hold the low 6 bits of `lm[0..4]`; bytes 8..11 hold the low 4 bits of
//! `ls[4..8]` (low nibble) and `lm[4..8]` (high nibble), with the high 2
//! bits of `ls[4..8]` and `lm[4..8]` OR'd into the top 2 bits of bytes
//! 0..3 and 4..7 respectively.

use half::f16;

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;
pub const BLOCK_BYTES: usize = 2 + 2 + K_SCALE_SIZE + QK_K / 2; // 144

const GROUP_MAX_EPS: f32 = 1e-15;

/// `nearest_int` — `ggml-quants.c:559`.
#[inline(always)]
fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4_194_303.0);
    let val = fval + 12_582_912.0;
    let i = val.to_bits();
    (i & 0x007f_ffff) as i32 - 0x0040_0000
}

/// `make_qkx2_quants` — `ggml-quants.c:737-816`.
#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(
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
/// Q4_K's `_impl` always passes a non-null weights slice; the weight-optional
/// branches are preserved for fidelity with the C source.
#[allow(clippy::too_many_arguments)]
fn make_qkx3_quants(
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
fn make_qp_quants(n: usize, nmax: i32, x: &[f32], l: &mut [u8], quant_weights: &[f32]) -> f32 {
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

/// Quantize an F32 buffer to Q4_K bytes.
///
/// Mirrors dispatcher `quantize_q4_K` at `ggml-quants.c:1564`. When
/// `imatrix` is `Some`, the `_impl` path runs per row with `quant_weights`
/// aliased to the same row-length slice each iteration (the C dispatcher
/// reuses the same `quant_weights` pointer for every row).
pub fn quantize(src: &[f32], n_per_row: usize, imatrix: Option<&[f32]>) -> Vec<u8> {
    assert!(
        n_per_row % QK_K == 0,
        "n_per_row {} not multiple of QK_K {}",
        n_per_row,
        QK_K
    );
    assert!(
        src.len() % n_per_row == 0,
        "src len {} not multiple of n_per_row {}",
        src.len(),
        n_per_row
    );
    if let Some(qw) = imatrix {
        assert_eq!(
            qw.len(),
            n_per_row,
            "imatrix len {} must equal n_per_row {} (dispatcher reuses pointer per row)",
            qw.len(),
            n_per_row
        );
    }

    let n_rows = src.len() / n_per_row;
    let row_blocks = n_per_row / QK_K;
    let mut out = Vec::with_capacity(n_rows * row_blocks * BLOCK_BYTES);

    for row in 0..n_rows {
        let row_x = &src[row * n_per_row..(row + 1) * n_per_row];
        match imatrix {
            None => quantize_row_ref(row_x, &mut out),
            Some(qw) => quantize_row_impl(row_x, qw, &mut out),
        }
    }

    out
}

/// `quantize_row_q4_K_ref` — `ggml-quants.c:1395-1465`.
fn quantize_row_ref(x: &[f32], out: &mut Vec<u8>) {
    debug_assert_eq!(x.len() % QK_K, 0);
    let nb = x.len() / QK_K;

    let mut l_arr = [0u8; QK_K];
    let mut l_aux = [0u8; 32];
    let mut weights = [0.0f32; 32];
    let mut mins = [0.0f32; QK_K / 32];
    let mut scales = [0.0f32; QK_K / 32];

    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut max_scale = 0.0f32;
        let mut max_min = 0.0f32;
        for j in 0..QK_K / 32 {
            let sub = &xb[32 * j..32 * (j + 1)];
            let mut sum_x2 = 0.0f32;
            for &v in sub {
                sum_x2 += v * v;
            }
            let av_x = (sum_x2 / 32.0).sqrt();
            for l in 0..32 {
                weights[l] = av_x + sub[l].abs();
            }
            let (l_chunk, _) = l_arr[32 * j..].split_at_mut(32);
            scales[j] = make_qkx2_quants(
                32,
                15,
                sub,
                &weights,
                l_chunk,
                &mut mins[j],
                &mut l_aux,
                -1.0,
                0.1,
                20,
                false,
            );
            let scale = scales[j];
            if scale > max_scale {
                max_scale = scale;
            }
            let min = mins[j];
            if min > max_min {
                max_min = min;
            }
        }

        let inv_scale = if max_scale > 0.0 {
            63.0 / max_scale
        } else {
            0.0
        };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

        let mut scales_packed = [0u8; K_SCALE_SIZE];
        for j in 0..QK_K / 32 {
            // C: uint8_t ls = nearest_int(inv_scale*scales[j]);
            //    ls = MIN(63, ls);
            // Truncating cast i32 -> u8 then MIN(63, ls) — but ls is u8 so
            // we keep the value in u8 throughout.
            let ls_i = nearest_int(inv_scale * scales[j]);
            let lm_i = nearest_int(inv_min * mins[j]);
            let mut ls = ls_i as u8;
            let mut lm = lm_i as u8;
            if ls > 63 {
                ls = 63;
            }
            if lm > 63 {
                lm = 63;
            }
            if j < 4 {
                scales_packed[j] = ls;
                scales_packed[j + 4] = lm;
            } else {
                scales_packed[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                scales_packed[j - 4] |= (ls >> 4) << 6;
                scales_packed[j] |= (lm >> 4) << 6;
            }
        }

        let d_blk = max_scale / 63.0;
        let dmin_blk = max_min / 63.0;
        let d_f16 = f16::from_f32(d_blk);
        let dmin_f16 = f16::from_f32(dmin_blk);

        // Second-stage L[] recomputation uses the just-stored f16 d/dmin,
        // so we convert back through f16 to match C bit-for-bit.
        let d_dq = d_f16.to_f32();
        let dmin_dq = dmin_f16.to_f32();
        for j in 0..QK_K / 32 {
            let (sc, m) = get_scale_min_k4(j, &scales_packed);
            let d = d_dq * sc as f32;
            if d == 0.0 {
                continue;
            }
            let dm = dmin_dq * m as f32;
            for ii in 0..32 {
                let l_raw = nearest_int((xb[32 * j + ii] + dm) / d);
                let li = l_raw.max(0).min(15) as u8;
                l_arr[32 * j + ii] = li;
            }
        }

        // Write block: d (2B) + dmin (2B) + scales (12B) + qs (128B).
        out.extend_from_slice(&d_f16.to_le_bytes());
        out.extend_from_slice(&dmin_f16.to_le_bytes());
        out.extend_from_slice(&scales_packed);

        // qs: per the C loop `for (j = 0; j < QK_K; j += 64)`, each 64-byte
        // group of L[] becomes 32 nibble bytes packing low=L[j+l],
        // high=L[j+l+32].
        let mut j = 0;
        while j < QK_K {
            for l in 0..32 {
                let lo = l_arr[j + l];
                let hi = l_arr[j + l + 32];
                out.push(lo | (hi << 4));
            }
            j += 64;
        }
    }
}

/// `quantize_row_q4_K_impl` — `ggml-quants.c:1491-1562`.
fn quantize_row_impl(x: &[f32], quant_weights: &[f32], out: &mut Vec<u8>) {
    debug_assert_eq!(x.len() % QK_K, 0);
    debug_assert_eq!(quant_weights.len(), x.len());

    let nb = x.len() / QK_K;

    let mut l_arr = [0u8; QK_K];
    let mut l_aux = [0u8; 32];
    let mut ls_arr = [0u8; QK_K / 32];
    let mut lm_arr = [0u8; QK_K / 32];
    let mut weights = [0.0f32; 32];
    let mut sw = [0.0f32; QK_K / 32];
    let mut mins = [0.0f32; QK_K / 32];
    let mut scales = [0.0f32; QK_K / 32];

    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut sum_x2 = 0.0f32;
        for &v in xb {
            sum_x2 += v * v;
        }
        let sigma2 = 2.0 * sum_x2 / QK_K as f32;
        let _av_x = sigma2.sqrt();

        for j in 0..QK_K / 32 {
            let sub = &xb[32 * j..32 * (j + 1)];
            // C: `quant_weights + QK_K*i + 32*j` — the row-level pointer
            // dispatcher passes is the WHOLE row, indexed by block-i.
            let qw = &quant_weights[QK_K * i + 32 * j..QK_K * i + 32 * (j + 1)];
            for l in 0..32 {
                weights[l] = qw[l] * (sigma2 + sub[l] * sub[l]).sqrt();
            }
            let mut sumw = 0.0f32;
            for &w in &weights {
                sumw += w;
            }
            sw[j] = sumw;
            let (l_chunk, _) = l_arr[32 * j..].split_at_mut(32);
            scales[j] = make_qkx3_quants(
                32,
                15,
                sub,
                Some(&weights),
                l_chunk,
                &mut mins[j],
                &mut l_aux,
                -0.9,
                0.05,
                36,
                false,
            );
        }

        let d_block = make_qp_quants(QK_K / 32, 63, &scales, &mut ls_arr, &sw);
        let m_block = make_qp_quants(QK_K / 32, 63, &mins, &mut lm_arr, &sw);

        let mut scales_packed = [0u8; K_SCALE_SIZE];
        for j in 0..QK_K / 32 {
            let ls = ls_arr[j];
            let lm = lm_arr[j];
            if j < 4 {
                scales_packed[j] = ls;
                scales_packed[j + 4] = lm;
            } else {
                scales_packed[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                scales_packed[j - 4] |= (ls >> 4) << 6;
                scales_packed[j] |= (lm >> 4) << 6;
            }
        }

        let d_f16 = f16::from_f32(d_block);
        let dmin_f16 = f16::from_f32(m_block);
        let d_dq = d_f16.to_f32();
        let dmin_dq = dmin_f16.to_f32();

        for j in 0..QK_K / 32 {
            let (sc, m) = get_scale_min_k4(j, &scales_packed);
            let d = d_dq * sc as f32;
            if d == 0.0 {
                continue;
            }
            let dm = dmin_dq * m as f32;
            for ii in 0..32 {
                let l_raw = nearest_int((xb[32 * j + ii] + dm) / d);
                let li = l_raw.max(0).min(15) as u8;
                l_arr[32 * j + ii] = li;
            }
        }

        out.extend_from_slice(&d_f16.to_le_bytes());
        out.extend_from_slice(&dmin_f16.to_le_bytes());
        out.extend_from_slice(&scales_packed);

        let mut j = 0;
        while j < QK_K {
            for l in 0..32 {
                let lo = l_arr[j + l];
                let hi = l_arr[j + l + 32];
                out.push(lo | (hi << 4));
            }
            j += 64;
        }
    }
}

/// `get_scale_min_k4` — `ggml-quants.c:818-825`. Unpacks the (d, m) pair
/// for sub-block `j` from the 12-byte packed `scales` array.
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8; K_SCALE_SIZE]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn fixture_path(name: &str) -> PathBuf {
        let manifest = std::env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR not set by cargo test");
        PathBuf::from(manifest)
            .join("tests/fixtures/ggml_quants")
            .join(name)
    }

    fn read_f32s(name: &str) -> Vec<f32> {
        let bytes = fs::read(fixture_path(name)).expect("read fixture");
        assert!(
            bytes.len() % 4 == 0,
            "fixture {} not a multiple of 4 bytes",
            name
        );
        let mut out = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        out
    }

    fn read_bytes(name: &str) -> Vec<u8> {
        fs::read(fixture_path(name)).expect("read fixture")
    }

    fn make_imatrix(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_add(0x6D2B79F5);
                let mut t = state;
                t = (t ^ (t >> 15)).wrapping_mul(t | 1);
                t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
                let u = t ^ (t >> 14);
                let v = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
                v.abs() + 1e-3
            })
            .collect()
    }

    #[test]
    fn byte_cmp_noim() {
        let input = read_f32s("q4_k_512_noim_input.bin");
        let expected = read_bytes("q4_k_512_noim_expected.bin");
        let got = quantize(&input, 512, None);
        assert_eq!(got.len(), expected.len(), "Q4_K noim length mismatch");
        assert_eq!(got, expected, "Q4_K noim byte-cmp failed");
    }

    #[test]
    fn byte_cmp_im() {
        let input = read_f32s("q4_k_512_im_input.bin");
        let expected = read_bytes("q4_k_512_im_expected.bin");
        let imatrix = make_imatrix(512, 2);
        let got = quantize(&input, 512, Some(&imatrix));
        assert_eq!(got.len(), expected.len(), "Q4_K im length mismatch");
        assert_eq!(got, expected, "Q4_K im byte-cmp failed");
    }
}
