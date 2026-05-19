//! Q5_K quantizer — ADR-033 P0 pure-Rust port of
//! `quantize_row_q5_K_ref`   at `/opt/llama.cpp/ggml/src/ggml-quants.c:1582`
//! `quantize_row_q5_K_impl`  at `/opt/llama.cpp/ggml/src/ggml-quants.c:1696`
//! `quantize_q5_K`           at `/opt/llama.cpp/ggml/src/ggml-quants.c:1789`
//! (SHA pinned in `data/llama_cpp_pin.txt`).
//!
//! Block layout from `ggml-common.h:338-345`:
//! ```text
//! #define QK_K 256
//! #define K_SCALE_SIZE 12
//! typedef struct {
//!     ggml_half d;                 // 2 bytes, super-block scale (F16 LE)
//!     ggml_half dmin;              // 2 bytes, super-block min   (F16 LE)
//!     uint8_t scales[K_SCALE_SIZE];// 12 bytes, 6-bit packed sub-scales/mins
//!     uint8_t qh[QK_K/8];          // 32 bytes, high bits of 5-bit quants
//!     uint8_t qs[QK_K/2];          // 128 bytes, low 4 bits of 5-bit quants
//! } block_q5_K;                    // 176 bytes total
//! ```
//!
//! 5-bit asymmetric K-quant, super-block of 256, 8 sub-blocks of 32.
//! Two dispatch paths:
//! - `imatrix == None`  → `_ref`: per-sub-block `make_qkx2_quants`
//!   (rmin=-0.5, rdelta=0.1, nstep=15) with `weights[l] = av_x + |x|`.
//! - `imatrix == Some`  → `_impl`: per-sub-block `make_qkx3_quants`
//!   (rmin=-0.9, rdelta=0.05, nstep=36) → block-level `make_qp_quants`
//!   to consolidate the 8 sub-scales/mins into a single block scale.
//!
//! Byte-parity gotchas:
//! - `nearest_int` uses the 12582912.f magic-shift round-half-to-even.
//! - `mins[j]` is stored POSITIVE: both `make_qkx2/3_quants` write
//!   `*the_min = -min` (with C-local `min ≤ 0`), so the returned scalar
//!   is ≥ 0 and feeds directly into the unsigned `make_qp_quants`.
//! - Sub-scale packing for j∈[0,4): `scales[j]=ls; scales[j+4]=lm`
//!   stores the full 6 bits. For j∈[4,8): low 4 bits go into
//!   `scales[j+4]` (ls in nibble lo, lm in nibble hi); high 2 bits of
//!   ls go into bits 6-7 of `scales[j-4]`, high 2 bits of lm into bits
//!   6-7 of `scales[j-0]` (overwriting the upper 2 bits of the LS/LM
//!   bytes for j-4 and j-0 respectively, which were `ls`/`lm` of those
//!   lower-j sub-blocks but only the low 6 bits matter there).
//! - The re-encode loop in BOTH paths goes through `get_scale_min_k4`
//!   to recover the packed 6-bit sub-scale/sub-min and quantize the
//!   final L values, NOT through the originally-returned `scales[]`/
//!   `mins[]` — the F16 round-trip of d/dmin is part of the bit-exact
//!   spec.
//! - `qh` packing: bit `m1 << (n/32)` of `qh[j]` carries the high bit
//!   of the j-th element of each 64-element chunk; `m2` carries the
//!   high bit of element `j+32` of that chunk. m1,m2 start at 1,2 and
//!   shift `<<= 2` for each of the 4 chunks (n=0,64,128,192) — so qh
//!   bits 0/1, 2/3, 4/5, 6/7 cover the four chunks.

use half::f16;

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;
pub const BLOCK_BYTES: usize = 2 + 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2; // 176

const GROUP_MAX_EPS: f32 = 1e-15;

/// Replicates llama.cpp's `nearest_int` (ggml-quants.c:559) — round-half-to-even
/// via the 12582912.f magic shift. Must be bit-exact for byte parity.
#[inline]
fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4194303.0);
    let val = fval + 12582912.0;
    let i = val.to_bits() as i32;
    (i & 0x007fffff) - 0x00400000
}

/// Pure-Rust port of `make_qkx2_quants` (ggml-quants.c:737-816).
/// Returns `(scale, the_min)` where `the_min == -min` (the C call site
/// receives `*the_min = -min` written at the function end, and the
/// post-solver C-local `min` is clamped to ≤ 0, so `the_min ≥ 0`).
fn make_qkx2_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l_out: &mut [u8],
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> (f32, f32) {
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
    if max == min {
        for i in 0..n {
            l_out[i] = 0;
        }
        return (0.0, -min);
    }
    let mut iscale = (nmax as f32) / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..n {
        let l = nearest_int(iscale * (x[i] - min));
        l_out[i] = l.max(0).min(nmax) as u8;
        let mut diff = scale * (l_out[i] as f32) + min - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        let w = weights[i];
        best_error += w * diff;
    }
    if nstep < 1 {
        return (scale, -min);
    }
    for is in 0..=nstep {
        let iscale_try = (rmin + rdelta * (is as f32) + (nmax as f32)) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let l = nearest_int(iscale_try * (x[i] - min));
            let l = l.max(0).min(nmax);
            laux[i] = l as u8;
            let w = weights[i];
            let lf = l as f32;
            sum_l += w * lf;
            sum_l2 += w * lf * lf;
            sum_xl += w * lf * x[i];
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
                let mut diff = this_scale * (laux[i] as f32) + this_min - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                let w = weights[i];
                cur_error += w * diff;
            }
            if cur_error < best_error {
                for i in 0..n {
                    l_out[i] = laux[i];
                }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
            // Silence unused-assignment lint; C reassigns `iscale` here.
            let _ = iscale;
            iscale = iscale_try;
        }
    }
    (scale, -min)
}

/// Pure-Rust port of `make_qkx3_quants` (ggml-quants.c:931-1012).
/// Same return convention as `make_qkx2_quants`: `(scale, -min)`.
/// Diverges from qkx2 by: (a) initial branch uses `max <= min` (vs
/// `max == min`), (b) initial-pass clamping mirrors the inner-loop
/// clamp `[0, nmax]`, (c) returns immediately if `nstep < 1` after
/// the initial pass.
fn make_qkx3_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l_out: &mut [u8],
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> (f32, f32) {
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
    if max <= min {
        for i in 0..n {
            l_out[i] = 0;
        }
        return (0.0, -min);
    }
    let mut iscale = (nmax as f32) / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_mad = 0.0f32;
    for i in 0..n {
        let l = nearest_int(iscale * (x[i] - min));
        l_out[i] = l.max(0).min(nmax) as u8;
        let mut diff = scale * (l_out[i] as f32) + min - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        let w = weights[i];
        best_mad += w * diff;
    }
    if nstep < 1 {
        return (scale, -min);
    }
    for is in 0..=nstep {
        let iscale_try = (rmin + rdelta * (is as f32) + (nmax as f32)) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let l = nearest_int(iscale_try * (x[i] - min));
            let l = l.max(0).min(nmax);
            laux[i] = l as u8;
            let w = weights[i];
            let lf = l as f32;
            sum_l += w * lf;
            sum_l2 += w * lf * lf;
            sum_xl += w * lf * x[i];
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
                let mut diff = this_scale * (laux[i] as f32) + this_min - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                let w = weights[i];
                mad += w * diff;
            }
            if mad < best_mad {
                for i in 0..n {
                    l_out[i] = laux[i];
                }
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
            let _ = iscale;
            iscale = iscale_try;
        }
    }
    (scale, -min)
}

/// Pure-Rust port of `make_qp_quants` (ggml-quants.c:1014-1085).
/// Unsigned positive-quant solver — operates on a tiny n=8 vector (the
/// sub-scales or sub-mins of a Q5_K super-block) with `nmax=63`.
fn make_qp_quants(n: usize, nmax: i32, x: &[f32], l_out: &mut [u8], quant_weights: &[f32]) -> f32 {
    let mut max = 0.0f32;
    for i in 0..n {
        if x[i] > max {
            max = x[i];
        }
    }
    if max < GROUP_MAX_EPS {
        for i in 0..n {
            l_out[i] = 0;
        }
        return 0.0;
    }
    let mut iscale = (nmax as f32) / max;
    for i in 0..n {
        // C: `L[i] = nearest_int(iscale * x[i])` — note: NO clamp here in
        // the initial pass, matching the C source exactly. The cast to
        // uint8_t is implicit; `nearest_int` returns i32 and the cast
        // wraps to 0..=255. For our domain (`x ≥ 0`, `iscale*x ≤ nmax=63`)
        // the value fits in u8 without wrapping.
        l_out[i] = nearest_int(iscale * x[i]) as u8;
    }
    let scale = 1.0 / iscale;
    let mut best_mse = 0.0f32;
    for i in 0..n {
        let diff = x[i] - scale * (l_out[i] as f32);
        let w = quant_weights[i];
        best_mse += w * diff * diff;
    }
    for is in -4..=4i32 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1 * (is as f32) + (nmax as f32)) / max;
        let scale_is = 1.0 / iscale_is;
        let mut mse = 0.0f32;
        for i in 0..n {
            let l = nearest_int(iscale_is * x[i]);
            let l = l.min(nmax);
            let diff = x[i] - scale_is * (l as f32);
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
        let mut l = nearest_int(iscale * x[i]);
        l = l.min(nmax);
        l_out[i] = l as u8;
        let w = quant_weights[i];
        sumlx += w * x[i] * (l as f32);
        suml2 += w * (l as f32) * (l as f32);
    }
    for _itry in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = quant_weights[i];
            let li = l_out[i] as f32;
            let mut slx = sumlx - w * x[i] * li;
            let mut sl2 = suml2 - w * li * li;
            if slx > 0.0 && sl2 > 0.0 {
                let new_l = nearest_int(x[i] * sl2 / slx);
                let new_l = new_l.min(nmax);
                if new_l != l_out[i] as i32 {
                    slx += w * x[i] * (new_l as f32);
                    sl2 += w * (new_l as f32) * (new_l as f32);
                    if slx * slx * suml2 > sumlx * sumlx * sl2 {
                        l_out[i] = new_l as u8;
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

/// Mirror of `get_scale_min_k4` (ggml-quants.c:818-825) — unpacks the
/// 6-bit sub-scale and sub-min for sub-block `j` from the 12-byte
/// `scales` array.
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Pack the 6-bit sub-scale `ls` and sub-min `lm` for sub-block `j`
/// into the 12-byte `scales` array. Mirrors the C inline packing at
/// lines 1619-1626 (ref) and 1738-1745 (impl).
#[inline]
fn pack_scale_min_k4(j: usize, scales: &mut [u8; K_SCALE_SIZE], ls: u8, lm: u8) {
    if j < 4 {
        scales[j] = ls;
        scales[j + 4] = lm;
    } else {
        scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
        scales[j - 4] |= (ls >> 4) << 6;
        scales[j] |= (lm >> 4) << 6;
    }
}

/// Common tail: given final `d`, `dmin`, `mins[]`, `scales[]`, re-encode
/// the row into L[QK_K] via `get_scale_min_k4` (after F16 round-trip),
/// then pack into `qh[32]` + `ql[128]`. Mirrors the C re-encode loop
/// shared by `_ref` (lines 1631-1663) and `_impl` (lines 1750-1782).
fn finalize_block(
    x: &[f32],
    d_f16: f16,
    dmin_f16: f16,
    scales: &[u8; K_SCALE_SIZE],
    l_init: &[u8; QK_K],
    out: &mut Vec<u8>,
) {
    let d_back = f16::to_f32(d_f16);
    let dmin_back = f16::to_f32(dmin_f16);

    // C preserves the prior L[] (populated by the per-sub-block
    // make_qkx2/make_qkx3 + make_qp passes) for sub-blocks where the
    // reconstructed `d == 0` (`continue;` at ggml-quants.c:1631-1663
    // and :1750-1782). Codex review of commit 0bd0e7eb (2026-05-18)
    // flagged that starting from zeros diverges when any sub-block hits
    // sc==0 or f16(d)==0. Initialize from the caller's pre-populated
    // L[].
    let mut l: [u8; QK_K] = *l_init;
    for j in 0..QK_K / 32 {
        let (sc, m) = get_scale_min_k4(j, scales);
        let d = d_back * (sc as f32);
        if d == 0.0 {
            continue;
        }
        let dm = dmin_back * (m as f32);
        for ii in 0..32 {
            let li = nearest_int((x[32 * j + ii] + dm) / d);
            let clamped = li.max(0).min(31);
            l[32 * j + ii] = clamped as u8;
        }
    }

    // Emit header: d, dmin, scales[12]
    out.extend_from_slice(&d_f16.to_le_bytes());
    out.extend_from_slice(&dmin_f16.to_le_bytes());
    out.extend_from_slice(scales);

    // qh[32]
    let qh_pos = out.len();
    out.extend_from_slice(&[0u8; QK_K / 8]);
    // ql[128]
    let ql_pos = out.len();
    out.extend_from_slice(&[0u8; QK_K / 2]);

    let mut m1: u8 = 1;
    let mut m2: u8 = 2;
    // Each chunk of 64 elements writes 32 ql bytes; m1/m2 shift left by 2.
    for chunk in 0..4 {
        let n = chunk * 64;
        let ql_off = ql_pos + chunk * 32;
        for j in 0..32 {
            let mut l1 = l[n + j] as i32;
            if l1 > 15 {
                l1 -= 16;
                out[qh_pos + j] |= m1;
            }
            let mut l2 = l[n + j + 32] as i32;
            if l2 > 15 {
                l2 -= 16;
                out[qh_pos + j] |= m2;
            }
            out[ql_off + j] = (l1 as u8) | ((l2 as u8) << 4);
        }
        m1 <<= 2;
        m2 <<= 2;
    }
}

/// Pure-Rust port of `quantize_row_q5_K_ref` (ggml-quants.c:1582).
fn quantize_row_q5_k_ref(x: &[f32], out: &mut Vec<u8>) {
    debug_assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;

    let mut mins = [0.0f32; QK_K / 32];
    let mut scales_arr = [0.0f32; QK_K / 32];
    let mut weights = [0.0f32; 32];
    let mut l_buf = [0u8; 32];
    let mut laux = [0u8; 32];

    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut max_scale: f32 = 0.0;
        let mut max_min: f32 = 0.0;
        // Per-codex 0bd0e7eb review: L[QK_K] must accumulate across all
        // sub-blocks (matching C's `L[QK_K]` in q5_K_ref) so finalize_block
        // preserves correct values for d==0 sub-blocks.
        let mut l_full = [0u8; QK_K];
        for j in 0..QK_K / 32 {
            // weights[l] = av_x + |x[l]|; av_x = sqrt(sum_x2 / 32)
            let mut sum_x2 = 0.0f32;
            for l in 0..32 {
                sum_x2 += xb[32 * j + l] * xb[32 * j + l];
            }
            let av_x = (sum_x2 / 32.0).sqrt();
            for l in 0..32 {
                weights[l] = av_x + xb[32 * j + l].abs();
            }
            let (scale, the_min) = make_qkx2_quants(
                32,
                31,
                &xb[32 * j..32 * j + 32],
                &weights,
                &mut l_full[32 * j..32 * j + 32],
                &mut laux,
                -0.5,
                0.1,
                15,
                false,
            );
            scales_arr[j] = scale;
            mins[j] = the_min;
            let _ = &mut l_buf;

            if scale > max_scale {
                max_scale = scale;
            }
            if the_min > max_min {
                max_min = the_min;
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
            // C: ls/lm are uint8_t — nearest_int returns int and the C
            // implicit cast truncates to low 8 bits. For non-negative
            // values bounded by 63 (post-MIN) this fits in u8 directly.
            let ls = (nearest_int(inv_scale * scales_arr[j]) as u8).min(63);
            let lm = (nearest_int(inv_min * mins[j]) as u8).min(63);
            pack_scale_min_k4(j, &mut scales_packed, ls, lm);
        }
        let d_f16 = f16::from_f32(max_scale / 63.0);
        let dmin_f16 = f16::from_f32(max_min / 63.0);

        finalize_block(xb, d_f16, dmin_f16, &scales_packed, &l_full, out);
    }
}

/// Pure-Rust port of `quantize_row_q5_K_impl` (ggml-quants.c:1696).
fn quantize_row_q5_k_impl(x: &[f32], quant_weights: &[f32], out: &mut Vec<u8>) {
    debug_assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;

    let mut mins = [0.0f32; QK_K / 32];
    let mut scales_arr = [0.0f32; QK_K / 32];
    let mut sw = [0.0f32; QK_K / 32];
    let mut ls_arr = [0u8; QK_K / 32];
    let mut lm_arr = [0u8; QK_K / 32];
    let mut weights = [0.0f32; 32];
    // Per-codex 0bd0e7eb: accumulate full L[QK_K] across sub-blocks so
    // finalize_block preserves prior values for d==0 sub-blocks.
    let mut l_full = [0u8; QK_K];
    let mut laux = [0u8; 32];

    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let qw_block = &quant_weights[i * QK_K..(i + 1) * QK_K];

        let mut sum_x2 = 0.0f32;
        for l in 0..QK_K {
            sum_x2 += xb[l] * xb[l];
        }
        let sigma2 = 2.0 * sum_x2 / (QK_K as f32);
        // av_x = sqrt(sigma2); kept for parity even though unused when
        // quant_weights is non-null.
        let _av_x = sigma2.sqrt();

        for j in 0..QK_K / 32 {
            let qw = &qw_block[32 * j..32 * j + 32];
            for l in 0..32 {
                weights[l] = qw[l] * (sigma2 + xb[32 * j + l] * xb[32 * j + l]).sqrt();
            }
            let mut sumw = 0.0f32;
            for l in 0..32 {
                sumw += weights[l];
            }
            sw[j] = sumw;

            let (scale, the_min) = make_qkx3_quants(
                32,
                31,
                &xb[32 * j..32 * j + 32],
                &weights,
                &mut l_full[32 * j..32 * j + 32],
                &mut laux,
                -0.9,
                0.05,
                36,
                false,
            );
            scales_arr[j] = scale;
            mins[j] = the_min;
        }

        let d_block = make_qp_quants(QK_K / 32, 63, &scales_arr, &mut ls_arr, &sw);
        let m_block = make_qp_quants(QK_K / 32, 63, &mins, &mut lm_arr, &sw);

        let mut scales_packed = [0u8; K_SCALE_SIZE];
        for j in 0..QK_K / 32 {
            let ls = ls_arr[j].min(63);
            let lm = lm_arr[j].min(63);
            pack_scale_min_k4(j, &mut scales_packed, ls, lm);
        }
        let d_f16 = f16::from_f32(d_block);
        let dmin_f16 = f16::from_f32(m_block);

        finalize_block(xb, d_f16, dmin_f16, &scales_packed, &l_full, out);
    }
}

/// Quantize an F32 buffer to Q5_K bytes.
///
/// Mirrors `quantize_q5_K` (ggml-quants.c:1789):
/// - `imatrix == None`        → row-by-row `_ref` over the whole buffer.
/// - `imatrix == Some(slice)` → per-row `_impl` (single n_per_row imatrix
///   vector reused for every row, mirroring C's single-pointer convention).
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
    if let Some(im) = imatrix {
        assert_eq!(
            im.len(),
            n_per_row,
            "imatrix length {} must equal n_per_row {}",
            im.len(),
            n_per_row
        );
    }

    let total_blocks = src.len() / QK_K;
    let mut out = Vec::with_capacity(total_blocks * BLOCK_BYTES);

    match imatrix {
        None => {
            quantize_row_q5_k_ref(src, &mut out);
        }
        Some(qw) => {
            let nrow = src.len() / n_per_row;
            for row in 0..nrow {
                let row_x = &src[row * n_per_row..(row + 1) * n_per_row];
                quantize_row_q5_k_impl(row_x, qw, &mut out);
            }
        }
    }

    out
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
        let input = read_f32s("q5_k_512_noim_input.bin");
        let expected = read_bytes("q5_k_512_noim_expected.bin");
        let got = quantize(&input, 512, None);
        assert_eq!(got.len(), expected.len(), "Q5_K noim length mismatch");
        assert_eq!(got, expected, "Q5_K noim byte-cmp failed");
    }

    #[test]
    fn byte_cmp_im() {
        let input = read_f32s("q5_k_512_im_input.bin");
        let expected = read_bytes("q5_k_512_im_expected.bin");
        // Harness `IMATRIX_SEED=2` (scripts/ggml_quants_harness/generate_all.sh:21).
        let imatrix = make_imatrix(512, 2);
        let got = quantize(&input, 512, Some(&imatrix));
        assert_eq!(got.len(), expected.len(), "Q5_K im length mismatch");
        assert_eq!(got, expected, "Q5_K im byte-cmp failed");
    }
}
