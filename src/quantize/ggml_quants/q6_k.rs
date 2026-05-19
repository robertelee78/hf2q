//! Q6_K quantizer — ADR-033 P0 pure-Rust port of
//! `quantize_row_q6_K_ref` (`/opt/llama.cpp/ggml/src/ggml-quants.c:1807`),
//! `quantize_row_q6_K_impl` (`.../ggml-quants.c:1908`), and the dispatcher
//! `quantize_q6_K` (`.../ggml-quants.c:1992`) at the SHA pinned in
//! `data/llama_cpp_pin.txt`.
//!
//! Block layout from `ggml-common.h:350-358`:
//! ```text
//! #define QK_K 256
//! typedef struct {
//!     uint8_t ql[QK_K/2];      // 128 bytes: low 4 bits of quants
//!     uint8_t qh[QK_K/4];      // 64  bytes: high 2 bits of quants
//!     int8_t  scales[QK_K/16]; // 16  bytes: per-sub-block scales (signed)
//!     ggml_half d;             // 2   bytes: super-block scale (F16 LE)
//! } block_q6_K;                // sizeof == 210
//! ```
//!
//! Q6_K has 16 sub-blocks of 16 elements each. Per-sub-block scales are
//! stored FLAT as `int8_t` (no 6-bit packing tricks like Q4_K/Q5_K).
//! Both `_ref` and `_impl` delegate the per-sub-block scale to
//! `make_qx_quants(16, 32, x, L, rmse_type=1, qw)`. The only difference
//! is that `_impl` passes `quant_weights + QK_K*i + 16*ib` for `qw`
//! (the slice for that sub-block) while `_ref` passes NULL — see the
//! commented-out `weights` block in `quantize_row_q6_K_impl` at
//! `ggml-quants.c:1918-1920,1930-1931` which shows the historical
//! sigma2-weighted path was removed.

use half::f16;

pub const QK_K: usize = 256;
pub const BLOCK_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2; // 128 + 64 + 16 + 2 = 210

const GROUP_MAX_EPS: f32 = 1e-15;

/// Mirror of llama.cpp's `nearest_int` (`ggml-quants.c:559`) — bit-trick
/// round-to-nearest-even via float bias `12582912 = 1.5 * 2^23`.
#[inline(always)]
fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4_194_303.0);
    let val = fval + 12_582_912.0;
    let i = val.to_bits() as i32;
    (i & 0x007f_ffff) - 0x0040_0000
}

/// Mirror of `make_qx_quants` (`ggml-quants.c:566`). Q6_K uses
/// `(n=16, nmax=32, rmse_type=1, qw=quant_weights_or_null)`.
fn make_qx_quants(n: usize, nmax: i32, x: &[f32], l: &mut [i8], rmse_type: i32, qw: &[f32]) -> f32 {
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

/// Writes one Q6_K super-block (210 bytes) for f32 slice `xb[0..QK_K]`.
/// `qw` is either an empty slice (ref path) or a `QK_K`-len slice of
/// imatrix weights (impl path). Mirrors the post-scale-selection tail
/// shared between `_ref` (`ggml-quants.c:1839-1873`) and `_impl`
/// (`ggml-quants.c:1953-1986`).
fn quantize_one_block(xb: &[f32], qw: &[f32], out: &mut Vec<u8>) {
    debug_assert_eq!(xb.len(), QK_K);
    debug_assert!(qw.is_empty() || qw.len() == QK_K);

    let mut l_buf = [0i8; QK_K];
    let mut scales = [0f32; QK_K / 16];

    let mut max_scale: f32 = 0.0;
    let mut max_abs_scale: f32 = 0.0;

    for ib in 0..(QK_K / 16) {
        let x_sub = &xb[16 * ib..16 * (ib + 1)];
        let l_sub = &mut l_buf[16 * ib..16 * (ib + 1)];
        let qw_sub: &[f32] = if qw.is_empty() {
            &[]
        } else {
            &qw[16 * ib..16 * (ib + 1)]
        };
        let scale = make_qx_quants(16, 32, x_sub, l_sub, 1, qw_sub);
        scales[ib] = scale;
        let abs_scale = scale.abs();
        if abs_scale > max_abs_scale {
            max_abs_scale = abs_scale;
            max_scale = scale;
        }
    }

    if max_abs_scale < GROUP_MAX_EPS {
        // memset(&y[i], 0, sizeof(block_q6_K)); y[i].d = FP32_TO_FP16(0.f).
        // Output ql[128] zero, qh[64] zero, scales[16] zero, d = f16(0).
        let start = out.len();
        out.resize(start + BLOCK_BYTES, 0);
        // d already 0 from memset; explicit zero f16 little-endian == [0,0] — already done.
        return;
    }

    let iscale = -128.0_f32 / max_scale;
    let d_f16 = f16::from_f32(1.0 / iscale);

    // Compute final int8 scales (clamped MIN(127, nearest_int(iscale*scales[ib]))).
    // Note: C uses `int8_t y[i].scales[ib]`; assignment truncates the int into
    // int8 storage. `MIN(127, n)` only clamps the high side, so values below
    // -128 are theoretically possible — but `iscale = -128/max_scale` and
    // `|scales[ib]| <= |max_scale|`, so `iscale*scales[ib]` is in [-128, 128];
    // nearest_int can yield -128. We mirror C exactly: only clamp on the high
    // end, then truncate-cast to i8 (which is the assignment-to-int8 behavior).
    let mut sc_i8 = [0i8; QK_K / 16];
    for ib in 0..(QK_K / 16) {
        let ni = nearest_int(iscale * scales[ib]);
        let clamped_hi = ni.min(127);
        sc_i8[ib] = clamped_hi as i8;
    }

    // Re-quantize using the (truncated) i8 scales — note that `y[i].scales[j]`
    // in C is read back as `int8_t` (sign-extended), so we re-promote `sc_i8`
    // to f32 via the signed int route.
    let d_back = d_f16.to_f32();
    for j in 0..(QK_K / 16) {
        let d = d_back * (sc_i8[j] as f32);
        if d == 0.0 {
            // C's `if (!d) continue;` — leaves L[16*j .. 16*j+16] at whatever
            // value `make_qx_quants` wrote (in [0, 63] from L = li+32). That's
            // fine: the packing step below ANDs with 0xF / shifts >> 4.
            continue;
        }
        for ii in 0..16 {
            let li = nearest_int(xb[16 * j + ii] / d);
            let li_c = li.max(-32).min(31);
            l_buf[16 * j + ii] = (li_c + 32) as i8;
        }
    }

    // Emit ql (128) + qh (64) + scales (16) + d (2) in struct order.
    let start = out.len();
    out.resize(start + BLOCK_BYTES, 0);
    let block = &mut out[start..start + BLOCK_BYTES];
    let (ql_buf, rest) = block.split_at_mut(QK_K / 2); // 128
    let (qh_buf, rest) = rest.split_at_mut(QK_K / 4); // 64
    let (sc_buf, d_buf) = rest.split_at_mut(QK_K / 16); // 16, 2

    // Pack ql / qh: the C loop iterates j in {0, 128} with l in [0,32).
    // For each 128-element chunk we consume L[j..j+128], producing 64 bytes
    // of ql and 32 bytes of qh, then advance ql by 64 and qh by 32.
    let mut ql_off = 0;
    let mut qh_off = 0;
    let mut j = 0;
    while j < QK_K {
        for l in 0..32usize {
            let q1 = (l_buf[j + l] as u8) & 0x0F;
            let q2 = (l_buf[j + l + 32] as u8) & 0x0F;
            let q3 = (l_buf[j + l + 64] as u8) & 0x0F;
            let q4 = (l_buf[j + l + 96] as u8) & 0x0F;
            ql_buf[ql_off + l] = q1 | (q3 << 4);
            ql_buf[ql_off + l + 32] = q2 | (q4 << 4);
            let h1 = (l_buf[j + l] as u8) >> 4;
            let h2 = (l_buf[j + l + 32] as u8) >> 4;
            let h3 = (l_buf[j + l + 64] as u8) >> 4;
            let h4 = (l_buf[j + l + 96] as u8) >> 4;
            qh_buf[qh_off + l] = h1 | (h2 << 2) | (h3 << 4) | (h4 << 6);
        }
        ql_off += 64;
        qh_off += 32;
        j += 128;
    }

    for ib in 0..(QK_K / 16) {
        sc_buf[ib] = sc_i8[ib] as u8;
    }
    let d_bytes = d_f16.to_le_bytes();
    d_buf[0] = d_bytes[0];
    d_buf[1] = d_bytes[1];
}

/// `quantize_row_q6_K_ref` (`ggml-quants.c:1807`).
fn quantize_row_ref(x: &[f32], out: &mut Vec<u8>) {
    debug_assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;
    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        quantize_one_block(xb, &[], out);
    }
}

/// `quantize_row_q6_K_impl` (`ggml-quants.c:1908`). Operates on one row;
/// `quant_weights` is the per-row imatrix slice (length `n_per_row`).
fn quantize_row_impl(x: &[f32], quant_weights: &[f32], out: &mut Vec<u8>) {
    debug_assert!(x.len() % QK_K == 0);
    debug_assert_eq!(quant_weights.len(), x.len());
    let nb = x.len() / QK_K;
    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let qw = &quant_weights[i * QK_K..(i + 1) * QK_K];
        quantize_one_block(xb, qw, out);
    }
}

/// Quantize an F32 buffer to Q6_K bytes.
///
/// Mirrors the dispatcher at `ggml-quants.c:1992` (`quantize_q6_K`).
/// `imatrix` is reused per-row (the C dispatcher feeds the same
/// `quant_weights` pointer to every row at line 2000).
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

    match imatrix {
        None => {
            quantize_row_ref(src, &mut out);
        }
        Some(qw) => {
            for row in 0..n_rows {
                let row_src = &src[row * n_per_row..(row + 1) * n_per_row];
                quantize_row_impl(row_src, qw, &mut out);
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

    /// Mulberry32 + abs() + 1e-3 — mirrors the harness's
    /// `make_imatrix(n_per_row, imatrix_seed=2)`.
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
        let input = read_f32s("q6_k_512_noim_input.bin");
        let expected = read_bytes("q6_k_512_noim_expected.bin");
        let got = quantize(&input, 512, None);
        assert_eq!(got.len(), expected.len(), "Q6_K noim length mismatch");
        assert_eq!(got, expected, "Q6_K noim byte-cmp failed");
    }

    #[test]
    fn byte_cmp_im() {
        let input = read_f32s("q6_k_512_im_input.bin");
        let expected = read_bytes("q6_k_512_im_expected.bin");
        let imatrix = make_imatrix(512, 2);
        let got = quantize(&input, 512, Some(&imatrix));
        assert_eq!(got.len(), expected.len(), "Q6_K im length mismatch");
        assert_eq!(got, expected, "Q6_K im byte-cmp failed");
    }
}
