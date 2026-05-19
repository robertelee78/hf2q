//! Q4_1 quantizer — ADR-033 P0 pure-Rust port of
//! `quantize_row_q4_1_ref` (`/opt/llama.cpp/ggml/src/ggml-quants.c:108`) and
//! `quantize_row_q4_1_impl` (`.../ggml-quants.c:2067`), with the dispatcher
//! at `quantize_q4_1` (`.../ggml-quants.c:2097`) selecting between them on
//! `quant_weights != NULL`.
//!
//! Block layout from `ggml-common.h:191-202`:
//! ```text
//! #define QK4_1 32
//! typedef struct {
//!     union { struct { ggml_half d; ggml_half m; }; ggml_half2 dm; };
//!     uint8_t qs[QK4_1/2];   // 16 nibble bytes
//! } block_q4_1;              // sizeof == 2*2 + 16 == 20
//! ```
//!
//! The `_impl` path delegates the scale/min search to `make_qkx3_quants`
//! (`.../ggml-quants.c:931`) with parameters `(QK4_1, 15, ..., -0.9, 0.05,
//! 36, use_mad=false)` and stores `m = GGML_FP32_TO_FP16(-min)` because
//! `make_qkx3_quants` returns `the_min = -min` already negated (see
//! `.../ggml-quants.c:1010`); the C call site then negates a second time
//! at `quantize_row_q4_1_impl:2090` (`y[ib].m = GGML_FP32_TO_FP16(-min)`).

use half::f16;

pub const QK4_1: usize = 32;
pub const BLOCK_BYTES: usize = 2 + 2 + QK4_1 / 2; // 20

/// Mirror of llama.cpp's `nearest_int` (`ggml-quants.c:559`) — bit-trick
/// round-to-nearest-even via float bias `12582912 = 1.5 * 2^23`.
#[inline]
fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4_194_303.0);
    let val = fval + 12_582_912.0;
    let i = val.to_bits();
    (i & 0x007f_ffff) as i32 - 0x0040_0000
}

/// Quantize an F32 buffer to Q4_1 bytes.
///
/// `src.len()` must be a multiple of `n_per_row`, and `n_per_row` must be
/// a multiple of `QK4_1`. When `imatrix` is `Some`, the `_impl` path runs
/// per row with `quant_weights` aliased to the whole imatrix slice (the
/// dispatcher feeds the same `quant_weights` pointer to every row — see
/// `quantize_q4_1:2104-2107`).
pub fn quantize(src: &[f32], n_per_row: usize, imatrix: Option<&[f32]>) -> Vec<u8> {
    assert!(
        n_per_row % QK4_1 == 0,
        "n_per_row {} not multiple of QK4_1 {}",
        n_per_row,
        QK4_1
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
    let row_blocks = n_per_row / QK4_1;
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

/// `quantize_row_q4_1_ref` (`ggml-quants.c:108-143`).
fn quantize_row_ref(x: &[f32], out: &mut Vec<u8>) {
    let qk = QK4_1;
    debug_assert_eq!(x.len() % qk, 0);
    let nb = x.len() / qk;

    for i in 0..nb {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for j in 0..qk {
            let v = x[i * qk + j];
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }

        let d = (max - min) / ((1 << 4) - 1) as f32;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        out.extend_from_slice(&f16::from_f32(min).to_le_bytes());

        // 16 nibble-packed bytes: low = first half [0..16), high = second half [16..32).
        for j in 0..qk / 2 {
            let x0 = (x[i * qk + j] - min) * id;
            let x1 = (x[i * qk + qk / 2 + j] - min) * id;

            // C: MIN(15, (int8_t)(x0 + 0.5f)) — note (int8_t) cast truncates
            // toward zero on positive floats, then clamps high side only.
            let xi0 = ((x0 + 0.5) as i8).min(15) as u8;
            let xi1 = ((x1 + 0.5) as i8).min(15) as u8;

            out.push(xi0 | (xi1 << 4));
        }
    }
}

/// `quantize_row_q4_1_impl` (`ggml-quants.c:2067-2095`).
fn quantize_row_impl(x: &[f32], quant_weights: &[f32], out: &mut Vec<u8>) {
    debug_assert_eq!(x.len() % QK4_1, 0);
    debug_assert_eq!(quant_weights.len(), x.len());

    let n_per_row = x.len();
    let mut sum_x2 = 0.0f32;
    for &v in x {
        sum_x2 += v * v;
    }
    let sigma2 = sum_x2 / n_per_row as f32;

    let nb = n_per_row / QK4_1;
    let mut weight = [0.0f32; QK4_1];
    let mut l_arr = [0u8; QK4_1];
    let mut l_aux = [0u8; QK4_1];

    for ib in 0..nb {
        let xb = &x[QK4_1 * ib..QK4_1 * (ib + 1)];
        let qw = &quant_weights[QK4_1 * ib..QK4_1 * (ib + 1)];
        for j in 0..QK4_1 {
            weight[j] = qw[j] * (sigma2 + xb[j] * xb[j]).sqrt();
        }
        let mut min_neg = 0.0f32; // populated by make_qkx3_quants as -min
        let d = make_qkx3_quants(
            QK4_1,
            15,
            xb,
            Some(&weight),
            &mut l_arr,
            &mut min_neg,
            &mut l_aux,
            -0.9,
            0.05,
            36,
            false,
        );
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        // `make_qkx3_quants` writes `the_min = -min`; the C call site then
        // stores `GGML_FP32_TO_FP16(-min)`, i.e. negating the OUT param
        // `min_neg` back. Net: `m = f16(-(-min)) = f16(min_neg negated)`.
        // Source: `ggml-quants.c:2090`.
        out.extend_from_slice(&f16::from_f32(-min_neg).to_le_bytes());
        for j in 0..16 {
            out.push(l_arr[j] | (l_arr[j + 16] << 4));
        }
    }
}

/// `make_qkx3_quants` (`ggml-quants.c:931-1012`).
///
/// Q4_1's `_impl` always passes a non-null weights slice, so the weight-
/// optional branches of the C source are kept here for fidelity but the
/// caller-side path is always `Some(weights)`.
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
        let input = read_f32s("q4_1_64_noim_input.bin");
        let expected = read_bytes("q4_1_64_noim_expected.bin");
        let got = quantize(&input, 64, None);
        assert_eq!(got.len(), expected.len(), "Q4_1 noim length mismatch");
        assert_eq!(got, expected, "Q4_1 noim byte-cmp failed");
    }

    #[test]
    fn byte_cmp_im() {
        let input = read_f32s("q4_1_64_im_input.bin");
        let expected = read_bytes("q4_1_64_im_expected.bin");
        let imatrix = make_imatrix(64, 2);
        let got = quantize(&input, 64, Some(&imatrix));
        assert_eq!(got.len(), expected.len(), "Q4_1 im length mismatch");
        assert_eq!(got, expected, "Q4_1 im byte-cmp failed");
    }
}
