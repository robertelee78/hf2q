//! Q4_0 quantizer — ADR-033 P0 pure-Rust port of
//! `quantize_row_q4_0_ref` at `/opt/llama.cpp/ggml/src/ggml-quants.c:71`
//! and `quantize_row_q4_0_impl` at `/opt/llama.cpp/ggml/src/ggml-quants.c:2008`
//! (SHA pinned in `data/llama_cpp_pin.txt`).
//!
//! Block layout from `ggml-common.h:184-189`:
//! ```text
//! #define QK4_0 32
//! typedef struct {
//!     ggml_half d;           // 2 bytes, F16, little-endian
//!     uint8_t qs[QK4_0 / 2]; // 16 bytes, 4-bit nibbles
//! } block_q4_0;
//! static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2);
//! ```
//!
//! The dispatcher at `ggml-quants.c:2052` (`quantize_q4_0`) calls
//! `quantize_row_q4_0_ref` when `quant_weights == NULL`, else
//! `quantize_row_q4_0_impl` which uses `make_qx_quants` with
//! `weight[j] = qw[j] * sqrt(sigma2 + xb[j]^2)`.

use half::f16;

pub const QK4_0: usize = 32;
pub const BLOCK_BYTES: usize = 2 + QK4_0 / 2; // 18

const GROUP_MAX_EPS: f32 = 1e-15;

/// Mirrors `nearest_int` at `ggml-quants.c:559`. The C version uses a
/// bit-cast trick that rounds via the float-to-int hardware mode (RNE).
/// We use `roundf` semantics via `(x + sign*0.5).floor()` — but the
/// safer 1:1 port is to mirror the bitcast.
#[inline(always)]
fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4194303.0);
    let val = fval + 12582912.0;
    let i = val.to_bits() as i32;
    (i & 0x007fffff) - 0x00400000
}

/// Mirrors `make_qx_quants` at `ggml-quants.c:566` for the specific
/// call site used by Q4_0: `rmse_type = 1`, `qw` is non-null. We keep
/// the general structure so it byte-matches reference for the Q4_0
/// path.
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

/// Pure reference path (no imatrix). Mirrors `quantize_row_q4_0_ref`
/// at `ggml-quants.c:71`.
fn quantize_row_ref(src: &[f32], out: &mut Vec<u8>) {
    let qk = QK4_0;
    debug_assert!(src.len() % qk == 0);
    let nb = src.len() / qk;
    for i in 0..nb {
        let block = &src[i * qk..(i + 1) * qk];

        let mut amax: f32 = 0.0;
        let mut max: f32 = 0.0;
        for j in 0..qk {
            let v = block[j];
            if amax < v.abs() {
                amax = v.abs();
                max = v;
            }
        }

        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());

        for j in 0..qk / 2 {
            let x0 = block[j] * id;
            let x1 = block[qk / 2 + j] * id;

            // C: const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            // (int8_t) is a truncating cast (toward zero); for positive
            // (x0+8.5) the result is floor. Negative values cannot
            // occur here because max/-8 produces id with sign that
            // shifts x*id into the [0, 15] range, but follow C exactly.
            let xi0 = ((x0 + 8.5) as i8).min(15) as u8;
            let xi1 = ((x1 + 8.5) as i8).min(15) as u8;

            out.push(xi0 | (xi1 << 4));
        }
    }
}

/// imatrix-aware path. Mirrors `quantize_row_q4_0_impl` at
/// `ggml-quants.c:2008`. Operates one row at a time.
fn quantize_row_impl(x: &[f32], imatrix: &[f32], out: &mut Vec<u8>) {
    let n_per_row = x.len();
    debug_assert_eq!(imatrix.len(), n_per_row);
    debug_assert!(n_per_row % QK4_0 == 0);

    let mut weight = [0.0f32; QK4_0];
    let mut l_buf = [0i8; QK4_0];

    let mut sum_x2: f32 = 0.0;
    for j in 0..n_per_row {
        sum_x2 += x[j] * x[j];
    }
    let sigma2 = sum_x2 / n_per_row as f32;

    let nb = n_per_row / QK4_0;
    for ib in 0..nb {
        let xb = &x[QK4_0 * ib..QK4_0 * (ib + 1)];
        let qw = &imatrix[QK4_0 * ib..QK4_0 * (ib + 1)];
        for j in 0..QK4_0 {
            weight[j] = qw[j] * (sigma2 + xb[j] * xb[j]).sqrt();
        }
        let d = make_qx_quants(QK4_0, 8, xb, &mut l_buf, 1, &weight);
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for j in 0..16 {
            // C: y[ib].qs[j] = L[j] | (L[j+16] << 4);
            // L[] holds values in [0, 15] (nmax=8 + clamped in [-8, 7]).
            let lo = l_buf[j] as u8;
            let hi = l_buf[j + 16] as u8;
            out.push(lo | (hi << 4));
        }
    }
}

/// Quantize an F32 buffer to Q4_0 bytes.
///
/// Mirrors dispatcher at `ggml-quants.c:2052`. When `imatrix` is
/// `Some`, the imatrix-aware `_impl` path is used per-row; otherwise
/// the `_ref` path is used.
pub fn quantize(src: &[f32], n_per_row: usize, imatrix: Option<&[f32]>) -> Vec<u8> {
    assert!(
        n_per_row % QK4_0 == 0,
        "n_per_row {} not multiple of QK4_0 {}",
        n_per_row,
        QK4_0
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
            "imatrix len {} must equal n_per_row {}",
            im.len(),
            n_per_row
        );
    }

    let total_blocks = src.len() / QK4_0;
    let mut out = Vec::with_capacity(total_blocks * BLOCK_BYTES);

    let nrow = src.len() / n_per_row;
    match imatrix {
        None => {
            quantize_row_ref(src, &mut out);
        }
        Some(im) => {
            for row in 0..nrow {
                let row_src = &src[row * n_per_row..(row + 1) * n_per_row];
                quantize_row_impl(row_src, im, &mut out);
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
        let input = read_f32s("q4_0_64_noim_input.bin");
        let expected = read_bytes("q4_0_64_noim_expected.bin");
        let got = quantize(&input, 64, None);
        assert_eq!(got.len(), expected.len(), "Q4_0 noim length mismatch");
        assert_eq!(got, expected, "Q4_0 noim byte-cmp failed");
    }

    #[test]
    fn byte_cmp_im() {
        let input = read_f32s("q4_0_64_im_input.bin");
        let expected = read_bytes("q4_0_64_im_expected.bin");
        let imatrix = make_imatrix(64, 2);
        let got = quantize(&input, 64, Some(&imatrix));
        assert_eq!(got.len(), expected.len(), "Q4_0 im length mismatch");
        assert_eq!(got, expected, "Q4_0 im byte-cmp failed");
    }
}
