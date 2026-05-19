//! Q2_K quantizer — ADR-033 P0 pure-Rust port of
//! `quantize_row_q2_K_ref` (`/opt/llama.cpp/ggml/src/ggml-quants.c:829`) and
//! `quantize_row_q2_K_impl` (`.../ggml-quants.c:1087`), with the dispatcher
//! at `quantize_q2_K` (`.../ggml-quants.c:1149`) selecting between them on
//! `quant_weights != NULL`.
//!
//! Block layout from `ggml-common.h:288-298`:
//! ```text
//! #define QK_K 256
//! typedef struct {
//!     uint8_t scales[QK_K/16];  // 16 bytes — low nibble = scale, high nibble = min
//!     uint8_t qs[QK_K/4];       // 64 bytes — 2-bit packed quants
//!     ggml_half d;              // super-block scale for quantized scales
//!     ggml_half dmin;           // super-block scale for quantized mins
//! } block_q2_K;
//! static_assert(sizeof(block_q2_K) == 2*sizeof(ggml_half) + QK_K/16 + QK_K/4); // 84
//! ```
//!
//! The `_ref` path uses `make_qkx2_quants` (`.../ggml-quants.c:737`) per
//! sub-block with `(16, 3, ..., -0.5, 0.1, 15, use_mad=true)` and per-sub-
//! block weight `fabsf(x)`. The `_impl` path uses `make_qkx3_quants`
//! (`.../ggml-quants.c:931`) with `(16, 3, ..., -0.9, 0.05, 36, use_mad=false)`
//! and weight `qw[l] * sqrtf(sigma2 + x*x)`, then `make_qp_quants`
//! (`.../ggml-quants.c:1014`) twice to quantize the per-sub-block scales
//! and mins jointly against `sw[]` weights.

use half::f16;

use super::common::{make_qkx2_quants, make_qkx3_quants, make_qp_quants, nearest_int};

pub const QK_K: usize = 256;
pub const BLOCK_BYTES: usize = QK_K / 16 + QK_K / 4 + 2 + 2; // 84

/// Quantize an F32 buffer to Q2_K bytes.
///
/// `src.len()` must be a multiple of `n_per_row`, and `n_per_row` must be
/// a multiple of `QK_K`. When `imatrix` is `Some`, the `_impl` path runs
/// per row with `quant_weights` aliased to the whole imatrix slice (the
/// dispatcher at `quantize_q2_K:1156-1160` feeds the same `quant_weights`
/// pointer to every row — it advances `src` and `qrow` but not
/// `quant_weights`).
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

/// `quantize_row_q2_K_ref` (`ggml-quants.c:829-897`).
fn quantize_row_ref(x: &[f32], out: &mut Vec<u8>) {
    debug_assert_eq!(x.len() % QK_K, 0);
    let nb = x.len() / QK_K;

    let q4scale = 15.0f32;

    let mut l_buf = [0u8; QK_K];
    let mut l_aux = [0u8; 16];
    let mut weights = [0.0f32; 16];
    let mut mins = [0.0f32; QK_K / 16];
    let mut scales = [0.0f32; QK_K / 16];

    for i in 0..nb {
        let xb = &x[QK_K * i..QK_K * (i + 1)];

        let mut max_scale = 0.0f32;
        let mut max_min = 0.0f32;
        for j in 0..QK_K / 16 {
            for l in 0..16 {
                weights[l] = xb[16 * j + l].abs();
            }
            let mut mn = 0.0f32;
            let sc = make_qkx2_quants(
                16,
                3,
                &xb[16 * j..16 * j + 16],
                &weights,
                &mut l_buf[16 * j..16 * j + 16],
                &mut mn,
                &mut l_aux,
                -0.5,
                0.1,
                15,
                true,
            );
            scales[j] = sc;
            mins[j] = mn;
            if sc > max_scale {
                max_scale = sc;
            }
            if mn > max_min {
                max_min = mn;
            }
        }

        // 16 byte scratch for `scales` packed nibbles (low = scale, high = min).
        let mut scales_packed = [0u8; QK_K / 16];

        let d_f16: f16;
        if max_scale > 0.0 {
            let iscale = q4scale / max_scale;
            for j in 0..QK_K / 16 {
                let l = nearest_int(iscale * scales[j]);
                // C stores into uint8_t — wraps via truncation; values
                // should land in [0,15] but we mirror the raw cast.
                scales_packed[j] = l as u8;
            }
            d_f16 = f16::from_f32(max_scale / q4scale);
        } else {
            for j in 0..QK_K / 16 {
                scales_packed[j] = 0;
            }
            d_f16 = f16::from_f32(0.0);
        }

        let dmin_f16: f16;
        if max_min > 0.0 {
            let iscale = q4scale / max_min;
            for j in 0..QK_K / 16 {
                let l = nearest_int(iscale * mins[j]);
                // C: y[i].scales[j] |= (l << 4); — l is int (up to 15),
                // shifted into high nibble.
                scales_packed[j] |= (l as u8) << 4;
            }
            dmin_f16 = f16::from_f32(max_min / q4scale);
        } else {
            dmin_f16 = f16::from_f32(0.0);
        }

        // Requantize step: use the f16-roundtripped d/dmin to re-pick L.
        let d_round = d_f16.to_f32();
        let dmin_round = dmin_f16.to_f32();
        for j in 0..QK_K / 16 {
            let d = d_round * (scales_packed[j] & 0xF) as f32;
            if d == 0.0 {
                continue;
            }
            let dm = dmin_round * (scales_packed[j] >> 4) as f32;
            for ii in 0..16 {
                let l = nearest_int((xb[16 * j + ii] + dm) / d);
                let l = l.max(0).min(3) as u8;
                l_buf[16 * j + ii] = l;
            }
        }

        // Emit block: scales[16] || qs[64] || d || dmin
        out.extend_from_slice(&scales_packed);
        // 2-bit pack: for each 128-element half, 32 bytes pack rows
        // (l[j+l], l[j+l+32], l[j+l+64], l[j+l+96]) into one byte at
        // shifts 0,2,4,6.
        let mut qs = [0u8; QK_K / 4];
        let mut j = 0usize;
        while j < QK_K {
            for l in 0..32 {
                qs[j / 4 + l] = l_buf[j + l]
                    | (l_buf[j + l + 32] << 2)
                    | (l_buf[j + l + 64] << 4)
                    | (l_buf[j + l + 96] << 6);
            }
            j += 128;
        }
        out.extend_from_slice(&qs);
        out.extend_from_slice(&d_f16.to_le_bytes());
        out.extend_from_slice(&dmin_f16.to_le_bytes());
    }
}

/// `quantize_row_q2_K_impl` (`ggml-quants.c:1087-1147`).
fn quantize_row_impl(x: &[f32], quant_weights: &[f32], out: &mut Vec<u8>) {
    debug_assert_eq!(x.len() % QK_K, 0);
    debug_assert_eq!(quant_weights.len(), x.len());

    let nb = x.len() / QK_K;

    let mut l_buf = [0u8; QK_K];
    let mut l_aux = [0u8; 16];
    let mut mins = [0.0f32; QK_K / 16];
    let mut scales = [0.0f32; QK_K / 16];
    let mut sw = [0.0f32; QK_K / 16];
    let mut weight = [0.0f32; 16];
    let mut ls = [0u8; QK_K / 16];
    let mut lm = [0u8; QK_K / 16];

    for i in 0..nb {
        let xb = &x[QK_K * i..QK_K * (i + 1)];
        let qw_row = &quant_weights[QK_K * i..QK_K * (i + 1)];

        for v in sw.iter_mut() {
            *v = 0.0;
        }
        let mut sumx2 = 0.0f32;
        for j in 0..QK_K {
            sumx2 += xb[j] * xb[j];
        }
        let sigma2 = sumx2 / QK_K as f32;

        for j in 0..QK_K / 16 {
            let qw = &qw_row[16 * j..16 * j + 16];
            for l in 0..16 {
                weight[l] = qw[l] * (sigma2 + xb[16 * j + l] * xb[16 * j + l]).sqrt();
            }
            // C bug-or-feature at ggml-quants.c:1109 — iterates l over
            // [0, QK_K/16=16) which equals [0,16), so this sums the
            // first 16 entries of `weight[]` (the full sub-block).
            for l in 0..QK_K / 16 {
                sw[j] += weight[l];
            }
            let mut min_neg = 0.0f32;
            let sc = make_qkx3_quants(
                16,
                3,
                &xb[16 * j..16 * j + 16],
                Some(&weight),
                &mut l_buf[16 * j..16 * j + 16],
                &mut min_neg,
                &mut l_aux,
                -0.9,
                0.05,
                36,
                false,
            );
            scales[j] = sc;
            mins[j] = min_neg; // C stores `mins[j]` = `the_min` = `-min`.
        }

        let mut dm = make_qp_quants(QK_K / 16, 15, &scales, &mut ls, &sw);
        let mut mm = make_qp_quants(QK_K / 16, 15, &mins, &mut lm, &sw);

        let d_f16 = f16::from_f32(dm);
        let dmin_f16 = f16::from_f32(mm);
        dm = d_f16.to_f32();
        mm = dmin_f16.to_f32();

        let mut scales_packed = [0u8; QK_K / 16];
        for j in 0..QK_K / 16 {
            scales_packed[j] = ls[j] | (lm[j] << 4);
        }

        // Requantize (`requantize = true` in C).
        for j in 0..QK_K / 16 {
            let d = dm * (scales_packed[j] & 0xF) as f32;
            if d == 0.0 {
                continue;
            }
            let m = mm * (scales_packed[j] >> 4) as f32;
            for ii in 0..16 {
                let l = nearest_int((xb[16 * j + ii] + m) / d);
                let l = l.max(0).min(3) as u8;
                l_buf[16 * j + ii] = l;
            }
        }

        out.extend_from_slice(&scales_packed);
        let mut qs = [0u8; QK_K / 4];
        let mut j = 0usize;
        while j < QK_K {
            for l in 0..32 {
                qs[j / 4 + l] = l_buf[j + l]
                    | (l_buf[j + l + 32] << 2)
                    | (l_buf[j + l + 64] << 4)
                    | (l_buf[j + l + 96] << 6);
            }
            j += 128;
        }
        out.extend_from_slice(&qs);
        out.extend_from_slice(&d_f16.to_le_bytes());
        out.extend_from_slice(&dmin_f16.to_le_bytes());
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
        let input = read_f32s("q2_k_512_noim_input.bin");
        let expected = read_bytes("q2_k_512_noim_expected.bin");
        let got = quantize(&input, 512, None);
        assert_eq!(got.len(), expected.len(), "Q2_K noim length mismatch");
        assert_eq!(got, expected, "Q2_K noim byte-cmp failed");
    }

    #[test]
    fn byte_cmp_im() {
        let input = read_f32s("q2_k_512_im_input.bin");
        let expected = read_bytes("q2_k_512_im_expected.bin");
        let imatrix = make_imatrix(512, 2);
        let got = quantize(&input, 512, Some(&imatrix));
        assert_eq!(got.len(), expected.len(), "Q2_K im length mismatch");
        assert_eq!(got, expected, "Q2_K im byte-cmp failed");
    }
}
