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

use super::common::{make_qkx2_quants, make_qkx3_quants, make_qp_quants, nearest_int};

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;
pub const BLOCK_BYTES: usize = 2 + 2 + K_SCALE_SIZE + QK_K / 2; // 144


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
