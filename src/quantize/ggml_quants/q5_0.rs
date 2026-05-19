//! Q5_0 quantizer — ADR-033 P0 pure-Rust port of
//! `quantize_row_q5_0_ref` at `/opt/llama.cpp/ggml/src/ggml-quants.c:145`
//! and `quantize_row_q5_0_impl` at `/opt/llama.cpp/ggml/src/ggml-quants.c:2112`
//! (SHA pinned in `data/llama_cpp_pin.txt`).
//!
//! Block layout from `ggml-common.h:219-225`:
//! ```text
//! #define QK5_0 32
//! typedef struct {
//!     ggml_half d;            // 2 bytes, F16 LE
//!     uint8_t   qh[4];        // 4 bytes — high (5th) bits as u32 LE
//!     uint8_t   qs[QK5_0/2];  // 16 bytes — packed 4-bit low nibbles
//! } block_q5_0;
//! // sizeof == 22 bytes
//! ```
//!
//! QH layout (from the C ref, `ggml-quants.c:181-182`):
//! `qh` is materialized as a `uint32_t` then `memcpy`'d into `qh[4]` as
//! little-endian bytes. Within that u32, the high (5th) bit of quant
//! `j ∈ [0, 16)` lives at bit `j` (low half) and the high bit of quant
//! `j+16` lives at bit `j + QK5_0/2 = j + 16` (high half). I.e. the
//! halves are stored in distinct 16-bit lanes, **not** interleaved.
//!
//! Dispatcher (`quantize_q5_0`, `ggml-quants.c:2151`): `quant_weights` is
//! `NULL` → `quantize_row_q5_0_ref`; otherwise → `quantize_row_q5_0_impl`
//! (uses `make_qx_quants(QK5_0, 16, …, rmse_type=1, weight)` with the
//! per-block `weight[j] = qw[j] * sqrt(sigma2 + x[j]^2)` scheme shared
//! with Q4_0/Q4_1/Q5_1).

use half::f16;

use crate::quantize::k_quant::make_qx_quants;

pub const QK5_0: usize = 32;
pub const BLOCK_BYTES: usize = 2 + 4 + QK5_0 / 2; // 22

/// Quantize an F32 buffer to Q5_0 bytes.
///
/// `n_per_row` must be a multiple of `QK5_0`, and `src.len()` must be a
/// multiple of `n_per_row`. With `imatrix = None` this is byte-identical
/// to `quantize_row_q5_0_ref`; with `Some(qw)` it follows
/// `quantize_row_q5_0_impl` (per-row sigma2, then per-block weighted
/// `make_qx_quants`).
pub fn quantize(src: &[f32], n_per_row: usize, imatrix: Option<&[f32]>) -> Vec<u8> {
    assert!(
        n_per_row % QK5_0 == 0,
        "n_per_row {} not multiple of QK5_0 {}",
        n_per_row,
        QK5_0
    );
    assert!(
        src.len() % n_per_row == 0,
        "src len {} not multiple of n_per_row {}",
        src.len(),
        n_per_row
    );
    // The imatrix may be either per-row (len == n_per_row, reused across
    // every row — mirrors how `ggml_quantize_chunk` passes a single
    // `quant_weights` pointer through `quantize_q5_0` without advancing
    // it per row) or per-element (len == src.len(), one slice per row).
    if let Some(qw) = imatrix {
        assert!(
            qw.len() == n_per_row || qw.len() == src.len(),
            "imatrix len {} must equal n_per_row {} or src len {}",
            qw.len(),
            n_per_row,
            src.len()
        );
    }

    let nrow = src.len() / n_per_row;
    let blocks_per_row = n_per_row / QK5_0;
    let mut out = Vec::with_capacity(nrow * blocks_per_row * BLOCK_BYTES);

    for row in 0..nrow {
        let x_row = &src[row * n_per_row..(row + 1) * n_per_row];
        match imatrix {
            None => quantize_row_ref(x_row, &mut out),
            Some(qw_all) => {
                let qw_row = if qw_all.len() == n_per_row {
                    qw_all
                } else {
                    &qw_all[row * n_per_row..(row + 1) * n_per_row]
                };
                quantize_row_impl(x_row, qw_row, &mut out);
            }
        }
    }

    out
}

/// Mirror of `quantize_row_q5_0_ref` (ggml-quants.c:145-187). Appends
/// `n/QK5_0` blocks of 22 bytes each to `out`.
fn quantize_row_ref(x: &[f32], out: &mut Vec<u8>) {
    debug_assert!(x.len() % QK5_0 == 0);
    let nb = x.len() / QK5_0;

    for i in 0..nb {
        let block = &x[i * QK5_0..(i + 1) * QK5_0];

        // amax / max scan (signed max-by-abs).
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in block {
            let av = v.abs();
            if amax < av {
                amax = av;
                max = v;
            }
        }

        let d = max / -16.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let d_f16 = f16::from_f32(d);
        out.extend_from_slice(&d_f16.to_le_bytes());

        // qs (16 bytes) + qh (4 bytes) — assemble then write in C struct
        // order: d, qh[4], qs[16].
        let mut qh: u32 = 0;
        let mut qs = [0u8; QK5_0 / 2];

        for j in 0..(QK5_0 / 2) {
            let x0 = block[j] * id;
            let x1 = block[QK5_0 / 2 + j] * id;

            // C: `MIN(31, (int8_t)(x0 + 16.5f))`. C truncates toward
            // zero on float→int conversion. Since x0+16.5 is in the
            // approximate range [0.5, 31.5] for in-distribution values,
            // truncation == floor here; for safety we cast f32→i32 via
            // truncation (matching C's int8_t conversion behavior for
            // small magnitudes, which is what this path produces) and
            // clamp to ≤31.
            let xi0_i = (x0 + 16.5) as i32; // truncation toward zero
            let xi1_i = (x1 + 16.5) as i32;
            let xi0 = xi0_i.min(31) as u8;
            let xi1 = xi1_i.min(31) as u8;

            qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            qh |= (((xi0 & 0x10) >> 4) as u32) << j;
            qh |= (((xi1 & 0x10) >> 4) as u32) << (j + QK5_0 / 2);
        }

        out.extend_from_slice(&qh.to_le_bytes());
        out.extend_from_slice(&qs);
    }
}

/// Mirror of `quantize_row_q5_0_impl` (ggml-quants.c:2112-2149).
fn quantize_row_impl(x: &[f32], qw: &[f32], out: &mut Vec<u8>) {
    debug_assert_eq!(x.len(), qw.len());
    debug_assert!(x.len() % QK5_0 == 0);

    let n_per_row = x.len();

    // sigma2 = mean(x^2) over the whole row.
    let mut sum_x2 = 0.0f32;
    for &v in x {
        sum_x2 += v * v;
    }
    let sigma2 = sum_x2 / n_per_row as f32;

    let nb = n_per_row / QK5_0;
    let mut weight = [0.0f32; QK5_0];
    let mut l = [0i8; QK5_0];

    for ib in 0..nb {
        let xb = &x[QK5_0 * ib..QK5_0 * (ib + 1)];
        let qwb = &qw[QK5_0 * ib..QK5_0 * (ib + 1)];

        for j in 0..QK5_0 {
            weight[j] = qwb[j] * (sigma2 + xb[j] * xb[j]).sqrt();
        }

        let d = make_qx_quants(QK5_0, 16, xb, &mut l, 1, Some(&weight));

        let d_f16 = f16::from_f32(d);
        out.extend_from_slice(&d_f16.to_le_bytes());

        let mut qh: u32 = 0;
        let mut qs = [0u8; QK5_0 / 2];

        for j in 0..16 {
            let xi0 = l[j] as u8; // L[j] ∈ [0, 31] after make_qx_quants offset
            let xi1 = l[j + 16] as u8;

            qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            qh |= (((xi0 & 0x10) >> 4) as u32) << j;
            qh |= (((xi1 & 0x10) >> 4) as u32) << (j + QK5_0 / 2);
        }

        out.extend_from_slice(&qh.to_le_bytes());
        out.extend_from_slice(&qs);
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

    /// Mulberry32 PRNG — must byte-match the generator used by
    /// `scripts/ggml_quants_harness/gen` to build the `_im` fixtures.
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
        let input = read_f32s("q5_0_64_noim_input.bin");
        let expected = read_bytes("q5_0_64_noim_expected.bin");
        let got = quantize(&input, 64, None);
        assert_eq!(got.len(), expected.len(), "Q5_0 noim length mismatch");
        assert_eq!(got, expected, "Q5_0 noim byte-cmp failed");
    }

    #[test]
    fn byte_cmp_im() {
        let input = read_f32s("q5_0_64_im_input.bin");
        let expected = read_bytes("q5_0_64_im_expected.bin");
        // Per `scripts/ggml_quants_harness/generate_all.sh`: imatrix_seed=2
        // and imatrix length == n_per_row (reused across every row by the
        // C harness, mirroring `quantize_q5_0`'s single-pointer ABI).
        let imatrix = make_imatrix(64, 2);
        let got = quantize(&input, 64, Some(&imatrix));
        assert_eq!(got.len(), expected.len(), "Q5_0 im length mismatch");
        assert_eq!(got, expected, "Q5_0 im byte-cmp failed");
    }
}
