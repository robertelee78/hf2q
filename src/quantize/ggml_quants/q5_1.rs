//! Q5_1 quantizer — ADR-033 P0 pure-Rust port of
//! `quantize_row_q5_1_ref`   at `/opt/llama.cpp/ggml/src/ggml-quants.c:189`
//! `quantize_row_q5_1_impl`  at `/opt/llama.cpp/ggml/src/ggml-quants.c:2166`
//! `quantize_q5_1`           at `/opt/llama.cpp/ggml/src/ggml-quants.c:2204`
//! (SHA pinned in `data/llama_cpp_pin.txt`).
//!
//! Block layout from `ggml-common.h:227-239`:
//! ```text
//! #define QK5_1 32
//! typedef struct {
//!     ggml_half d;            // 2 bytes, F16 delta, little-endian
//!     ggml_half m;            // 2 bytes, F16 min,   little-endian
//!     uint8_t   qh[4];        // 5-th bits, packed as a u32
//!     uint8_t   qs[QK5_1/2];  // 16 bytes, low 4 bits as nibbles
//! } block_q5_1;               // 24 bytes total
//! ```
//!
//! Asymmetric, 5-bit. Two dispatch paths:
//! - `imatrix == None`  → `_ref`: simple min/max → d=(max-min)/31, m=min.
//! - `imatrix == Some`  → `_impl`: imatrix-aware `make_qkx3_quants` solver.
//!
//! Byte-parity gotchas:
//! - Magic-number rounding `nearest_int(fval)` must be replicated bit-for-bit
//!   (round-half-to-even via 12582912.f shift).
//! - `_ref` rounds via `(uint8_t)(x0 + 0.5f)` which is C truncation toward 0
//!   after add — equivalent to floor(x+0.5) for non-negative x (post-clamp
//!   range [0, 31] is non-negative by construction from `(x - min)*id`).
//! - `_impl` stores `m = GGML_FP32_TO_FP16(-min)` (note the sign — `the_min`
//!   from `make_qkx3_quants` is already `-min`, but `_ref` stores raw `min`
//!   directly; the difference is intentional, see lines 211 vs 2189).

use half::f16;

pub const QK5_1: usize = 32;
pub const BLOCK_BYTES: usize = 2 + 2 + 4 + QK5_1 / 2; // 24

/// Replicates llama.cpp's `nearest_int` (ggml-quants.c:559) — round-half-to-even
/// via the 12582912.f magic shift. Must be bit-exact for byte parity.
#[inline]
fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4194303.0);
    let val = fval + 12582912.0;
    let i = val.to_bits() as i32;
    (i & 0x007fffff) - 0x00400000
}

/// Pure-Rust port of `make_qkx3_quants` (ggml-quants.c:931).
/// Returns `(scale, the_min)` where `the_min == -min` (see line 1010).
/// `L` is filled with quantized indices in `[0, nmax]`.
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
    let mut iscale = nmax as f32 / (max - min);
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
            // Silence unused-assignment lints — Rust doesn't complain here in
            // practice but mirrors C scoping.
            let _ = iscale;
            iscale = iscale_try;
        }
    }
    (scale, -min)
}

/// Pure-Rust port of `quantize_row_q5_1_ref` (ggml-quants.c:189).
fn quantize_row_q5_1_ref(x: &[f32], out: &mut Vec<u8>) {
    let qk = QK5_1;
    debug_assert!(x.len() % qk == 0);
    let nb = x.len() / qk;

    for i in 0..nb {
        let block = &x[i * qk..(i + 1) * qk];
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in block {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }

        let d = (max - min) / 31.0; // (1 << 5) - 1
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let d_f16 = f16::from_f32(d);
        let m_f16 = f16::from_f32(min);
        out.extend_from_slice(&d_f16.to_le_bytes());
        out.extend_from_slice(&m_f16.to_le_bytes());

        let mut qh: u32 = 0;
        // Reserve qs[16] slots; write qh after.
        let qh_pos = out.len();
        out.extend_from_slice(&[0u8; 4]);
        let qs_pos = out.len();
        out.extend_from_slice(&[0u8; QK5_1 / 2]);

        for j in 0..qk / 2 {
            let x0 = (block[j] - min) * id;
            let x1 = (block[qk / 2 + j] - min) * id;

            // C: `(uint8_t)(x0 + 0.5f)` is truncation toward 0 after add.
            // For our domain (x in [0, 31] post-clamp), x + 0.5 is non-negative
            // and equivalent to floor(x + 0.5).
            let xi0 = (x0 + 0.5) as u8;
            let xi1 = (x1 + 0.5) as u8;

            out[qs_pos + j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            qh |= (((xi0 & 0x10) as u32) >> 4) << j;
            qh |= (((xi1 & 0x10) as u32) >> 4) << (j + qk / 2);
        }
        // Write qh as little-endian (memcpy of uint32_t into uint8_t qh[4]
        // on the LE targets we support).
        let qh_bytes = qh.to_le_bytes();
        out[qh_pos..qh_pos + 4].copy_from_slice(&qh_bytes);
    }
}

/// Pure-Rust port of `quantize_row_q5_1_impl` (ggml-quants.c:2166).
fn quantize_row_q5_1_impl(x: &[f32], quant_weights: &[f32], out: &mut Vec<u8>) {
    debug_assert_eq!(QK5_1, 32);

    let n_per_row = x.len();
    let mut sum_x2 = 0.0f32;
    for j in 0..n_per_row {
        sum_x2 += x[j] * x[j];
    }
    let sigma2 = sum_x2 / (n_per_row as f32);

    let nb = n_per_row / QK5_1;
    let mut weight = [0.0f32; QK5_1];
    let mut l_buf = [0u8; QK5_1];
    let mut laux = [0u8; QK5_1];

    for ib in 0..nb {
        let xb = &x[QK5_1 * ib..QK5_1 * (ib + 1)];
        let qw = &quant_weights[QK5_1 * ib..QK5_1 * (ib + 1)];
        for j in 0..QK5_1 {
            weight[j] = qw[j] * (sigma2 + xb[j] * xb[j]).sqrt();
        }
        let (d, the_min) = make_qkx3_quants(
            QK5_1, 31, xb, &weight, &mut l_buf, &mut laux, -0.9, 0.05, 36, false,
        );
        // C: y[ib].d = FP32_TO_FP16(d);
        // C: y[ib].m = FP32_TO_FP16(-min);  where `min` in the C local is the
        // post-solver `min` (negated again via `-min` to produce the stored
        // value). `make_qkx3_quants` returns `*the_min = -min`, so the C
        // expression `-min` (with min == the_min from the out-param, after
        // line `min = this_min` updates) is the C local — but the value the
        // function ultimately writes is `*the_min = -min` then C does
        // `FP32_TO_FP16(-min)` using its own local `min` (which is the raw
        // post-solver value, not -min). We return `(scale, -min)` so to match
        // C's `FP32_TO_FP16(-min)` we negate `the_min` again.
        let d_f16 = f16::from_f32(d);
        let m_f16 = f16::from_f32(-the_min);
        out.extend_from_slice(&d_f16.to_le_bytes());
        out.extend_from_slice(&m_f16.to_le_bytes());

        let mut qh: u32 = 0;
        let qh_pos = out.len();
        out.extend_from_slice(&[0u8; 4]);
        let qs_pos = out.len();
        out.extend_from_slice(&[0u8; QK5_1 / 2]);

        for j in 0..16 {
            let xi0 = l_buf[j];
            let xi1 = l_buf[j + 16];
            out[qs_pos + j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh |= (((xi0 & 0x10) as u32) >> 4) << j;
            // C uses QK5_0/2 (= 16) here; QK5_1/2 is also 16. Identical.
            qh |= (((xi1 & 0x10) as u32) >> 4) << (j + 16);
        }
        let qh_bytes = qh.to_le_bytes();
        out[qh_pos..qh_pos + 4].copy_from_slice(&qh_bytes);
    }
}

/// Quantize an F32 buffer to Q5_1 bytes.
///
/// Dispatcher mirrors `quantize_q5_1` (ggml-quants.c:2204):
/// - `imatrix == None`        → row-by-row `_ref` over the whole buffer.
/// - `imatrix == Some(slice)` → per-row `_impl` (imatrix is broadcast as a
///   single n_per_row vector; the caller is responsible for per-row layout
///   if rows differ — matches C semantics where `quant_weights` is a single
///   pointer reused for every row).
pub fn quantize(src: &[f32], n_per_row: usize, imatrix: Option<&[f32]>) -> Vec<u8> {
    assert!(
        n_per_row % QK5_1 == 0,
        "n_per_row {} not multiple of QK5_1 {}",
        n_per_row,
        QK5_1
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

    let total_blocks = src.len() / QK5_1;
    let mut out = Vec::with_capacity(total_blocks * BLOCK_BYTES);

    match imatrix {
        None => {
            quantize_row_q5_1_ref(src, &mut out);
        }
        Some(qw) => {
            let nrow = src.len() / n_per_row;
            for row in 0..nrow {
                let row_x = &src[row * n_per_row..(row + 1) * n_per_row];
                quantize_row_q5_1_impl(row_x, qw, &mut out);
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
        let input = read_f32s("q5_1_64_noim_input.bin");
        let expected = read_bytes("q5_1_64_noim_expected.bin");
        let got = quantize(&input, 64, None);
        assert_eq!(got.len(), expected.len(), "Q5_1 noim length mismatch");
        assert_eq!(got, expected, "Q5_1 noim byte-cmp failed");
    }

    #[test]
    fn byte_cmp_im() {
        let input = read_f32s("q5_1_64_im_input.bin");
        let expected = read_bytes("q5_1_64_im_expected.bin");
        // Harness `IMATRIX_SEED=2` (scripts/ggml_quants_harness/generate_all.sh:21).
        let imatrix = make_imatrix(64, 2);
        let got = quantize(&input, 64, Some(&imatrix));
        assert_eq!(got.len(), expected.len(), "Q5_1 im length mismatch");
        assert_eq!(got, expected, "Q5_1 im byte-cmp failed");
    }
}
