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
use rayon::prelude::*;

use super::common::{make_qx_quants, nearest_int, GROUP_MAX_EPS};

pub const QK_K: usize = 256;
pub const BLOCK_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2; // 128 + 64 + 16 + 2 = 210

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
    let row_bytes = row_blocks * BLOCK_BYTES;
    // ADR-036 Layer A: per-row parallelism via rayon.
    let mut out = vec![0u8; n_rows * row_bytes];
    out.par_chunks_exact_mut(row_bytes)
        .enumerate()
        .for_each(|(row, dst)| {
            let row_src = &src[row * n_per_row..(row + 1) * n_per_row];
            let mut tmp = Vec::with_capacity(row_bytes);
            match imatrix {
                None => quantize_row_ref(row_src, &mut tmp),
                Some(qw) => quantize_row_impl(row_src, qw, &mut tmp),
            }
            debug_assert_eq!(tmp.len(), row_bytes);
            dst.copy_from_slice(&tmp);
        });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    /// 2026-05-20 broad-sample diagnostic — confirm the make_qx_quants
    /// fix at b921616e is globally robust on Qwen 3.5 Q6_K, not just
    /// output.weight row 149. Samples 12 rows across 4 Q6_K tensors.
    #[test]
    #[ignore]
    fn qwen35_q6k_broad_sample_dump() {
        let cases: &[(&str, usize, &[usize])] = &[
            ("output_weight", 2048, &[0, 149, 100000]),
            ("blk_0_attn_qkv_weight", 2048, &[0, 100, 1000]),
            ("blk_0_ffn_down_exps_weight", 512, &[0, 1000, 50000]),
            ("blk_10_attn_qkv_weight", 2048, &[0, 500, 1500]),
        ];
        let mut grand_total_diff = 0usize;
        let mut grand_total = 0usize;
        for (name, n_per_row, rows) in cases {
            for &row_idx in *rows {
                let f32_path = format!("/tmp/c_quant_repro/qwen35_{}_row{}_f32.bin", name, row_idx);
                let q6k_path = format!("/tmp/c_quant_repro/qwen35_{}_row{}_q6k.bin", name, row_idx);
                let f32_bytes =
                    std::fs::read(&f32_path).unwrap_or_else(|_| panic!("{} must exist", f32_path));
                let canonical =
                    std::fs::read(&q6k_path).unwrap_or_else(|_| panic!("{} must exist", q6k_path));
                let n_blocks = n_per_row / 256;
                assert_eq!(f32_bytes.len(), n_per_row * 4);
                assert_eq!(canonical.len(), n_blocks * BLOCK_BYTES);
                let f32: Vec<f32> = f32_bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let q6k = quantize(&f32, *n_per_row, None);
                let diff: usize = q6k
                    .iter()
                    .zip(canonical.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                if diff > 0 {
                    println!("{} row {}: {} bytes differ", name, row_idx, diff);
                }
                grand_total_diff += diff;
                grand_total += q6k.len();
            }
        }
        println!(
            "\nQ6_K BROAD GRAND TOTAL: {}/{} bytes differ ({:.6}%)",
            grand_total_diff,
            grand_total,
            100.0 * grand_total_diff as f64 / grand_total as f64
        );
    }

    /// One-off bisection harness for the Qwen 3.5 lm_head Q6_K block 1199 residual.
    /// Reads the exact 2048 F32 values that canonical reads for row 149 of
    /// output.weight (extracted from canonical's F16 GGUF) and prints what
    /// hf2q's Q6_K kernel produces for block 7. Compared against canonical's
    /// linked-library output (scales[12] should be -61/195).
    #[test]
    #[ignore]
    fn qwen35_lm_head_row149_q6k_block7_dump() {
        let bytes = std::fs::read("/tmp/c_quant_repro/row149_canonical_f32.bin")
            .expect("/tmp/c_quant_repro/row149_canonical_f32.bin must exist");
        assert_eq!(bytes.len(), 8192, "expected 2048 F32 = 8192 bytes");
        let mut f32_vals = Vec::with_capacity(2048);
        for chunk in bytes.chunks_exact(4) {
            f32_vals.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        let q6k_bytes = quantize(&f32_vals, 2048, None);
        assert_eq!(q6k_bytes.len(), 8 * BLOCK_BYTES);
        let blk7 = &q6k_bytes[7 * BLOCK_BYTES..8 * BLOCK_BYTES];
        let ql_hex: String = blk7[..8].iter().map(|b| format!("{:02x}", b)).collect();
        let scales_i8: Vec<i8> = blk7[192..208].iter().map(|&b| b as i8).collect();
        let scales_u8: Vec<u8> = blk7[192..208].to_vec();
        println!("hf2q row149 block7: ql[..8]={}", ql_hex);
        println!("hf2q row149 block7: scales i8 = {:?}", scales_i8);
        println!("hf2q row149 block7: scales u8 = {:?}", scales_u8);
        println!(
            "hf2q row149 block7: d (f16) = {:02x} {:02x}",
            blk7[208], blk7[209]
        );
        // Canonical reference: scales u8 = [81, 87, 68, 164, 145, 72, 104, 196, 67, 50, 170, 100, 195, 160, 200, 128]
        let canon: Vec<u8> = vec![
            81, 87, 68, 164, 145, 72, 104, 196, 67, 50, 170, 100, 195, 160, 200, 128,
        ];
        for ib in 0..16 {
            if scales_u8[ib] != canon[ib] {
                println!(
                    "  MISMATCH ib={} hf2q={} canon={} delta={}",
                    ib,
                    scales_u8[ib],
                    canon[ib],
                    scales_u8[ib] as i32 - canon[ib] as i32
                );
            }
        }
    }

    fn fixture_path(name: &str) -> PathBuf {
        let manifest =
            std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set by cargo test");
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
