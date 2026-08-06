//! IQ4_XS quantizer — ADR-033 §Pi pure-Rust port of
//! `quantize_row_iq4_nl_impl(super_block_size=QK_K=256, block_size=32, ...)`
//! at `/opt/llama.cpp/ggml/src/ggml-quants.c:4794` (SHA pinned in
//! `data/llama_cpp_pin.txt`).
//!
//! IQ4_XS shares the inner kernel with [`super::iq4_nl`]; the only
//! difference is multi-sub-block scale packing. Where IQ4_NL has one
//! F16 scale per 32-element block, IQ4_XS has a single super-block F16
//! scale `d` plus a 6-bit per-sub-block scale stored as 4-bit
//! `scales_l` + 2-bit `scales_h` packed across 8 sub-blocks of 32
//! elements each. Total: 136 bytes per 256-element super-block = 4.25
//! bits per weight.
//!
//! Block layout from `ggml-common.h` (`block_iq4_xs`):
//! ```text
//! #define QK_K 256
//! typedef struct {
//!     ggml_half d;             // 2 bytes, F16 super-block scale
//!     uint16_t scales_h;       // 2 bytes, 2-bit×8 sub-block scale top
//!     uint8_t  scales_l[QK_K/64];  // 4 bytes (8 sub-blocks × 4 bits, two per byte)
//!     uint8_t  qs[QK_K/2];     // 128 bytes, nibble-packed codebook indices
//! } block_iq4_xs;            // 136 bytes total
//! ```
//!
//! Two callers in C:
//!   - `quantize_row_iq4_xs_ref` (`:4963`) → calls `quantize_iq4_xs` with
//!     `quant_weights = NULL`, ntry=7 inside.
//!   - `quantize_iq4_xs`         (`:4943`) → calls `_impl(QK_K, 32, ..., qw, ntry=7)`.
//!
//! Both paths use ntry=7 — the convert pipeline ALWAYS routes through
//! `ggml_quantize_chunk` → `quantize_iq4_xs`, with NULL `quant_weights`
//! when no imatrix is provided. So `ntry=7` is the production path on
//! both fixture variants and on every real on-disk conversion we emit.
//! We unify on that.

use half::f16;
use rayon::prelude::*;

use super::common::{best_index_int8, nearest_int, GROUP_MAX_EPS};

/// IQ4_XS super-block size — matches `QK_K` in `ggml-common.h:213`.
pub const QK_K: usize = 256;
/// Per-sub-block size — fixed at 32 by the C dispatcher's
/// `quantize_iq4_nl_impl(QK_K, 32, ...)` call.
pub const SUB_BLOCK: usize = 32;
/// Block byte layout: 2 (d:f16) + 2 (scales_h:u16) + 4 (scales_l[QK_K/64]) + 128 (qs[QK_K/2]).
pub const BLOCK_BYTES: usize = 2 + 2 + QK_K / 64 + QK_K / 2;

/// `kvalues_iq4nl` codebook (`ggml-common.h:1110`). Shared with
/// [`super::iq4_nl`] — the IQ4_XS kernel maps every per-sub-block
/// quantized 4-bit nibble through this same table at dequantize time.
/// Must be monotonic non-decreasing for `best_index_int8`'s bisection.
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Per-super-block kernel. Mirrors `quantize_row_iq4_nl_impl(QK_K, 32, ...)`
/// at `ggml-quants.c:4794` exactly, with the multi-sub-block branch at
/// `:4865-4887` always taken.
///
/// FMA-contraction policy: `.mul_add()` everywhere the C source emits
/// an `a*b+c` pattern in the hot loops. The §P1 closure (ADR-033)
/// established that rustc's `--release` default `fp-contract=off` plus
/// clang's `-O3 -march=native` default `-ffp-contract=on` create
/// systematic 1-ULP drift unless mirrored. IQ4_NL's byte-identity
/// closure (`src/quantize/ggml_quants/iq4_nl.rs:127-128 + 148-149`)
/// proved the contraction policy for THIS kernel — IQ4_XS reuses the
/// same inner kernel so the same policy applies.
fn quantize_block_iq4_xs(
    x: &[f32],                     // length QK_K = 256
    out: &mut [u8],                // length BLOCK_BYTES = 136
    values: &[i8],                 // KVALUES_IQ4NL
    quant_weights: Option<&[f32]>, // length QK_K when Some
    ntry: i32,
) {
    debug_assert_eq!(x.len(), QK_K);
    debug_assert_eq!(out.len(), BLOCK_BYTES);
    debug_assert_eq!(values.len(), 16);

    let super_block_size = QK_K;
    let block_size = SUB_BLOCK; // 32
    let n_sub = super_block_size / block_size; // 8

    // sigma2 = (2/super_block_size) * sum(x^2)  — :4801-4803
    // Same FMA contraction policy as iq4_nl.rs (verified by §P1 byte-identity).
    let mut sigma2 = 0.0f32;
    for &v in x.iter() {
        sigma2 = v.mul_add(v, sigma2);
    }
    sigma2 *= 2.0 / (super_block_size as f32);

    // q4 cleared and dh=0 at :4805-4806 — we zero `out` lazily in the
    // write at the end. Keep a working L buffer for the entire 256
    // elements (per-sub-block writes into L[ib*32..(ib+1)*32]).
    let mut l_buf = [0u8; QK_K];
    // Per-sub-block scales — :4796 `float * scales` (length n_sub).
    let mut scales = [0.0f32; 8];

    // Per-sub-block weight buffer (length block_size). Lives across
    // iterations but always fully overwritten — :4796 `float * weight`.
    let mut weight = [0.0f32; SUB_BLOCK];

    // ----- First pass: per-sub-block scale fit — :4808-4863 -----
    let mut max_scale = 0.0f32;
    let mut amax_scale = 0.0f32;
    for ib in 0..n_sub {
        let xb_start = ib * block_size;
        let xb = &x[xb_start..xb_start + block_size];
        if let Some(qw) = quant_weights {
            let qwb = &qw[xb_start..xb_start + block_size];
            // weight[j] = qw[j] * sqrt(sigma2 + xb[j]^2)  — :4814
            for j in 0..block_size {
                weight[j] = qwb[j] * (sigma2 + xb[j] * xb[j]).sqrt();
            }
        } else {
            // weight[j] = xb[j]^2  — :4816
            for j in 0..block_size {
                weight[j] = xb[j] * xb[j];
            }
        }

        // amax / max scan — :4818-4824
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for j in 0..block_size {
            let ax = xb[j].abs();
            if ax > amax {
                amax = ax;
                max = xb[j];
            }
        }
        if amax < GROUP_MAX_EPS {
            // :4825-4827 — zero-block: scales[ib] = 0, continue (L stays 0 here,
            // but the L re-fill at :4878 will overwrite with the codebook-closest
            // index later in the multi-sub-block branch).
            scales[ib] = 0.0;
            continue;
        }

        // Initial scale estimate — :4829-4830
        let mut d = if ntry > 0 {
            -max / (values[0] as f32)
        } else {
            max / (values[0] as f32)
        };
        let mut id = 1.0 / d;

        // First pass: fill Lb from id*xb[j], accumulate sumqx/sumq2 — :4831-4841
        let lb = &mut l_buf[xb_start..xb_start + block_size];
        let mut sumqx = 0.0f32;
        let mut sumq2 = 0.0f32;
        for j in 0..block_size {
            let al = id * xb[j];
            let l = best_index_int8(values, al);
            lb[j] = l as u8;
            let q = values[l] as f32;
            let w = weight[j];
            sumqx = (w * q).mul_add(xb[j], sumqx);
            sumq2 = (w * q).mul_add(q, sumq2);
        }
        d = if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0 };
        let mut best = d * sumqx;

        // Refinement loop — :4843-4857. `for (int itry = -ntry; itry <= ntry; ++itry)`.
        let lo = -ntry;
        let hi = ntry;
        for itry in lo..=hi {
            id = ((itry as f32) + (values[0] as f32)) / max;
            let mut sumqx_t = 0.0f32;
            let mut sumq2_t = 0.0f32;
            for j in 0..block_size {
                let al = id * xb[j];
                let l = best_index_int8(values, al);
                let q = values[l] as f32;
                let w = weight[j];
                sumqx_t = (w * q).mul_add(xb[j], sumqx_t);
                sumq2_t = (w * q).mul_add(q, sumq2_t);
            }
            if sumq2_t > 0.0 && sumqx_t * sumqx_t > best * sumq2_t {
                d = sumqx_t / sumq2_t;
                best = d * sumqx_t;
            }
        }

        scales[ib] = d;
        let abs_d = d.abs();
        if abs_d > amax_scale {
            amax_scale = abs_d;
            max_scale = d;
        }
    }

    // ----- Multi-sub-block branch — :4865-4887 -----
    // super_block_size/block_size = n_sub = 8 > 1, so this branch always taken.
    let mut scales_h_acc: u16 = 0;
    let mut scales_l_buf = [0u8; QK_K / 64]; // 4 bytes for 8 sub-blocks (2 sub-blocks per byte)

    let d = -max_scale / 32.0;
    let dh = f16::from_f32(d);
    let id_super = if d != 0.0 { 1.0 / d } else { 0.0 };

    for ib in 0..n_sub {
        // Quantize per-sub-block scale to 6-bit signed [-32, 31] — :4872-4873
        let mut l_signed = nearest_int(id_super * scales[ib]);
        if l_signed < -32 {
            l_signed = -32;
        } else if l_signed > 31 {
            l_signed = 31;
        }
        let dl = d * (l_signed as f32);
        let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };

        // Re-fill Lb using the QUANTIZED sub-block scale — :4876-4880.
        // UNCONDITIONAL — covers the all-zero case (idl=0 → best_index_int8(0)
        // returns the codebook index closest to 0, = 8 for KVALUES_IQ4NL).
        let xb_start = ib * block_size;
        let xb = &x[xb_start..xb_start + block_size];
        let lb = &mut l_buf[xb_start..xb_start + block_size];
        for j in 0..block_size {
            lb[j] = best_index_int8(values, idl * xb[j]) as u8;
        }

        // Pack the 6-bit scale: l = l_signed + 32 ∈ [0, 63] — :4881-4886.
        // Split into 4-bit l_l (low nibble of scales_l[ib/2]) and 2-bit l_h
        // (shifted into scales_h at 2*ib position).
        let l_unsigned: u8 = (l_signed + 32) as u8;
        let l_l = l_unsigned & 0xf;
        let l_h = l_unsigned >> 4; // 2 bits
        if ib % 2 == 0 {
            scales_l_buf[ib / 2] = l_l;
        } else {
            scales_l_buf[ib / 2] |= l_l << 4;
        }
        scales_h_acc |= (l_h as u16) << (2 * ib);
    }

    // Nibble-pack L into qs — :4898-4902.
    // for i in 0..n_sub:
    //   for j in 0..16:
    //     qs[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4);
    let mut qs = [0u8; QK_K / 2]; // 128
    for i in 0..n_sub {
        let base = i * SUB_BLOCK;
        for j in 0..16 {
            qs[16 * i + j] = l_buf[base + j] | (l_buf[base + 16 + j] << 4);
        }
    }

    // Emit block: [d:f16(2) | scales_h:u16(2) | scales_l(4) | qs(128)] = 136 bytes.
    out[0..2].copy_from_slice(&dh.to_le_bytes());
    out[2..4].copy_from_slice(&scales_h_acc.to_le_bytes());
    out[4..8].copy_from_slice(&scales_l_buf);
    out[8..].copy_from_slice(&qs);
}

/// Quantize an F32 buffer to IQ4_XS bytes.
///
/// `src.len()` must be a multiple of `n_per_row`, and `n_per_row` must
/// be a multiple of `QK_K = 256`. Both production paths invoke the
/// same `_impl` with `ntry=7` — the `quantize_iq4_xs` dispatcher at
/// `ggml-quants.c:4943` reached through `ggml_quantize_chunk`. The
/// only difference is whether the imatrix participates in
/// weighted scale selection (the `weight[j] = qw[j] * sqrt(...)`
/// branch above).
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
            "imatrix len {} must equal n_per_row {} (per-row weights)",
            im.len(),
            n_per_row,
        );
    }

    // `quantize_iq4_xs` (`ggml-quants.c:4954`) always passes ntry=7, mirroring
    // its IQ4_NL sibling. This is the path taken by `ggml_quantize_chunk`
    // used by our fixture harness and by every real conversion call site.
    let ntry: i32 = 7;
    let nblock_per_row = n_per_row / QK_K;
    let nrows = src.len() / n_per_row;
    let row_bytes = nblock_per_row * BLOCK_BYTES;
    // ADR-036 Layer A: rows are independent; the iq4_xs per-block kernel
    // writes into the slice the caller passes — no shared output state
    // between blocks. Parallelize across rows; mirrors iq4_nl.rs.
    let mut out = vec![0u8; nrows * row_bytes];
    out.par_chunks_exact_mut(row_bytes)
        .enumerate()
        .for_each(|(row, row_dst)| {
            let row_src = &src[row * n_per_row..(row + 1) * n_per_row];
            for ibl in 0..nblock_per_row {
                let xb = &row_src[ibl * QK_K..(ibl + 1) * QK_K];
                let qw_block = imatrix.map(|im| &im[ibl * QK_K..(ibl + 1) * QK_K]);
                let blk_out = &mut row_dst[ibl * BLOCK_BYTES..(ibl + 1) * BLOCK_BYTES];
                quantize_block_iq4_xs(xb, blk_out, &KVALUES_IQ4NL, qw_block, ntry);
            }
        });

    out
}

/// Dequantize IQ4_XS bytes back to F32.
///
/// Pure-Rust mirror of `dequantize_row_iq4_xs` at `ggml-quants.c:2667`.
/// Used by the round-trip unit test and (in the future) by any
/// debug/validation tool that wants to inspect the result of a
/// quantize pass.
#[cfg(test)]
pub fn dequantize(bytes: &[u8], n_per_row: usize) -> Vec<f32> {
    assert_eq!(bytes.len() % BLOCK_BYTES, 0);
    assert_eq!(n_per_row % QK_K, 0);
    let nblock_per_row = n_per_row / QK_K;
    let row_bytes = nblock_per_row * BLOCK_BYTES;
    assert_eq!(bytes.len() % row_bytes, 0);
    let nrows = bytes.len() / row_bytes;
    let mut out = vec![0.0f32; nrows * n_per_row];
    for row in 0..nrows {
        let row_src = &bytes[row * row_bytes..(row + 1) * row_bytes];
        let row_dst = &mut out[row * n_per_row..(row + 1) * n_per_row];
        for ibl in 0..nblock_per_row {
            let blk = &row_src[ibl * BLOCK_BYTES..(ibl + 1) * BLOCK_BYTES];
            let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
            let scales_l = &blk[4..8];
            let qs = &blk[8..];
            let dst_blk = &mut row_dst[ibl * QK_K..(ibl + 1) * QK_K];
            // dequantize_row_iq4_xs:2677-2686
            for ib in 0..(QK_K / 32) {
                // ls = scales_l[ib/2] >> 4*(ib%2) & 0xf | ((scales_h >> 2*ib) & 3) << 4
                let lo = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf;
                let hi = ((scales_h >> (2 * ib)) & 0x3) as u8;
                let ls = (lo | (hi << 4)) as i32;
                let dl = d * ((ls - 32) as f32);
                let qs_sub = &qs[16 * ib..16 * (ib + 1)];
                let dst_sub = &mut dst_blk[32 * ib..32 * (ib + 1)];
                for j in 0..16 {
                    dst_sub[j] = dl * (KVALUES_IQ4NL[(qs_sub[j] & 0xf) as usize] as f32);
                    dst_sub[j + 16] = dl * (KVALUES_IQ4NL[(qs_sub[j] >> 4) as usize] as f32);
                }
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
        let manifest =
            std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set by cargo test");
        PathBuf::from(manifest)
            .join("tests/fixtures/ggml_quants")
            .join(name)
    }

    fn read_f32s(name: &str) -> Vec<f32> {
        let bytes = fs::read(fixture_path(name)).expect("read fixture");
        assert!(bytes.len() % 4 == 0);
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn read_bytes(name: &str) -> Vec<u8> {
        fs::read(fixture_path(name)).expect("read fixture")
    }

    /// Same mulberry32 PRNG used by the fixture harness for imatrix
    /// synthesis (`scripts/ggml_quants_harness/gen.c:42-49`).
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

    /// **§P1 ship gate**: byte-cmp against the canonical
    /// `ggml_quantize_chunk(GGML_TYPE_IQ4_XS, ...)` output. Synthetic
    /// fixtures lie (per ADR-033 §P1 closure FMA-non-associativity
    /// learnings); this is the actual ship gate.
    #[test]
    fn byte_cmp_noim() {
        let input = read_f32s("iq4_xs_512_noim_input.bin");
        let expected = read_bytes("iq4_xs_512_noim_expected.bin");
        let got = quantize(&input, 512, None);
        assert_eq!(got.len(), expected.len(), "IQ4_XS noim length mismatch");
        assert_eq!(got, expected, "IQ4_XS noim byte-cmp failed");
    }

    #[test]
    fn byte_cmp_im() {
        let input = read_f32s("iq4_xs_512_im_input.bin");
        let expected = read_bytes("iq4_xs_512_im_input.bin"); // input bytes; expected is the next file
        let _ = expected; // keep clippy quiet; we override below
        let expected = read_bytes("iq4_xs_512_im_expected.bin");
        let imatrix = make_imatrix(512, 2);
        let got = quantize(&input, 512, Some(&imatrix));
        assert_eq!(got.len(), expected.len(), "IQ4_XS im length mismatch");
        assert_eq!(got, expected, "IQ4_XS im byte-cmp failed");
    }

    /// Round-trip sanity: quantize → dequantize should land within a
    /// reasonable tolerance of the input. Not byte-identity (lossy
    /// quant); this just proves the dequantize matches the C layout.
    #[test]
    fn round_trip_sanity() {
        // Synthetic 1-row input, well within range.
        let input: Vec<f32> = (0..QK_K)
            .map(|i| {
                let t = i as f32 / QK_K as f32;
                0.5 * (t * std::f32::consts::TAU * 4.0).sin()
            })
            .collect();
        let bytes = quantize(&input, QK_K, None);
        assert_eq!(bytes.len(), BLOCK_BYTES);
        let recovered = dequantize(&bytes, QK_K);
        assert_eq!(recovered.len(), QK_K);
        // RMSE should be well below 0.05 (4.25 bpw on a smooth signal).
        let mut sse = 0.0f64;
        for j in 0..QK_K {
            let e = (recovered[j] - input[j]) as f64;
            sse += e * e;
        }
        let rmse = (sse / QK_K as f64).sqrt();
        assert!(rmse < 0.05, "IQ4_XS round-trip RMSE too high: {rmse}");
    }

    /// All-zero input → kernel writes the codebook-closest-to-zero
    /// index (=8 in KVALUES_IQ4NL) for every position. Dequantize
    /// then yields 0.0 because every sub-block scale is also zero.
    #[test]
    fn all_zero_block() {
        let input = vec![0.0f32; QK_K];
        let bytes = quantize(&input, QK_K, None);
        let recovered = dequantize(&bytes, QK_K);
        for v in recovered.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    /// Multi-row, multi-block-per-row dispatcher exercise.
    #[test]
    fn multi_row_multi_block() {
        let n_per_row = QK_K * 3;
        let nrows = 2;
        let input: Vec<f32> = (0..nrows * n_per_row)
            .map(|i| ((i as f32) * 0.001).sin() * 0.3)
            .collect();
        let bytes = quantize(&input, n_per_row, None);
        let expected_bytes = nrows * (n_per_row / QK_K) * BLOCK_BYTES;
        assert_eq!(bytes.len(), expected_bytes);
        let recovered = dequantize(&bytes, n_per_row);
        assert_eq!(recovered.len(), nrows * n_per_row);
        // Sanity: max abs error well within IQ4_XS's nominal precision.
        let max_err = input
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.1, "max abs error too high: {max_err}");
    }
}
