//! IQ4_NL quantizer — ADR-033 P0 pure-Rust port of
//! `quantize_row_iq4_nl_impl` at `/opt/llama.cpp/ggml/src/ggml-quants.c:4794`
//! (SHA pinned in `data/llama_cpp_pin.txt`).
//!
//! Block layout from `ggml-common.h` (`block_iq4_nl`):
//! ```text
//! #define QK4_NL 32
//! typedef struct {
//!     ggml_half d;             // 2 bytes, F16, little-endian
//!     uint8_t   qs[QK4_NL/2];  // 16 bytes, nibble-packed codebook indices
//! } block_iq4_nl;            // 18 bytes total
//! ```
//!
//! Codebook `kvalues_iq4nl[16]` from `ggml-common.h:1110-1112` (16 signed
//! int8 values, monotonically non-decreasing — required by
//! `best_index_int8`'s binary search at `ggml-quants.c:24`).
//!
//! Two callers in C:
//!   - `quantize_row_iq4_nl_ref` (`:4928`) → calls `_impl(32, 32, …, NULL, ntry=-1)`
//!   - `quantize_iq4_nl`         (`:4905`) → calls `_impl(32, 32, …, qw,   ntry= 7)`
//!
//! Our fixture harness at `scripts/ggml_quants_harness/gen.c` writes via
//! `ggml_quantize_chunk`, which dispatches to `quantize_iq4_nl` for both
//! the imatrix-present and imatrix-absent cases (passing NULL for the
//! latter). So `ntry=7` is the production path on both fixture variants
//! and on every real on-disk conversion we emit. We unify on that.

use half::f16;

use super::common::{best_index_int8, GROUP_MAX_EPS};

pub const QK4_NL: usize = 32;
pub const BLOCK_BYTES: usize = 2 + QK4_NL / 2; // 18

/// `kvalues_iq4nl` codebook (`ggml-common.h:1110`). Must be monotonic
/// non-decreasing for `best_index_int8`'s bisection to be correct.
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Pure-Rust mirror of `quantize_row_iq4_nl_impl` at `ggml-quants.c:4794`,
/// specialized to `super_block_size == block_size == QK4_NL == 32` (the
/// only configuration used by both callers — `quantize_iq4_nl` at :4905
/// and `quantize_row_iq4_nl_ref` at :4928). In that single-block regime
/// `scales_h` / `scales_l` are unused, so we drop them. The final
/// nibble-pack at :4898-4902 collapses to one `i==0` iteration.
fn quantize_block_iq4_nl(
    x: &[f32],                  // length QK4_NL
    out: &mut [u8],             // length BLOCK_BYTES (18)
    values: &[i8],              // KVALUES_IQ4NL
    quant_weights: Option<&[f32]>, // length QK4_NL when Some
    ntry: i32,
) {
    debug_assert_eq!(x.len(), QK4_NL);
    debug_assert_eq!(out.len(), BLOCK_BYTES);
    let super_block_size = QK4_NL;
    let block_size = QK4_NL;

    // sigma2 = (2/super_block_size) * sum(x^2)  — :4801-4803
    let mut sigma2 = 0.0f32;
    for &v in x.iter() {
        sigma2 += v * v;
    }
    sigma2 *= 2.0 / (super_block_size as f32);

    // q4 cleared and dh=0 at :4805-4806 — we zero `out` lazily below.
    let mut q4 = [0u8; QK4_NL / 2];

    // Per-block weight buffer (length block_size).
    let mut weight = [0.0f32; QK4_NL];
    let mut l_buf = [0u8; QK4_NL];

    // ----- ib=0 (the only iteration) — :4809-4863 -----
    let xb = x;
    if let Some(qw) = quant_weights {
        // weight[j] = qw[j] * sqrt(sigma2 + xb[j]^2)  — :4814
        for j in 0..block_size {
            weight[j] = qw[j] * (sigma2 + xb[j] * xb[j]).sqrt();
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

    // scales[0] init — overwritten below; only used in :4888 branch.
    let mut scale0 = 0.0f32;
    let all_zero = amax < GROUP_MAX_EPS;

    if !all_zero {
        // Initial scale estimate — :4829-4830
        let mut d = if ntry > 0 {
            -max / (values[0] as f32)
        } else {
            max / (values[0] as f32)
        };
        let mut id = 1.0 / d;

        // First pass: fill L from id*xb[j], accumulate weighted sumqx/sumq2 — :4831-4841
        let mut sumqx = 0.0f32;
        let mut sumq2 = 0.0f32;
        for j in 0..block_size {
            let al = id * xb[j];
            let l = best_index_int8(values, al);
            l_buf[j] = l as u8;
            let q = values[l] as f32;
            let w = weight[j];
            sumqx += w * q * xb[j];
            sumq2 += w * q * q;
        }
        d = if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0 };
        let mut best = d * sumqx;

        // Refinement loop — :4843-4857. ntry<0 ⇒ empty range.
        // C: `for (int itry = -ntry; itry <= ntry; ++itry)`.
        let lo = -ntry;
        let hi = ntry;
        if lo <= hi {
            for itry in lo..=hi {
                // id = (itry + values[0]) / max  — :4844
                id = ((itry as f32) + (values[0] as f32)) / max;
                let mut sumqx_t = 0.0f32;
                let mut sumq2_t = 0.0f32;
                for j in 0..block_size {
                    let al = id * xb[j];
                    let l = best_index_int8(values, al);
                    let q = values[l] as f32;
                    let w = weight[j];
                    sumqx_t += w * q * xb[j];
                    sumq2_t += w * q * q;
                }
                if sumq2_t > 0.0 && sumqx_t * sumqx_t > best * sumq2_t {
                    d = sumqx_t / sumq2_t;
                    best = d * sumqx_t;
                }
            }
        }

        scale0 = d;
        // max_scale tracking at :4859-4862 is dead in the single-block
        // branch (only the multi-sub-block branch at :4865 reads it).
    }

    // super_block_size == block_size ⇒ single-block branch :4888-4896
    let d_final = scale0;
    let dh = f16::from_f32(d_final);

    if !all_zero && ntry > 0 {
        // Final L re-fill from d_final — :4890-4895
        let id = if scale0 != 0.0 { 1.0 / scale0 } else { 0.0 };
        for j in 0..super_block_size {
            l_buf[j] = best_index_int8(values, id * x[j]) as u8;
        }
    }
    // If all_zero: L stayed all-zero from init, q4 will be all-zero
    // (and dh=0 since scale0 stays 0) — matches :4805-4806 invariant.

    // Nibble-pack — :4898-4902, single iteration i=0
    for j in 0..16 {
        q4[j] = l_buf[j] | (l_buf[16 + j] << 4);
    }

    // Emit block: 2-byte f16 d, then 16 bytes q4
    out[..2].copy_from_slice(&dh.to_le_bytes());
    out[2..].copy_from_slice(&q4);
}

/// Quantize an F32 buffer to IQ4_NL bytes.
///
/// `src.len()` must be a multiple of `n_per_row`, and `n_per_row` must
/// be a multiple of `QK4_NL`. Both paths invoke the same `_impl` with
/// `ntry=7` — the `quantize_iq4_nl` dispatcher at `ggml-quants.c:4905`
/// reached through `ggml_quantize_chunk`. The only difference is whether
/// the imatrix participates in weighted scale selection.
pub fn quantize(src: &[f32], n_per_row: usize, imatrix: Option<&[f32]>) -> Vec<u8> {
    assert!(
        n_per_row % QK4_NL == 0,
        "n_per_row {} not multiple of QK4_NL {}",
        n_per_row,
        QK4_NL
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

    // `quantize_iq4_nl` (`ggml-quants.c:4919`) always passes ntry=7, even
    // when the dispatcher's `quant_weights` argument is NULL. This is the
    // path taken by `ggml_quantize_chunk` used by our fixture harness and
    // by every real conversion call site.
    let ntry: i32 = 7;
    let nblock_per_row = n_per_row / QK4_NL;
    let nrows = src.len() / n_per_row;
    let mut out = vec![0u8; nrows * nblock_per_row * BLOCK_BYTES];

    for row in 0..nrows {
        let row_src = &src[row * n_per_row..(row + 1) * n_per_row];
        let row_out_start = row * nblock_per_row * BLOCK_BYTES;
        for ibl in 0..nblock_per_row {
            let xb = &row_src[ibl * QK4_NL..(ibl + 1) * QK4_NL];
            let qw_block = imatrix.map(|im| &im[ibl * QK4_NL..(ibl + 1) * QK4_NL]);
            let blk_start = row_out_start + ibl * BLOCK_BYTES;
            let blk_out = &mut out[blk_start..blk_start + BLOCK_BYTES];
            quantize_block_iq4_nl(xb, blk_out, &KVALUES_IQ4NL, qw_block, ntry);
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

    /// Same Mulberry32-flavor PRNG used by the fixture harness
    /// (`scripts/ggml_quants_harness/gen.c:24-30`,42-49) for imatrix
    /// synthesis. `n_per_row = 64`, seed = `IMATRIX_SEED = 2` (per
    /// `generate_all.sh:21`).
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
        let input = read_f32s("iq4_nl_64_noim_input.bin");
        let expected = read_bytes("iq4_nl_64_noim_expected.bin");
        let got = quantize(&input, 64, None);
        assert_eq!(got.len(), expected.len(), "IQ4_NL noim length mismatch");
        assert_eq!(got, expected, "IQ4_NL noim byte-cmp failed");
    }

    #[test]
    fn byte_cmp_im() {
        let input = read_f32s("iq4_nl_64_im_input.bin");
        let expected = read_bytes("iq4_nl_64_im_expected.bin");
        let imatrix = make_imatrix(64, 2);
        let got = quantize(&input, 64, Some(&imatrix));
        assert_eq!(got.len(), expected.len(), "IQ4_NL im length mismatch");
        assert_eq!(got, expected, "IQ4_NL im byte-cmp failed");
    }
}
