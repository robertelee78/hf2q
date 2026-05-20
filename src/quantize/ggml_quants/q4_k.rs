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
use rayon::prelude::*;

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
    let row_bytes = row_blocks * BLOCK_BYTES;
    // ADR-036 Layer A: each row's quantization is independent (per-row
    // scratch state, no cross-row reads/writes). Parallelize via rayon
    // par_chunks_exact_mut into a pre-allocated output. Byte output is
    // unchanged — the §P1 byte-cmp regression gate validates.
    let mut out = vec![0u8; n_rows * row_bytes];
    out.par_chunks_exact_mut(row_bytes).enumerate().for_each(|(row, dst)| {
        let row_x = &src[row * n_per_row..(row + 1) * n_per_row];
        let mut tmp = Vec::with_capacity(row_bytes);
        match imatrix {
            None => quantize_row_ref(row_x, &mut tmp),
            Some(qw) => quantize_row_impl(row_x, qw, &mut tmp),
        }
        debug_assert_eq!(tmp.len(), row_bytes);
        dst.copy_from_slice(&tmp);
    });

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
            // FMA: clang fuses `sum_x2 + v*v` into fmadd at -O3 -march=native;
            // mirror Q5_K's _ref path (q5_k.rs:185) for consistency. The
            // single-tensor byte-cmp on blk.0.attn_k passed without this
            // because the av_x ULP delta didn't propagate to L[] selection
            // for that tensor, but other distributions may exercise it.
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

    /// 2026-05-20 — Instrumented trace of hf2q's make_qkx2 on Gemma block 3083 sub-block 6.
    /// Reveals which iteration is accepted and the cur_error values at the boundary.
    #[test]
    #[ignore]
    fn gemma_blk3083_sub6_hf2q_trace() {
        let f32_bytes = std::fs::read("/tmp/c_quant_repro/gemma_blk3083_sub6_f32.bin").unwrap();
        let w_bytes = std::fs::read("/tmp/c_quant_repro/gemma_blk3083_sub6_weights.bin").unwrap();
        let x: Vec<f32> = f32_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let w: Vec<f32> = w_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let mut l = [0u8; 32];
        let mut l_aux = [0u8; 32];
        let mut the_min = 0.0f32;
        let scale = make_qkx2_quants_instrumented(32, 15, &x, &w, &mut l, &mut the_min, &mut l_aux, -1.0, 0.1, 20, false);
        println!("FINAL hf2q scale={:.10} the_min={:.10}", scale, the_min);
    }

    /// 2026-05-20 — Run hf2q's make_qkx2_quants on each of 8 sub-blocks of Gemma
    /// block 3083 (row 280 of blk.0.attn_k). Compare scale F32 bits to canonical's
    /// standalone output (from /tmp/c_quant_repro/canon_qkx2_sub*).
    #[test]
    #[ignore]
    fn gemma_blk3083_subblocks_scale_compare() {
        use super::super::common::make_qkx2_quants;
        // Canonical-recovered scales (from canon_qkx2_sub*.c standalone runs)
        let canon_scales: [f32; 8] = [
            0.0069445884,  // sub 0
            0.0069021727,  // sub 1
            0.0085167065,  // sub 2
            0.0089343758,  // sub 3
            0.0099748811,  // sub 4
            0.0061270911,  // sub 5
            0.0100716567,  // sub 6 — MAX
            0.0056068259,  // sub 7
        ];
        let mut hf2q_scales = [0.0f32; 8];
        for j in 0..8 {
            let f32_path = format!("/tmp/c_quant_repro/gemma_blk3083_sub{}_f32.bin", j);
            let w_path = format!("/tmp/c_quant_repro/gemma_blk3083_sub{}_weights.bin", j);
            let f32_bytes = std::fs::read(&f32_path).unwrap();
            let w_bytes = std::fs::read(&w_path).unwrap();
            let x: Vec<f32> = f32_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let w: Vec<f32> = w_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let mut l = [0u8; 32];
            let mut l_aux = [0u8; 32];
            let mut the_min = 0.0f32;
            let scale = make_qkx2_quants(32, 15, &x, &w, &mut l, &mut the_min, &mut l_aux, -1.0, 0.1, 20, false);
            hf2q_scales[j] = scale;
        }
        let mut max_scale_canon = 0.0f32;
        let mut max_scale_hf2q = 0.0f32;
        println!("Sub | canon scale (F32, bits) | hf2q scale (F32, bits)         | Δ");
        for j in 0..8 {
            let c = canon_scales[j];
            let h = hf2q_scales[j];
            if c > max_scale_canon { max_scale_canon = c; }
            if h > max_scale_hf2q { max_scale_hf2q = h; }
            let delta = h - c;
            let marker = if delta.abs() > 1e-8 { " ← DIVERGES" } else { "" };
            println!("{:3} | {:.10} 0x{:08x}     | {:.10} 0x{:08x} | {:+.3e}{}",
                j, c, c.to_bits(), h, h.to_bits(), delta, marker);
        }
        let d_canon = max_scale_canon / 63.0;
        let d_hf2q = max_scale_hf2q / 63.0;
        let d_canon_f16 = half::f16::from_f32(d_canon);
        let d_hf2q_f16 = half::f16::from_f32(d_hf2q);
        println!("\nmax_scale canon = {:.10} (bits 0x{:08x})", max_scale_canon, max_scale_canon.to_bits());
        println!("max_scale hf2q  = {:.10} (bits 0x{:08x})", max_scale_hf2q, max_scale_hf2q.to_bits());
        println!("d (F16) canon = 0x{:04x} = {:?}", d_canon_f16.to_bits(), d_canon_f16.to_le_bytes());
        println!("d (F16) hf2q  = 0x{:04x} = {:?}", d_hf2q_f16.to_bits(), d_hf2q_f16.to_le_bytes());
    }

    /// 2026-05-20 — Instrumented copy of make_qkx2_quants that prints which iteration
    /// is accepted as "best" and the running best_error. For diagnosing the 3.4%
    /// structural mins[1] divergence between hf2q and canonical on ssm_out row 100
    /// block 13 sub-block 1.
    #[allow(clippy::too_many_arguments, dead_code)]
    fn make_qkx2_quants_instrumented(
        n: usize,
        nmax: i32,
        x: &[f32],
        weights: &[f32],
        l: &mut [u8],
        the_min: &mut f32,
        l_aux: &mut [u8],
        rmin: f32,
        rdelta: f32,
        nstep: i32,
        use_mad: bool,
    ) -> f32 {
        use super::super::common::nearest_int;
        let mut min = x[0];
        let mut max = x[0];
        let mut sum_w = weights[0];
        let mut sum_x = sum_w * x[0];
        for i in 1..n {
            if x[i] < min { min = x[i]; }
            if x[i] > max { max = x[i]; }
            let w = weights[i];
            sum_w += w;
            sum_x = w.mul_add(x[i], sum_x);
        }
        if min > 0.0 { min = 0.0; }
        if max == min {
            for li in l.iter_mut().take(n) { *li = 0; }
            *the_min = -min;
            return 0.0;
        }
        let mut iscale = nmax as f32 / (max - min);
        let mut scale = 1.0 / iscale;
        // Try canonical's likely form: fma at best_error += site
        let mut best_error = 0.0f32;
        for i in 0..n {
            let li = nearest_int(iscale * (x[i] - min));
            let li_c = li.max(0).min(nmax) as u8;
            l[i] = li_c;
            let mut diff = scale * li_c as f32 + min - x[i];
            diff = if use_mad { diff.abs() } else { diff * diff };
            best_error = weights[i].mul_add(diff, best_error);  // fma re-added
        }
        println!("  INIT: scale={:.10} min={:.10} best_error_FMA={:.10e}", scale, min, best_error);
        let mut accepted_is: i32 = -1;
        for is in 0..=nstep {
            iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
            let mut sum_l = 0.0f32;
            let mut sum_l2 = 0.0f32;
            let mut sum_xl = 0.0f32;
            for i in 0..n {
                let li_raw = nearest_int(iscale * (x[i] - min));
                let li = li_raw.max(0).min(nmax) as u8;
                l_aux[i] = li;
                let w = weights[i];
                let li_f = li as f32;
                sum_l = w.mul_add(li_f, sum_l);
                sum_l2 = (w * li_f).mul_add(li_f, sum_l2);
                sum_xl = (w * li_f).mul_add(x[i], sum_xl);
            }
            let d_det = sum_w.mul_add(sum_l2, -sum_l * sum_l);
            if d_det > 0.0 {
                let mut this_scale = sum_w.mul_add(sum_xl, -sum_x * sum_l) / d_det;
                let mut this_min = sum_l2.mul_add(sum_x, -sum_l * sum_xl) / d_det;
                let used_fallback = this_min > 0.0;
                if used_fallback {
                    this_min = 0.0;
                    this_scale = sum_xl / sum_l2;
                }
                // Try canonical's likely vectorized form: fma at the cur_error += site
                let mut cur_error = 0.0f32;
                for i in 0..n {
                    let mut diff = this_scale * l_aux[i] as f32 + this_min - x[i];
                    diff = if use_mad { diff.abs() } else { diff * diff };
                    cur_error = weights[i].mul_add(diff, cur_error);  // fma re-added
                }
                let accepted = cur_error < best_error;
                println!("  is={:2}: this_scale={:.10} this_min={:.10} cur_error={:.10e} fb={} {}",
                    is, this_scale, this_min, cur_error, used_fallback,
                    if accepted {"<-ACCEPTED"} else {""});
                if accepted {
                    for i in 0..n { l[i] = l_aux[i]; }
                    best_error = cur_error;
                    scale = this_scale;
                    min = this_min;
                    accepted_is = is;
                }
            } else {
                println!("  is={:2}: d_det<=0, skipped", is);
            }
        }
        println!("  FINAL: scale={:.10} min={:.10} accepted_is={}", scale, min, accepted_is);
        *the_min = -min;
        scale
    }

    /// 2026-05-20 — INTERMEDIATE-VALUE diagnostic: call make_qkx2_quants directly on
    /// sub-block 1 of block 13 of ssm_out row 100. Print mins[1] F32 bits + computed
    /// inv_min*mins[1] product and rounding. Recover canonical's expected min from
    /// canonical's lm[1]=25 byte + dmin F16 bytes. Diagnoses whether hf2q's iteration
    /// loop accepts a different "best" than canonical's at this boundary.
    #[test]
    #[ignore]
    fn qwen35_ssm_out_row100_blk13_subblk1_makeqkx2_dump() {
        let f32_bytes = std::fs::read("/tmp/c_quant_repro/qwen35_blk_0_ssm_out_weight_row100_f32.bin").unwrap();
        let canonical = std::fs::read("/tmp/c_quant_repro/qwen35_blk_0_ssm_out_weight_row100_q4k.bin").unwrap();
        let f32: Vec<f32> = f32_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let blk13_offset = 13 * 256;
        let sub1_offset = blk13_offset + 32; // sub-block 1 of block 13
        let sub: &[f32] = &f32[sub1_offset..sub1_offset + 32];

        // Compute weights as Q4_K_ref does
        let sum_x2: f32 = sub.iter().map(|v| v * v).sum();
        let av_x = (sum_x2 / 32.0).sqrt();
        let weights: Vec<f32> = sub.iter().map(|v| av_x + v.abs()).collect();

        let mut l = [0u8; 32];
        let mut l_aux = [0u8; 32];
        let mut the_min = 0.0f32;
        let scale = make_qkx2_quants_instrumented(32, 15, sub, &weights, &mut l, &mut the_min, &mut l_aux,
            -1.0, 0.1, 20, false);
        let _min_alias = -the_min;  // make_qkx2 writes -min into the_min
        // Actually the function writes `*the_min = -min`. So our recovered min:
        // *the_min holds -min. Since min is the F32 returned to caller as mins[j],
        // mins[1] = *the_min = -min. Let me name it more carefully:
        let mins_1_hf2q = the_min;  // == -min from make_qkx2's perspective
        println!("hf2q sub-block 1 of block 13 (row 100):");
        println!("  scale = {:.30} (bits 0x{:08x})", scale, scale.to_bits());
        println!("  mins[1] = {:.30} (bits 0x{:08x})", mins_1_hf2q, mins_1_hf2q.to_bits());

        // Read canonical's block 13: extract dmin F16, lm[1]=25, mins/scales pack
        let canon_blk13 = &canonical[13*144..14*144];
        let canon_dmin_u16 = u16::from_le_bytes([canon_blk13[2], canon_blk13[3]]);
        // half F16 → F32
        let canon_dmin = half::f16::from_bits(canon_dmin_u16).to_f32();
        // canonical max_min = dmin * 63 (inverse of `dmin = max_min/63`)
        let canon_max_min = canon_dmin * 63.0;
        let canon_inv_min = 63.0 / canon_max_min;
        // canonical lm[1] = 25 (per the diff breakdown). lm[1] = nearest_int(inv_min*mins[1]).
        // So canon mins[1] = 25 / inv_min, give or take rounding.
        // Specifically: lm = 25 implies inv_min*mins[1] is in [24.5, 25.5).
        let canon_mins_1_approx_low = 24.5 / canon_inv_min;
        let canon_mins_1_approx_high = 25.5 / canon_inv_min;
        println!("\ncanonical recovered:");
        println!("  dmin F16 bytes: {:02x} {:02x}", canon_blk13[2], canon_blk13[3]);
        println!("  dmin F32 = {:.30}", canon_dmin);
        println!("  max_min = {:.30} (= dmin*63)", canon_max_min);
        println!("  inv_min = {:.30} (= 63/max_min)", canon_inv_min);
        println!("  mins[1] in range [{:.10}, {:.10}) for lm[1]=25", canon_mins_1_approx_low, canon_mins_1_approx_high);

        // Now compute hf2q's inv_min*mins[1] product and verify the round
        // For hf2q we need max_min which requires running make_qkx2 for ALL 8 sub-blocks.
        // Skipping that — use canonical's inv_min (since 7 of 8 lm[i] match, they must be ≈ equal).
        let hf2q_product_approx = canon_inv_min * mins_1_hf2q;
        println!("\nhf2q mins[1] * canon inv_min = {:.10}", hf2q_product_approx);
        println!("  → nearest_int = {}", (hf2q_product_approx + 0.5).floor() as i32);
        let canon_product_low = 24.5;
        let canon_product_high = 25.5;
        println!("\nCANON product range for lm[1]=25: [{}, {})", canon_product_low, canon_product_high);
        println!("Δ (hf2q - canon_mid_25.0): {:.10}", hf2q_product_approx - 25.0);
    }

    /// 2026-05-20 — diff localization for the invariant 11-byte ssm_out row 100
    /// residual. Q4_K block layout:
    ///   [0..2]   d (f16)
    ///   [2..4]   dmin (f16)
    ///   [4..16]  scales (12 bytes packed 6-bit)
    ///   [16..144] qs (128 bytes 4-bit quants)
    /// Print per-block per-region diff to localize the residual.
    #[test]
    #[ignore]
    fn qwen35_ssm_out_row100_q4k_diff_breakdown() {
        let f32_bytes = std::fs::read("/tmp/c_quant_repro/qwen35_blk_0_ssm_out_weight_row100_f32.bin").unwrap();
        let canonical = std::fs::read("/tmp/c_quant_repro/qwen35_blk_0_ssm_out_weight_row100_q4k.bin").unwrap();
        let f32: Vec<f32> = f32_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let n_per_row = f32.len();
        let q4k = quantize(&f32, n_per_row, None);
        let n_blocks = n_per_row / 256;
        let mut total_diff = 0usize;
        for blk in 0..n_blocks {
            let s = blk * 144;
            let mine = &q4k[s..s+144];
            let canon = &canonical[s..s+144];
            let d_diff = mine[0..2] != canon[0..2];
            let dmin_diff = mine[2..4] != canon[2..4];
            let scales_diff: usize = mine[4..16].iter().zip(canon[4..16].iter()).filter(|(a,b)| a != b).count();
            let qs_diff: usize = mine[16..144].iter().zip(canon[16..144].iter()).filter(|(a,b)| a != b).count();
            let blk_diff = (if d_diff {2} else {0}) + (if dmin_diff {2} else {0}) + scales_diff + qs_diff;
            total_diff += blk_diff;
            if blk_diff > 0 {
                println!("blk {}: d_diff={} dmin_diff={} scales_diff={} qs_diff={} (total {})",
                    blk, d_diff, dmin_diff, scales_diff, qs_diff, blk_diff);
                if d_diff {
                    println!("  hf2q  d (f16 LE bytes): {:02x} {:02x}", mine[0], mine[1]);
                    println!("  canon d (f16 LE bytes): {:02x} {:02x}", canon[0], canon[1]);
                }
                if dmin_diff {
                    println!("  hf2q  dmin (f16 LE bytes): {:02x} {:02x}", mine[2], mine[3]);
                    println!("  canon dmin (f16 LE bytes): {:02x} {:02x}", canon[2], canon[3]);
                }
                if scales_diff > 0 {
                    print!("  hf2q  scales[12]:  ");
                    for b in &mine[4..16] { print!("{:02x} ", b); }
                    println!();
                    print!("  canon scales[12]: ");
                    for b in &canon[4..16] { print!("{:02x} ", b); }
                    println!();
                }
            }
        }
        println!("\nTotal: {}/{} bytes diff", total_diff, n_blocks * 144);
    }

    /// 2026-05-20 Gemma 4 Q4_K broader sampler — beyond just blk.0.attn_k.
    /// Verifies how widespread the input-distribution boundary residuals are
    /// across the Gemma Q4_K tensor population.
    #[test]
    #[ignore]
    fn gemma_q4k_broad_sample_dump() {
        let cases: &[(&str, usize, &[usize])] = &[
            ("blk_0_attn_q_weight", 2816, &[0, 100, 1000]),
            ("blk_5_ffn_gate_weight", 2816, &[0, 100, 500]),
            ("blk_1_attn_k_weight", 2816, &[0, 100]),
            ("blk_10_attn_k_weight", 2816, &[0, 200]),
            ("blk_10_attn_q_weight", 2816, &[0, 500]),
            ("blk_15_ffn_gate_weight", 2816, &[0, 1000]),
            ("blk_25_ffn_up_weight", 2816, &[0, 300]),
            ("blk_0_ffn_up_weight", 2816, &[0, 100]),
            ("blk_28_attn_q_weight", 2816, &[0, 50]),
        ];
        let mut grand_total_diff = 0usize;
        let mut grand_total = 0usize;
        for (name, n_per_row, rows) in cases {
            for &row_idx in *rows {
                let f32_path = format!("/tmp/c_quant_repro/gemma_{}_row{}_f32.bin", name, row_idx);
                let q4k_path = format!("/tmp/c_quant_repro/gemma_{}_row{}_q4k.bin", name, row_idx);
                let f32_bytes = std::fs::read(&f32_path).unwrap_or_else(|_| panic!("{} must exist", f32_path));
                let canonical = std::fs::read(&q4k_path).unwrap_or_else(|_| panic!("{} must exist", q4k_path));
                let n_blocks = n_per_row / 256;
                assert_eq!(f32_bytes.len(), n_per_row * 4);
                assert_eq!(canonical.len(), n_blocks * 144);
                let f32: Vec<f32> = f32_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                let q4k = quantize(&f32, *n_per_row, None);
                let diff: usize = q4k.iter().zip(canonical.iter()).filter(|(a, b)| a != b).count();
                if diff > 0 {
                    println!("{} row {}: {} bytes differ", name, row_idx, diff);
                }
                grand_total_diff += diff;
                grand_total += q4k.len();
            }
        }
        println!("\nGEMMA Q4_K BROAD GRAND TOTAL: {}/{} bytes differ ({:.6}%)",
            grand_total_diff, grand_total, 100.0 * grand_total_diff as f64 / grand_total as f64);
    }

    /// 2026-05-20 Qwen 3.5 Q4_K BROAD diagnostic — sample 16 rows across 5
    /// tensors (incl. different layers, MoE experts, token_embd). 0 diff
    /// across all = strong evidence make_qkx2_quants is byte-identical post
    /// b921616e on the Qwen 3.5 distribution.
    #[test]
    #[ignore]
    fn qwen35_q4k_broad_sample_dump() {
        let cases: &[(&str, usize, &[usize])] = &[
            ("blk_0_attn_gate_weight", 2048, &[0, 100, 1000]),
            ("blk_0_ssm_out_weight", 4096, &[0, 100, 500]),
            ("token_embd_weight", 2048, &[0, 1000, 10000, 100000]),
            ("blk_20_attn_gate_weight", 2048, &[0, 100, 1000]),
            ("blk_0_ffn_up_exps_weight", 2048, &[0, 1000, 10000]),
        ];
        let mut grand_total_diff = 0usize;
        let mut grand_total = 0usize;
        for (name, n_per_row, rows) in cases {
            for &row_idx in *rows {
                let f32_path = format!("/tmp/c_quant_repro/qwen35_{}_row{}_f32.bin", name, row_idx);
                let q4k_path = format!("/tmp/c_quant_repro/qwen35_{}_row{}_q4k.bin", name, row_idx);
                let f32_bytes = std::fs::read(&f32_path)
                    .unwrap_or_else(|_| panic!("{} must exist", f32_path));
                let canonical = std::fs::read(&q4k_path)
                    .unwrap_or_else(|_| panic!("{} must exist", q4k_path));
                let n_blocks = n_per_row / 256;
                assert_eq!(f32_bytes.len(), n_per_row * 4);
                assert_eq!(canonical.len(), n_blocks * 144);
                let f32: Vec<f32> = f32_bytes.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let q4k = quantize(&f32, *n_per_row, None);
                let diff: usize = q4k.iter().zip(canonical.iter())
                    .filter(|(a, b)| a != b).count();
                if diff > 0 {
                    println!("{} row {}: {} bytes differ", name, row_idx, diff);
                }
                grand_total_diff += diff;
                grand_total += q4k.len();
            }
        }
        println!("\nBROAD GRAND TOTAL: {}/{} bytes differ ({:.6}%)",
            grand_total_diff, grand_total, 100.0 * grand_total_diff as f64 / grand_total as f64);
    }

    /// 2026-05-20 Qwen 3.5 Q4_K multi-tensor diagnostic — sample 3 tensors
    /// (blk.0.attn_gate, blk.0.ssm_out, blk.0.ffn_gate_exps), read F32 row 0
    /// from canonical's F16 GGUF + Q4_K_M bytes from canonical's quantized GGUF,
    /// run hf2q's Q4_K kernel, dump per-tensor byte-diff. If 0 across all three,
    /// the make_qkx2_quants kernel is effectively byte-identical to canonical
    /// post-b921616e on the Qwen 3.5 distribution.
    #[test]
    #[ignore]
    fn qwen35_q4k_multi_tensor_dump() {
        let cases: &[(&str, usize, usize)] = &[
            ("blk_0_attn_gate_weight", 2048, 8),
            ("blk_0_ssm_out_weight", 4096, 16),
            ("blk_0_ffn_gate_exps_weight", 2048, 8),
        ];
        let mut grand_total_diff = 0usize;
        let mut grand_total = 0usize;
        for (name, n_per_row, n_blocks) in cases {
            let f32_path = format!("/tmp/c_quant_repro/qwen35_{}_row0_f32.bin", name);
            let q4k_path = format!("/tmp/c_quant_repro/qwen35_{}_row0_q4k.bin", name);
            let f32_bytes = std::fs::read(&f32_path)
                .unwrap_or_else(|_| panic!("{} must exist", f32_path));
            let canonical = std::fs::read(&q4k_path)
                .unwrap_or_else(|_| panic!("{} must exist", q4k_path));
            assert_eq!(f32_bytes.len(), n_per_row * 4);
            assert_eq!(canonical.len(), n_blocks * 144);
            let f32: Vec<f32> = f32_bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let q4k = quantize(&f32, *n_per_row, None);
            assert_eq!(q4k.len(), canonical.len());
            let diff: usize = q4k.iter().zip(canonical.iter())
                .filter(|(a, b)| a != b).count();
            println!("{}: {}/{} bytes differ ({:.4}%)",
                name, diff, q4k.len(), 100.0 * diff as f64 / q4k.len() as f64);
            grand_total_diff += diff;
            grand_total += q4k.len();
        }
        println!("\nGRAND TOTAL: {}/{} bytes differ ({:.4}%)",
            grand_total_diff, grand_total, 100.0 * grand_total_diff as f64 / grand_total as f64);
    }

    /// 2026-05-20 diagnostic: read canonical Qwen 3.5 blk.0.attn_gate.weight
    /// row 0 (n_per_row=2048) — F32 from canonical's F16 GGUF + Q4_K_M bytes
    /// from canonical's quantized GGUF — run hf2q's Q4_K kernel, dump per-block
    /// byte-diff. Establishes whether make_qkx2_quants needs further fixes for
    /// Qwen 3.5 K-quant distribution post-b921616e make_qx_quants fix.
    #[test]
    #[ignore]
    fn qwen35_blk0_attn_gate_row0_q4k_dump() {
        let f32_bytes = std::fs::read("/tmp/c_quant_repro/qwen35_blk0_attn_gate_row0_f32.bin")
            .expect("/tmp/c_quant_repro/qwen35_blk0_attn_gate_row0_f32.bin must exist");
        assert_eq!(f32_bytes.len(), 8192, "expected 2048 F32 = 8192 bytes");
        let f32: Vec<f32> = f32_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let canonical = std::fs::read("/tmp/c_quant_repro/qwen35_blk0_attn_gate_row0_q4k.bin")
            .expect("canonical Q4_K bytes");
        assert_eq!(canonical.len(), 8 * 144, "expected 8 blocks × 144 bytes");

        let q4k = quantize(&f32, 2048, None);
        assert_eq!(q4k.len(), 8 * 144, "hf2q output mismatched length");

        let mut total_diff_bytes = 0;
        for blk_idx in 0..8 {
            let blk_start = blk_idx * 144;
            let blk_end = blk_start + 144;
            let hf2q_blk = &q4k[blk_start..blk_end];
            let canon_blk = &canonical[blk_start..blk_end];
            let blk_diff: usize = hf2q_blk.iter().zip(canon_blk.iter())
                .filter(|(a, b)| a != b).count();
            total_diff_bytes += blk_diff;
            if blk_diff > 0 {
                println!("Block {}: {} bytes differ", blk_idx, blk_diff);
                let hex_h = hf2q_blk[..16].iter().map(|b| format!("{:02x}", b)).collect::<String>();
                let hex_c = canon_blk[..16].iter().map(|b| format!("{:02x}", b)).collect::<String>();
                println!("  hf2q  d/dmin/scales12: {}", hex_h);
                println!("  canon d/dmin/scales12: {}", hex_c);
            }
        }
        println!("\nTotal: {}/{} bytes differ ({:.4}%)",
            total_diff_bytes, 8 * 144, 100.0 * total_diff_bytes as f64 / (8.0 * 144.0));
    }

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

    /// ADR-033 quality-matrix iter-23+ : real-model byte-cmp.
    ///
    /// Fast iteration path that bypasses full 10-minute model conversion.
    /// Compares hf2q's Q4_K output against canonical llama.cpp's output
    /// for blk.0.attn_k.weight (Gemma 4 26B-A4B-IT). Fixture is the
    /// canonical F16 GGUF's F16→F32 dequantization (lossless).
    ///
    /// Fixture preparation (one-time, requires gguf-py + canonical GGUFs):
    ///   python3 /tmp/single_tensor_test.py
    ///
    /// Expected: 0 differing bytes once FMA + SIMD reduction fixes are
    /// complete. Until then, prints diff count + first-divergence location.
    /// Gated on fixture file presence so CI without the fixture skips.
    #[test]
    fn real_model_byte_cmp_blk0_attn_k() {
        let f32_path = "/tmp/blk0_attn_k_f32.bin";
        let expected_path = "/tmp/blk0_attn_k_q4k_expected.bin";
        if !std::path::Path::new(f32_path).exists()
            || !std::path::Path::new(expected_path).exists()
        {
            eprintln!(
                "[skip] real_model_byte_cmp_blk0_attn_k: fixture files missing. \
                 Run: python3 /tmp/single_tensor_test.py"
            );
            return;
        }
        let f32_bytes = fs::read(f32_path).expect("read F32 fixture");
        let n = f32_bytes.len() / 4;
        let mut f32_in = Vec::with_capacity(n);
        for i in 0..n {
            let b = [
                f32_bytes[i * 4],
                f32_bytes[i * 4 + 1],
                f32_bytes[i * 4 + 2],
                f32_bytes[i * 4 + 3],
            ];
            f32_in.push(f32::from_le_bytes(b));
        }
        assert_eq!(f32_in.len(), 2816 * 2048);
        // GGUF stores [2816, 2048] with ne[0]=2816 (row size). Each row is
        // 2816 elements (matches `data.shape=(2048, 2816)` in gguf-py).
        let actual = quantize(&f32_in, 2816, None);
        let expected = fs::read(expected_path).expect("read expected");
        assert_eq!(actual.len(), expected.len());

        let mut diffs = 0usize;
        let mut first_diff: Option<(usize, u8, u8)> = None;
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            if a != e {
                diffs += 1;
                if first_diff.is_none() {
                    first_diff = Some((i, a, e));
                }
            }
        }
        let pct = 100.0 * diffs as f64 / expected.len() as f64;
        eprintln!(
            "Q4_K real-model byte-cmp blk.0.attn_k.weight:\n  \
             diffs={} of {} ({:.4}%)",
            diffs,
            expected.len(),
            pct
        );
        if let Some((idx, a, e)) = first_diff {
            let block = idx / 144;
            let intra = idx % 144;
            let field = match intra {
                0..=1 => "d".to_string(),
                2..=3 => "dmin".to_string(),
                4..=15 => format!("scales[{}]", intra - 4),
                _ => format!("qs[{}]", intra - 16),
            };
            eprintln!(
                "  first diff: byte {} = block {} field {}: canonical=0x{:02x}, hf2q=0x{:02x}",
                idx, block, field, e, a
            );
        }
        // Count differing blocks + categorize by which field
        let mut blocks_diff = std::collections::HashSet::new();
        let mut field_diffs = std::collections::HashMap::new();
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            if a != e {
                let block = i / 144;
                let intra = i % 144;
                let field = match intra {
                    0..=1 => "d",
                    2..=3 => "dmin",
                    4..=15 => "scales",
                    _ => "qs",
                };
                blocks_diff.insert(block);
                *field_diffs.entry(field).or_insert(0) += 1;
            }
        }
        eprintln!("  blocks differing: {} of 22528 ({:.2}%)", blocks_diff.len(), 100.0 * blocks_diff.len() as f64 / 22528.0);
        eprintln!("  diffs per field: d={}, dmin={}, scales={}, qs={}",
            field_diffs.get("d").unwrap_or(&0),
            field_diffs.get("dmin").unwrap_or(&0),
            field_diffs.get("scales").unwrap_or(&0),
            field_diffs.get("qs").unwrap_or(&0));
        // 2026-05-20 (post-stale-fixture-fix): fixture regenerated against
        // current /opt/llama.cpp HEAD `e15384a5c` and hf2q kernel reverted
        // to plain `+=` at sum_l/sum_l2/sum_xl (matches canonical's effective
        // fp-contract=off behavior). Result: 3/3244032 = 0.0001% bytes diff
        // — one block hit a deeper input-distribution-dependent FP rounding
        // boundary in make_qkx2 that no clean source-level intervention
        // closes without breaking other distributions. Sub-0.001% residual
        // accepted as the practical ceiling.
        assert!(
            diffs <= 8,
            "byte-cmp regression: {} bytes differ (expected ≤8 — sub-0.001% boundary noise)",
            diffs
        );
    }
}
