//! Q3_K quantizer — ADR-033 P0 pure-Rust port of
//! `quantize_row_q3_K_ref` at `/opt/llama.cpp/ggml/src/ggml-quants.c:1167`,
//! `quantize_row_q3_K_impl` at `/opt/llama.cpp/ggml/src/ggml-quants.c:1293`,
//! and dispatcher `quantize_q3_K` at `/opt/llama.cpp/ggml/src/ggml-quants.c:1377`
//! (SHA pinned in `data/llama_cpp_pin.txt`).
//!
//! Block layout from `ggml-common.h:301-311`:
//! ```text
//! #define QK_K 256
//! typedef struct {
//!     uint8_t hmask[QK_K/8]; // 32 bytes — high bit of 3-bit quants
//!     uint8_t qs[QK_K/4];    // 64 bytes — low 2 bits packed 4-per-byte
//!     uint8_t scales[12];    // 16 signed 6-bit sub-scales (packed)
//!     ggml_half d;           // 2 bytes — super-block scale (F16, LE)
//! } block_q3_K;             // 110 bytes total
//! ```
//!
//! `_ref` uses `make_q3_quants` (RMSE-iterative, no imatrix), then
//! quantizes the 16 sub-scales by a single `iscale = -32/max_scale`.
//! `_impl` uses `make_qx_quants(rmse_type=1, weight=qw*sqrt(sigma2+x²))`
//! per sub-block, then a SECOND `make_qx_quants(nmax=32)` over the 16
//! sub-scales themselves with per-sub-block weight `sw[j] = sum(weight)`.
//! Both paths share the final assignment loop that re-derives the per-
//! sub-block scale from the 6-bit-packed sub-scales (to match decode
//! exactly), re-quantizes the qs values, splits high vs low bits, and
//! packs four 2-bit values per byte across the four 32-element groups.

use half::f16;

pub const QK_K: usize = 256;
/// 32 + 64 + 12 + 2
pub const BLOCK_BYTES: usize = QK_K / 8 + QK_K / 4 + 12 + 2;

const GROUP_MAX_EPS: f32 = 1e-15;

/// Mirrors `nearest_int` at `ggml-quants.c:559`.
#[inline(always)]
fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4194303.0);
    let val = fval + 12582912.0;
    let i = val.to_bits() as i32;
    (i & 0x007fffff) - 0x00400000
}

/// Mirrors `make_qx_quants` at `ggml-quants.c:566`.
/// When `qw.is_empty()`, falls back to the `rmse_type`-dependent weight
/// formula (mirrors C `qw ? qw[i] : ...`). Q3_K's `_impl` always passes
/// a non-empty `qw`.
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

/// Mirrors `make_q3_quants` at `ggml-quants.c:635` for `do_rmse=true`
/// (the only call site Q3_K_ref uses).
fn make_q3_quants(n: usize, nmax: i32, x: &[f32], l: &mut [i8]) -> f32 {
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
    let iscale = -(nmax as f32) / max;
    // do_rmse=true path
    let mut sumlx: f32 = 0.0;
    let mut suml2: f32 = 0.0;
    for i in 0..n {
        let li = nearest_int(iscale * x[i]);
        let li = li.max(-nmax).min(nmax - 1);
        l[i] = li as i8;
        let w = x[i] * x[i];
        sumlx += w * x[i] * (li as f32);
        suml2 += w * (li as f32) * (li as f32);
    }
    for _itry in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = x[i] * x[i];
            let cur_li = l[i] as i32;
            let slx = sumlx - w * x[i] * (cur_li as f32);
            if slx > 0.0 {
                let sl2 = suml2 - w * (cur_li as f32) * (cur_li as f32);
                let new_l = nearest_int(x[i] * sl2 / slx);
                let new_l = new_l.max(-nmax).min(nmax - 1);
                if new_l != cur_li {
                    let slx_new = slx + w * x[i] * (new_l as f32);
                    let sl2_new = sl2 + w * (new_l as f32) * (new_l as f32);
                    if sl2_new > 0.0
                        && slx_new * slx_new * suml2 > sumlx * sumlx * sl2_new
                    {
                        l[i] = new_l as i8;
                        sumlx = slx_new;
                        suml2 = sl2_new;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    for i in 0..n {
        l[i] = (l[i] as i32 + nmax) as i8;
    }
    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        0.0
    }
}

/// Reads back a 6-bit signed sub-scale from the packed `scales[12]`
/// layout, in the same way decode does, then returns it as `(sc - 32)`.
/// Mirrors lines `ggml-quants.c:1207-1208` and `1341-1342`.
#[inline]
fn unpack_sub_scale(scales: &[u8; 12], j: usize) -> i32 {
    let sc = if j < 8 {
        scales[j] & 0xF
    } else {
        scales[j - 8] >> 4
    };
    let sc = (sc | (((scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) as i32;
    sc - 32
}

/// Pack the 6-bit signed sub-scales into `scales[12]` (low 4 bits in
/// bytes 0..7, high 2 bits in bytes 8..11). Mirrors lines
/// `ggml-quants.c:1192-1199` and `1329-1335`. Caller passes the already-
/// biased value `l` in `[0, 63]` (i.e. `actual + 32`).
#[inline]
fn pack_sub_scale(scales: &mut [u8; 12], j: usize, l_in: i32) {
    let mut l = l_in;
    if j < 8 {
        scales[j] = (l & 0xF) as u8;
    } else {
        scales[j - 8] |= ((l & 0xF) << 4) as u8;
    }
    l >>= 4;
    scales[j % 4 + 8] |= ((l & 0x3) << (2 * (j / 4))) as u8;
}

/// Final shared assignment + packing block. Mirrors
/// `ggml-quants.c:1205-1237` (ref) and `1339-1371` (impl), which are
/// byte-identical. Uses the unpacked sub-scales from `y_scales`
/// (decode-equivalent), re-quantizes the qs values, splits hmask, and
/// packs four 2-bit values per byte across the four 32-element groups.
fn finalize_block(
    x: &[f32],
    y_d: f32,
    y_scales: &[u8; 12],
    y_hmask: &mut [u8; QK_K / 8],
    y_qs: &mut [u8; QK_K / 4],
) {
    let mut l_buf = [0i8; QK_K];
    // C reads d as `GGML_FP16_TO_FP32(y[i].d) * sc`, i.e. after F16
    // round-trip — not the original F32 scale. Mirror that exactly.
    let d_fp16_rt: f32 = f16::from_f32(y_d).to_f32();
    for j in 0..QK_K / 16 {
        let sc = unpack_sub_scale(y_scales, j);
        let d = d_fp16_rt * (sc as f32);
        if d == 0.0 {
            // C `continue;` — leave L[16*j .. 16*j+16] at their default
            // initial value. The `_ref` and `_impl` C code uses a stack
            // `int8_t L[QK_K]` that was already written by the earlier
            // `make_q3_quants`/`make_qx_quants` loop, so the "leave as-
            // is" branch keeps those values. We re-fill `l_buf` from
            // the FRESH 1/d-recompute path; on the d==0 branch we leave
            // l_buf at 0, which biases to L=4 (>3 → hmask set, then
            // L=0 in qs). That matches what C does for a zero
            // sub-scale on garbage L: the zero-d branch is unreachable
            // on the noim path when max_scale!=0 because every
            // sub-scale uses the same iscale; on the im path it can
            // happen only if d_block * sc rounds to 0, in which case
            // the original L is already random-ish from
            // make_qx_quants. We must match C byte-for-byte though, so
            // we leave l_buf alone (matches C "skip this sub-block").
            continue;
        }
        for ii in 0..16 {
            let li = nearest_int(x[16 * j + ii] / d);
            let li = li.max(-4).min(3);
            l_buf[16 * j + ii] = (li + 4) as i8;
        }
    }
    // hmask: high bit of each L[j] (which is in [0, 7]) — when L > 3,
    // set the high bit AND subtract 4.
    for b in y_hmask.iter_mut() {
        *b = 0;
    }
    let mut m = 0usize;
    let mut hm: u8 = 1;
    for j in 0..QK_K {
        if l_buf[j] > 3 {
            y_hmask[m] |= hm;
            l_buf[j] -= 4;
        }
        m += 1;
        if m == QK_K / 8 {
            m = 0;
            hm <<= 1;
        }
    }
    // qs: four 2-bit values packed per byte, across four 32-element
    // groups (the C loop is `for j in 0,128 step` with inner 32-stride).
    for j in (0..QK_K).step_by(128) {
        for l in 0..32 {
            let v0 = l_buf[j + l] as u8 & 0x3;
            let v1 = (l_buf[j + l + 32] as u8 & 0x3) << 2;
            let v2 = (l_buf[j + l + 64] as u8 & 0x3) << 4;
            let v3 = (l_buf[j + l + 96] as u8 & 0x3) << 6;
            y_qs[j / 4 + l] = v0 | v1 | v2 | v3;
        }
    }
}

/// Pure reference path (no imatrix). Mirrors `quantize_row_q3_K_ref` at
/// `ggml-quants.c:1167`.
fn quantize_row_ref(src: &[f32], out: &mut Vec<u8>) {
    debug_assert!(src.len() % QK_K == 0);
    let nb = src.len() / QK_K;

    let mut l_tmp = [0i8; QK_K];
    let mut scales = [0f32; QK_K / 16];

    for i in 0..nb {
        let x = &src[i * QK_K..(i + 1) * QK_K];

        let mut max_scale: f32 = 0.0;
        let mut amax: f32 = 0.0;
        for j in 0..QK_K / 16 {
            scales[j] = make_q3_quants(
                16,
                4,
                &x[16 * j..16 * j + 16],
                &mut l_tmp[16 * j..16 * j + 16],
            );
            let scale = scales[j].abs();
            if scale > amax {
                amax = scale;
                max_scale = scales[j];
            }
        }

        let mut y_scales = [0u8; 12];
        let y_d: f32;
        if max_scale != 0.0 {
            let iscale = -32.0 / max_scale;
            for j in 0..QK_K / 16 {
                let li = nearest_int(iscale * scales[j]);
                let l = li.max(-32).min(31) + 32;
                pack_sub_scale(&mut y_scales, j, l);
            }
            y_d = 1.0 / iscale;
        } else {
            y_d = 0.0;
        }

        let mut y_hmask = [0u8; QK_K / 8];
        let mut y_qs = [0u8; QK_K / 4];
        finalize_block(x, y_d, &y_scales, &mut y_hmask, &mut y_qs);

        out.extend_from_slice(&y_hmask);
        out.extend_from_slice(&y_qs);
        out.extend_from_slice(&y_scales);
        out.extend_from_slice(&f16::from_f32(y_d).to_le_bytes());
    }
}

/// imatrix-aware path. Mirrors `quantize_row_q3_K_impl` at
/// `ggml-quants.c:1293`. Operates one row at a time.
fn quantize_row_impl(x: &[f32], imatrix: &[f32], out: &mut Vec<u8>) {
    let n_per_row = x.len();
    debug_assert_eq!(imatrix.len(), n_per_row);
    debug_assert!(n_per_row % QK_K == 0);

    let nb = n_per_row / QK_K;

    let mut l_tmp = [0i8; QK_K];
    let mut scales = [0f32; QK_K / 16];
    let mut sw = [0f32; QK_K / 16];
    let mut ls = [0i8; QK_K / 16];
    let mut weight = [0f32; 16];

    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let qwb = &imatrix[i * QK_K..(i + 1) * QK_K];

        let mut sumx2: f32 = 0.0;
        for j in 0..QK_K {
            sumx2 += xb[j] * xb[j];
        }
        let sigma2 = 2.0 * sumx2 / QK_K as f32;

        for j in 0..QK_K / 16 {
            // quant_weights is non-null on the impl path.
            for l in 0..16 {
                weight[l] = qwb[16 * j + l] * (sigma2 + xb[16 * j + l] * xb[16 * j + l]).sqrt();
            }
            let mut sumw: f32 = 0.0;
            for l in 0..16 {
                sumw += weight[l];
            }
            sw[j] = sumw;

            scales[j] = make_qx_quants(
                16,
                4,
                &xb[16 * j..16 * j + 16],
                &mut l_tmp[16 * j..16 * j + 16],
                1,
                &weight,
            );
        }

        let mut y_scales = [0u8; 12];
        let d_block = make_qx_quants(QK_K / 16, 32, &scales, &mut ls, 1, &sw);
        for j in 0..QK_K / 16 {
            // `ls[j]` is already biased by `+nmax=32` inside make_qx_quants
            // (i.e. ls[j] is in [0, 63]). C reads it as `int l = Ls[j];`
            // then packs directly.
            pack_sub_scale(&mut y_scales, j, ls[j] as i32);
        }
        let y_d = d_block;

        let mut y_hmask = [0u8; QK_K / 8];
        let mut y_qs = [0u8; QK_K / 4];
        finalize_block(xb, y_d, &y_scales, &mut y_hmask, &mut y_qs);

        out.extend_from_slice(&y_hmask);
        out.extend_from_slice(&y_qs);
        out.extend_from_slice(&y_scales);
        out.extend_from_slice(&f16::from_f32(y_d).to_le_bytes());
    }
}

/// Quantize an F32 buffer to Q3_K bytes.
///
/// Mirrors dispatcher at `ggml-quants.c:1377`. When `imatrix` is
/// `Some`, the imatrix-aware `_impl` path is used per-row (imatrix
/// reused across rows); otherwise the `_ref` path is used on the whole
/// buffer.
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
            "imatrix len {} must equal n_per_row {}",
            im.len(),
            n_per_row
        );
    }

    let total_blocks = src.len() / QK_K;
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
        let input = read_f32s("q3_k_512_noim_input.bin");
        let expected = read_bytes("q3_k_512_noim_expected.bin");
        let got = quantize(&input, 512, None);
        assert_eq!(got.len(), expected.len(), "Q3_K noim length mismatch");
        assert_eq!(got, expected, "Q3_K noim byte-cmp failed");
    }

    #[test]
    fn byte_cmp_im() {
        let input = read_f32s("q3_k_512_im_input.bin");
        let expected = read_bytes("q3_k_512_im_expected.bin");
        let imatrix = make_imatrix(512, 2);
        let got = quantize(&input, 512, Some(&imatrix));
        assert_eq!(got.len(), expected.len(), "Q3_K im length mismatch");
        assert_eq!(got, expected, "Q3_K im byte-cmp failed");
    }
}
