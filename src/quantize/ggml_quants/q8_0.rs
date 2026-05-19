//! Q8_0 quantizer — ADR-033 P0 pure-Rust port of
//! `quantize_row_q8_0_ref` at `/opt/llama.cpp/ggml/src/ggml-quants.c:234`
//! (SHA pinned in `data/llama_cpp_pin.txt`).
//!
//! Block layout from `ggml-common.h:241-246`:
//! ```text
//! #define QK8_0 32
//! typedef struct {
//!     ggml_half d;       // 2 bytes, F16, little-endian
//!     int8_t  qs[QK8_0]; // 32 bytes
//! } block_q8_0;
//! static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0);
//! ```
//!
//! The dispatcher at `ggml-quants.c:2219` (`quantize_q8_0`) explicitly
//! discards the `quant_weights` (imatrix) argument — Q8_0 has no
//! imatrix-adaptive variant. The `noim` and `im` fixtures are
//! byte-identical (verified at fixture-generation time).

use half::f16;

pub const QK8_0: usize = 32;
pub const BLOCK_BYTES: usize = 2 + QK8_0; // 34

/// Quantize an F32 buffer to Q8_0 bytes.
///
/// `src.len()` must be a multiple of `n_per_row`, and `n_per_row` must
/// be a multiple of `QK8_0`. The `imatrix` argument is accepted for
/// API symmetry with other types but is ignored per
/// `ggml-quants.c:2220` (`(void)quant_weights`).
pub fn quantize(src: &[f32], n_per_row: usize, _imatrix: Option<&[f32]>) -> Vec<u8> {
    assert!(
        n_per_row % QK8_0 == 0,
        "n_per_row {} not multiple of QK8_0 {}",
        n_per_row,
        QK8_0
    );
    assert!(
        src.len() % n_per_row == 0,
        "src len {} not multiple of n_per_row {}",
        src.len(),
        n_per_row
    );

    let total_blocks = src.len() / QK8_0;
    let mut out = Vec::with_capacity(total_blocks * BLOCK_BYTES);

    for block_idx in 0..total_blocks {
        let block = &src[block_idx * QK8_0..(block_idx + 1) * QK8_0];

        let mut amax = 0.0f32;
        for &v in block {
            let a = v.abs();
            if a > amax {
                amax = a;
            }
        }

        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let d_f16 = f16::from_f32(d);
        out.extend_from_slice(&d_f16.to_le_bytes());

        for &v in block {
            let q = (v * id).round() as i8;
            out.push(q as u8);
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

    #[test]
    fn byte_cmp_noim() {
        let input = read_f32s("q8_0_64_noim_input.bin");
        let expected = read_bytes("q8_0_64_noim_expected.bin");
        let got = quantize(&input, 64, None);
        assert_eq!(got.len(), expected.len(), "Q8_0 noim length mismatch");
        assert_eq!(got, expected, "Q8_0 noim byte-cmp failed");
    }

    #[test]
    fn byte_cmp_im() {
        let input = read_f32s("q8_0_64_im_input.bin");
        let expected = read_bytes("q8_0_64_im_expected.bin");
        let imatrix = vec![1.0f32; 64];
        let got = quantize(&input, 64, Some(&imatrix));
        assert_eq!(
            got, expected,
            "Q8_0 im byte-cmp failed (Q8_0 must ignore imatrix per ggml-quants.c:2220)"
        );
    }
}
