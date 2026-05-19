//! `Quantizer` trait + `GgmlQuantizer` dispatch struct + factory.
//!
//! Per ADR-033 Decision §"Quantizer trait (Decision §2 concrete)":
//! unifies the 11 v1 kernel modules behind a single trait surface for
//! the policy-driven pipeline. Each call validates shape constraints
//! and dispatches to the matching `<type>::quantize` kernel.
//!
//! Wraps:
//! `q4_0` `q4_1` `q5_0` `q5_1` `q8_0` `iq4_nl` `q2_k` `q3_k` `q4_k`
//! `q5_k` `q6_k` (11 of llama.cpp's 14 numeric GgmlType values — F32 /
//! F16 / BF16 / Q8_1 / Q8_K are pass-through or internal-only and not
//! Quantizer-bound here).

use super::error::QuantizeError;
use super::ggml_type::GgmlType;

/// Unified quantizer surface over all v1 ggml_quants kernel modules.
/// Per ADR Decision §"Quantizer trait": every byte the pipeline emits
/// flows through `quantize`; the type-system enforces no-fallback at
/// the return type (`Result<Vec<u8>, QuantizeError>`).
pub trait Quantizer: Send + Sync {
    fn ggml_type(&self) -> GgmlType;
    fn quantize(
        &self,
        src: &[f32],
        n_per_row: usize,
        imatrix: Option<&[f32]>,
    ) -> Result<Vec<u8>, QuantizeError>;
}

/// Concrete dispatch struct: holds the target type, dispatches at
/// runtime via a single `match` on `self.ty`. Cheaper than a per-type
/// `Box<dyn Quantizer>` for stable, type-known call sites; the factory
/// returns this struct directly.
#[derive(Debug, Clone, Copy)]
pub struct GgmlQuantizer {
    pub ty: GgmlType,
}

impl GgmlQuantizer {
    pub const fn new(ty: GgmlType) -> Self {
        Self { ty }
    }
}

impl Quantizer for GgmlQuantizer {
    fn ggml_type(&self) -> GgmlType {
        self.ty
    }

    fn quantize(
        &self,
        src: &[f32],
        n_per_row: usize,
        imatrix: Option<&[f32]>,
    ) -> Result<Vec<u8>, QuantizeError> {
        let bs = self.ty.block_size();
        if n_per_row % bs != 0 {
            return Err(QuantizeError::NotBlockAligned {
                ggml_type: self.ty,
                n_per_row,
                block_size: bs,
            });
        }
        if src.len() % n_per_row != 0 {
            return Err(QuantizeError::NotRowAligned {
                src_len: src.len(),
                n_per_row,
            });
        }
        if let Some(im) = imatrix {
            if im.len() != n_per_row {
                return Err(QuantizeError::ImatrixLenMismatch {
                    n_per_row,
                    im_len: im.len(),
                });
            }
        }

        Ok(match self.ty {
            GgmlType::Q4_0 => super::q4_0::quantize(src, n_per_row, imatrix),
            GgmlType::Q4_1 => super::q4_1::quantize(src, n_per_row, imatrix),
            GgmlType::Q5_0 => super::q5_0::quantize(src, n_per_row, imatrix),
            GgmlType::Q5_1 => super::q5_1::quantize(src, n_per_row, imatrix),
            GgmlType::Q8_0 => super::q8_0::quantize(src, n_per_row, imatrix),
            GgmlType::IQ4_NL => super::iq4_nl::quantize(src, n_per_row, imatrix),
            GgmlType::Q2_K => super::q2_k::quantize(src, n_per_row, imatrix),
            GgmlType::Q3_K => super::q3_k::quantize(src, n_per_row, imatrix),
            GgmlType::Q4_K => super::q4_k::quantize(src, n_per_row, imatrix),
            GgmlType::Q5_K => super::q5_k::quantize(src, n_per_row, imatrix),
            GgmlType::Q6_K => super::q6_k::quantize(src, n_per_row, imatrix),
            // F32/F16/BF16/Q8_1/Q8_K — no Quantizer here. Pass-through
            // types are handled at the dispatcher layer (vision/audio
            // F16, explicit --quant f16/bf16, etc.) per ADR Decision
            // §"Vision / audio tensor patterns".
            other => return Err(QuantizeError::NoQuantizerForType(other)),
        })
    }
}

/// Factory: return a `GgmlQuantizer` for `ty`, or `NoQuantizerForType`
/// if the type has no kernel in this build.
pub fn quantizer_for(ty: GgmlType) -> Result<GgmlQuantizer, QuantizeError> {
    match ty {
        GgmlType::Q4_0
        | GgmlType::Q4_1
        | GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::Q8_0
        | GgmlType::IQ4_NL
        | GgmlType::Q2_K
        | GgmlType::Q3_K
        | GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q6_K => Ok(GgmlQuantizer::new(ty)),
        other => Err(QuantizeError::NoQuantizerForType(other)),
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
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn read_bytes(name: &str) -> Vec<u8> {
        fs::read(fixture_path(name)).expect("read fixture")
    }

    #[test]
    fn factory_returns_quantizer_for_v1_types() {
        for ty in [
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q8_0,
            GgmlType::IQ4_NL,
            GgmlType::Q2_K,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
        ] {
            let q = quantizer_for(ty).expect("v1 type has Quantizer impl");
            assert_eq!(q.ggml_type(), ty);
        }
    }

    #[test]
    fn factory_rejects_out_of_v1_types() {
        // F32/F16/BF16 are pass-through, not quantizers.
        // Q8_1/Q8_K are llama.cpp internals, not on-disk.
        for ty in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::BF16,
            GgmlType::Q8_1,
            GgmlType::Q8_K,
        ] {
            assert!(
                matches!(
                    quantizer_for(ty),
                    Err(QuantizeError::NoQuantizerForType(_))
                ),
                "unexpected: {:?} returned Ok",
                ty
            );
        }
    }

    /// Verify trait dispatch is byte-equivalent to direct kernel call
    /// for all 11 v1 types. Reuses the same fixtures as the kernel-
    /// level tests; this is the ADR §P1 acceptance criterion
    /// translated to the trait surface.
    #[test]
    fn dispatch_matches_kernel_all_types() {
        let cases: &[(GgmlType, &str, &str, usize)] = &[
            (GgmlType::Q4_0, "q4_0_64_noim_input.bin", "q4_0_64_noim_expected.bin", 64),
            (GgmlType::Q4_1, "q4_1_64_noim_input.bin", "q4_1_64_noim_expected.bin", 64),
            (GgmlType::Q5_0, "q5_0_64_noim_input.bin", "q5_0_64_noim_expected.bin", 64),
            (GgmlType::Q5_1, "q5_1_64_noim_input.bin", "q5_1_64_noim_expected.bin", 64),
            (GgmlType::Q8_0, "q8_0_64_noim_input.bin", "q8_0_64_noim_expected.bin", 64),
            (GgmlType::IQ4_NL, "iq4_nl_64_noim_input.bin", "iq4_nl_64_noim_expected.bin", 64),
            (GgmlType::Q2_K, "q2_k_512_noim_input.bin", "q2_k_512_noim_expected.bin", 512),
            (GgmlType::Q3_K, "q3_k_512_noim_input.bin", "q3_k_512_noim_expected.bin", 512),
            (GgmlType::Q4_K, "q4_k_512_noim_input.bin", "q4_k_512_noim_expected.bin", 512),
            (GgmlType::Q5_K, "q5_k_512_noim_input.bin", "q5_k_512_noim_expected.bin", 512),
            (GgmlType::Q6_K, "q6_k_512_noim_input.bin", "q6_k_512_noim_expected.bin", 512),
        ];
        for (ty, in_name, exp_name, n_per_row) in cases {
            let input = read_f32s(in_name);
            let expected = read_bytes(exp_name);
            let q = quantizer_for(*ty).unwrap();
            let got = q.quantize(&input, *n_per_row, None).unwrap();
            assert_eq!(got, expected, "trait dispatch differs from fixture for {:?}", ty);
        }
    }

    #[test]
    fn validation_rejects_bad_shapes() {
        let q = quantizer_for(GgmlType::Q4_0).unwrap();
        // n_per_row=33 isn't a multiple of QK4_0=32
        assert!(matches!(
            q.quantize(&[0.0; 33], 33, None),
            Err(QuantizeError::NotBlockAligned { .. })
        ));
        // src.len()=63 isn't a multiple of n_per_row=32
        assert!(matches!(
            q.quantize(&[0.0; 63], 32, None),
            Err(QuantizeError::NotRowAligned { .. })
        ));
        // imatrix len mismatch
        assert!(matches!(
            q.quantize(&[0.0; 64], 32, Some(&[0.0; 16])),
            Err(QuantizeError::ImatrixLenMismatch { .. })
        ));
    }

    #[test]
    fn ggml_type_round_trip_via_factory() {
        // u32 → GgmlType → GgmlQuantizer → ggml_type() should round-trip.
        for v in [2u32, 3, 6, 7, 8, 10, 11, 12, 13, 14, 20] {
            let ty = GgmlType::try_from(v).unwrap();
            let q = quantizer_for(ty).unwrap();
            assert_eq!(u32::from(q.ggml_type()), v);
        }
    }
}
