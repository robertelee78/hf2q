//! `GgufFtype` — Rust enum mirroring the peer's `enum ftype`
//! at the literal numeric values so
//! the GGUF header `general.file_type` byte matches across pipelines.
//!
//! Per ADR-033 Decision §"GgufFtype mapping (Decision §2 concrete)".
//!
//! v1 supported set per ADR §P0 amendment A (Q2_K + Q3_K added):
//!   AllF32 / MostlyF16 / BF16 / MostlyQ4_0 / MostlyQ4_1 / MostlyQ5_0
//!   / MostlyQ5_1 / MostlyQ2_K / MostlyQ3_K_S/M/L / MostlyQ4_K_S/M /
//!   MostlyQ5_K_S/M / MostlyQ6_K / MostlyQ8_0 / MostlyIQ4_NL.
//!
//! Holes (4, 5, 6, 19-24, 26-31) reserved for IQ2_*/IQ3_*/IQ1_*/TQ1_0/
//! TQ2_0 (out of v1 scope; reservation per Decision §6 reserved-name
//! typed-error stubs).

use super::error::QuantizeError;
use super::ggml_type::GgmlType;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)] // mirrors the peer's LLAMA_FTYPE_MOSTLY_Q4_0-style names
pub enum GgufFtype {
    AllF32 = 0,
    MostlyF16 = 1,
    MostlyQ4_0 = 2,
    MostlyQ4_1 = 3,
    // 4, 5, 6 unused
    MostlyQ8_0 = 7,
    MostlyQ5_0 = 8,
    MostlyQ5_1 = 9,
    MostlyQ2_K = 10,
    MostlyQ3_K_S = 11,
    MostlyQ3_K_M = 12,
    MostlyQ3_K_L = 13,
    MostlyQ4_K_S = 14,
    MostlyQ4_K_M = 15,
    MostlyQ5_K_S = 16,
    MostlyQ5_K_M = 17,
    MostlyQ6_K = 18,
    // 19..24 + 26..31: IQ-family placeholders. Convert pipeline has no
    // path to these in v1 (`primary_type` will panic if called on them),
    // but `target_for` needs the variants to express
    // `llama_tensor_get_type_impl`'s arch/ftype branches verbatim.
    // Per [[feedback-no-loop-suppression-2026-05-17]].
    MostlyIQ2_XXS = 19,
    MostlyIQ2_XS = 20,
    MostlyQ2_K_S = 21,
    MostlyIQ3_XS = 22,
    MostlyIQ3_XXS = 23,
    MostlyIQ1_S = 24,
    MostlyIQ4_NL = 25,
    MostlyIQ3_S = 26,
    MostlyIQ3_M = 27,
    MostlyIQ2_S = 28,
    MostlyIQ2_M = 29,
    MostlyIQ4_XS = 30,
    MostlyIQ1_M = 31,
    BF16 = 32,
    MostlyTQ1_0 = 36,
    MostlyTQ2_0 = 37,
    MostlyMXFP4_MOE = 38,
}

impl GgufFtype {
    /// The "primary" GgmlType for non-special tensors at this ftype.
    /// Mirrors the early default-pick logic in the peer's
    /// `llama_model_quantize_impl` before `llama_tensor_get_type_impl`
    /// runs the per-tensor override pass.
    pub const fn primary_type(self) -> GgmlType {
        match self {
            GgufFtype::AllF32 => GgmlType::F32,
            GgufFtype::MostlyF16 => GgmlType::F16,
            GgufFtype::BF16 => GgmlType::BF16,
            GgufFtype::MostlyQ4_0 => GgmlType::Q4_0,
            GgufFtype::MostlyQ4_1 => GgmlType::Q4_1,
            GgufFtype::MostlyQ5_0 => GgmlType::Q5_0,
            GgufFtype::MostlyQ5_1 => GgmlType::Q5_1,
            GgufFtype::MostlyQ8_0 => GgmlType::Q8_0,
            GgufFtype::MostlyIQ4_NL => GgmlType::IQ4_NL,
            GgufFtype::MostlyQ2_K | GgufFtype::MostlyQ2_K_S => GgmlType::Q2_K,
            GgufFtype::MostlyQ3_K_S | GgufFtype::MostlyQ3_K_M | GgufFtype::MostlyQ3_K_L => {
                GgmlType::Q3_K
            }
            GgufFtype::MostlyQ4_K_S | GgufFtype::MostlyQ4_K_M => GgmlType::Q4_K,
            GgufFtype::MostlyQ5_K_S | GgufFtype::MostlyQ5_K_M => GgmlType::Q5_K,
            GgufFtype::MostlyQ6_K => GgmlType::Q6_K,
            // IQ-family placeholders: primary == the matching ggml_type.
            GgufFtype::MostlyIQ2_XXS => GgmlType::IQ2_XXS,
            GgufFtype::MostlyIQ2_XS => GgmlType::IQ2_XS,
            GgufFtype::MostlyIQ3_XS => GgmlType::IQ3_S, // C: ftype Q3_K_XS → IQ3_S
            GgufFtype::MostlyIQ3_XXS => GgmlType::IQ3_XXS,
            GgufFtype::MostlyIQ1_S => GgmlType::IQ1_S,
            GgufFtype::MostlyIQ3_S | GgufFtype::MostlyIQ3_M => GgmlType::IQ3_S,
            GgufFtype::MostlyIQ2_S | GgufFtype::MostlyIQ2_M => GgmlType::IQ2_S,
            GgufFtype::MostlyIQ4_XS => GgmlType::IQ4_XS,
            GgufFtype::MostlyIQ1_M => GgmlType::IQ1_M,
            GgufFtype::MostlyTQ1_0 => GgmlType::TQ1_0,
            GgufFtype::MostlyTQ2_0 => GgmlType::TQ2_0,
            GgufFtype::MostlyMXFP4_MOE => GgmlType::MXFP4,
        }
    }

    /// CLI/log name matching the `--quant <name>` surface in Decision §6.
    pub const fn name(self) -> &'static str {
        match self {
            GgufFtype::AllF32 => "f32",
            GgufFtype::MostlyF16 => "f16",
            GgufFtype::BF16 => "bf16",
            GgufFtype::MostlyQ4_0 => "q4_0",
            GgufFtype::MostlyQ4_1 => "q4_1",
            GgufFtype::MostlyQ5_0 => "q5_0",
            GgufFtype::MostlyQ5_1 => "q5_1",
            GgufFtype::MostlyQ8_0 => "q8_0",
            GgufFtype::MostlyIQ4_NL => "iq4_nl",
            GgufFtype::MostlyQ2_K => "q2_k",
            GgufFtype::MostlyQ3_K_S => "q3_k_s",
            GgufFtype::MostlyQ3_K_M => "q3_k_m",
            GgufFtype::MostlyQ3_K_L => "q3_k_l",
            GgufFtype::MostlyQ4_K_S => "q4_k_s",
            GgufFtype::MostlyQ4_K_M => "q4_k_m",
            GgufFtype::MostlyQ5_K_S => "q5_k_s",
            GgufFtype::MostlyQ5_K_M => "q5_k_m",
            GgufFtype::MostlyQ6_K => "q6_k",
            GgufFtype::MostlyIQ2_XXS => "iq2_xxs",
            GgufFtype::MostlyIQ2_XS => "iq2_xs",
            GgufFtype::MostlyQ2_K_S => "q2_k_s",
            GgufFtype::MostlyIQ3_XS => "iq3_xs",
            GgufFtype::MostlyIQ3_XXS => "iq3_xxs",
            GgufFtype::MostlyIQ1_S => "iq1_s",
            GgufFtype::MostlyIQ3_S => "iq3_s",
            GgufFtype::MostlyIQ3_M => "iq3_m",
            GgufFtype::MostlyIQ2_S => "iq2_s",
            GgufFtype::MostlyIQ2_M => "iq2_m",
            GgufFtype::MostlyIQ4_XS => "iq4_xs",
            GgufFtype::MostlyIQ1_M => "iq1_m",
            GgufFtype::MostlyTQ1_0 => "tq1_0",
            GgufFtype::MostlyTQ2_0 => "tq2_0",
            GgufFtype::MostlyMXFP4_MOE => "mxfp4_moe",
        }
    }

    /// Inverse of `name()` — parse CLI string back to GgufFtype.
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "f32" => GgufFtype::AllF32,
            "f16" => GgufFtype::MostlyF16,
            "bf16" => GgufFtype::BF16,
            "q4_0" => GgufFtype::MostlyQ4_0,
            "q4_1" => GgufFtype::MostlyQ4_1,
            "q5_0" => GgufFtype::MostlyQ5_0,
            "q5_1" => GgufFtype::MostlyQ5_1,
            "q8_0" => GgufFtype::MostlyQ8_0,
            "iq4_nl" => GgufFtype::MostlyIQ4_NL,
            "q2_k" => GgufFtype::MostlyQ2_K,
            "q3_k_s" => GgufFtype::MostlyQ3_K_S,
            "q3_k_m" => GgufFtype::MostlyQ3_K_M,
            "q3_k_l" => GgufFtype::MostlyQ3_K_L,
            "q4_k_s" => GgufFtype::MostlyQ4_K_S,
            "q4_k_m" => GgufFtype::MostlyQ4_K_M,
            "q5_k_s" => GgufFtype::MostlyQ5_K_S,
            "q5_k_m" => GgufFtype::MostlyQ5_K_M,
            "q6_k" => GgufFtype::MostlyQ6_K,
            "iq2_xxs" => GgufFtype::MostlyIQ2_XXS,
            "iq2_xs" => GgufFtype::MostlyIQ2_XS,
            "q2_k_s" => GgufFtype::MostlyQ2_K_S,
            "iq3_xs" => GgufFtype::MostlyIQ3_XS,
            "iq3_xxs" => GgufFtype::MostlyIQ3_XXS,
            "iq1_s" => GgufFtype::MostlyIQ1_S,
            "iq3_s" => GgufFtype::MostlyIQ3_S,
            "iq3_m" => GgufFtype::MostlyIQ3_M,
            "iq2_s" => GgufFtype::MostlyIQ2_S,
            "iq2_m" => GgufFtype::MostlyIQ2_M,
            "iq4_xs" => GgufFtype::MostlyIQ4_XS,
            "iq1_m" => GgufFtype::MostlyIQ1_M,
            "tq1_0" => GgufFtype::MostlyTQ1_0,
            "tq2_0" => GgufFtype::MostlyTQ2_0,
            "mxfp4_moe" => GgufFtype::MostlyMXFP4_MOE,
            _ => return None,
        })
    }
}

impl TryFrom<u32> for GgufFtype {
    type Error = QuantizeError;
    fn try_from(v: u32) -> Result<Self, Self::Error> {
        Ok(match v {
            0 => GgufFtype::AllF32,
            1 => GgufFtype::MostlyF16,
            2 => GgufFtype::MostlyQ4_0,
            3 => GgufFtype::MostlyQ4_1,
            7 => GgufFtype::MostlyQ8_0,
            8 => GgufFtype::MostlyQ5_0,
            9 => GgufFtype::MostlyQ5_1,
            10 => GgufFtype::MostlyQ2_K,
            11 => GgufFtype::MostlyQ3_K_S,
            12 => GgufFtype::MostlyQ3_K_M,
            13 => GgufFtype::MostlyQ3_K_L,
            14 => GgufFtype::MostlyQ4_K_S,
            15 => GgufFtype::MostlyQ4_K_M,
            16 => GgufFtype::MostlyQ5_K_S,
            17 => GgufFtype::MostlyQ5_K_M,
            18 => GgufFtype::MostlyQ6_K,
            19 => GgufFtype::MostlyIQ2_XXS,
            20 => GgufFtype::MostlyIQ2_XS,
            21 => GgufFtype::MostlyQ2_K_S,
            22 => GgufFtype::MostlyIQ3_XS,
            23 => GgufFtype::MostlyIQ3_XXS,
            24 => GgufFtype::MostlyIQ1_S,
            25 => GgufFtype::MostlyIQ4_NL,
            26 => GgufFtype::MostlyIQ3_S,
            27 => GgufFtype::MostlyIQ3_M,
            28 => GgufFtype::MostlyIQ2_S,
            29 => GgufFtype::MostlyIQ2_M,
            30 => GgufFtype::MostlyIQ4_XS,
            31 => GgufFtype::MostlyIQ1_M,
            32 => GgufFtype::BF16,
            36 => GgufFtype::MostlyTQ1_0,
            37 => GgufFtype::MostlyTQ2_0,
            38 => GgufFtype::MostlyMXFP4_MOE,
            other => return Err(QuantizeError::UnknownGgufFtype(other)),
        })
    }
}

impl From<GgufFtype> for u32 {
    fn from(t: GgufFtype) -> u32 {
        t as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_type_matches_canonical_pick() {
        assert_eq!(GgufFtype::MostlyQ4_0.primary_type(), GgmlType::Q4_0);
        assert_eq!(GgufFtype::MostlyQ5_K_M.primary_type(), GgmlType::Q5_K);
        assert_eq!(GgufFtype::MostlyQ8_0.primary_type(), GgmlType::Q8_0);
        assert_eq!(GgufFtype::MostlyIQ4_NL.primary_type(), GgmlType::IQ4_NL);
        assert_eq!(GgufFtype::AllF32.primary_type(), GgmlType::F32);
        assert_eq!(GgufFtype::BF16.primary_type(), GgmlType::BF16);
    }

    #[test]
    fn name_round_trip() {
        let cases = &[
            GgufFtype::AllF32,
            GgufFtype::MostlyQ4_0,
            GgufFtype::MostlyQ4_K_M,
            GgufFtype::MostlyQ5_K_M,
            GgufFtype::MostlyQ6_K,
            GgufFtype::MostlyIQ4_NL,
            GgufFtype::BF16,
        ];
        for &f in cases {
            assert_eq!(GgufFtype::from_name(f.name()), Some(f));
        }
    }

    #[test]
    fn u32_round_trip() {
        for v in [
            0u32, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 25, 32,
        ] {
            let f = GgufFtype::try_from(v).unwrap();
            assert_eq!(u32::from(f), v);
        }
    }

    #[test]
    fn unknown_u32_errors() {
        // Holes are 4, 5, 6 (old Q4_1_SOME_F16 / Q4_2 / Q4_3 removed),
        // 33, 34, 35 (old Q4_0_4_4/4_8/8_8 removed), 39+ (NVFP4/Q1_0,
        // out of C-fidelity scope here).
        assert!(GgufFtype::try_from(4).is_err());
        assert!(GgufFtype::try_from(5).is_err());
        assert!(GgufFtype::try_from(6).is_err());
        assert!(GgufFtype::try_from(33).is_err());
        assert!(GgufFtype::try_from(34).is_err());
        assert!(GgufFtype::try_from(35).is_err());
        assert!(GgufFtype::try_from(99).is_err());
    }
}
