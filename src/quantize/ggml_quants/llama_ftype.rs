//! `LlamaFtype` — Rust enum mirroring llama.cpp's `enum llama_ftype`
//! (`/opt/llama.cpp/include/llama.h`) at the literal numeric values so
//! the GGUF header `general.file_type` byte matches across pipelines.
//!
//! Per ADR-033 Decision §"LlamaFtype mapping (Decision §2 concrete)".
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
#[allow(non_camel_case_types)] // mirrors llama.cpp's LLAMA_FTYPE_MOSTLY_Q4_0-style names
pub enum LlamaFtype {
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
    MostlyIQ4_NL = 25,
    BF16 = 32,
}

impl LlamaFtype {
    /// The "primary" GgmlType for non-special tensors at this ftype.
    /// Mirrors the early default-pick logic in llama.cpp's
    /// `llama_model_quantize_impl` before `llama_tensor_get_type_impl`
    /// runs the per-tensor override pass.
    pub const fn primary_type(self) -> GgmlType {
        match self {
            LlamaFtype::AllF32 => GgmlType::F32,
            LlamaFtype::MostlyF16 => GgmlType::F16,
            LlamaFtype::BF16 => GgmlType::BF16,
            LlamaFtype::MostlyQ4_0 => GgmlType::Q4_0,
            LlamaFtype::MostlyQ4_1 => GgmlType::Q4_1,
            LlamaFtype::MostlyQ5_0 => GgmlType::Q5_0,
            LlamaFtype::MostlyQ5_1 => GgmlType::Q5_1,
            LlamaFtype::MostlyQ8_0 => GgmlType::Q8_0,
            LlamaFtype::MostlyIQ4_NL => GgmlType::IQ4_NL,
            LlamaFtype::MostlyQ2_K => GgmlType::Q2_K,
            LlamaFtype::MostlyQ3_K_S | LlamaFtype::MostlyQ3_K_M | LlamaFtype::MostlyQ3_K_L => GgmlType::Q3_K,
            LlamaFtype::MostlyQ4_K_S | LlamaFtype::MostlyQ4_K_M => GgmlType::Q4_K,
            LlamaFtype::MostlyQ5_K_S | LlamaFtype::MostlyQ5_K_M => GgmlType::Q5_K,
            LlamaFtype::MostlyQ6_K => GgmlType::Q6_K,
        }
    }

    /// CLI/log name matching the `--quant <name>` surface in Decision §6.
    pub const fn name(self) -> &'static str {
        match self {
            LlamaFtype::AllF32 => "f32",
            LlamaFtype::MostlyF16 => "f16",
            LlamaFtype::BF16 => "bf16",
            LlamaFtype::MostlyQ4_0 => "q4_0",
            LlamaFtype::MostlyQ4_1 => "q4_1",
            LlamaFtype::MostlyQ5_0 => "q5_0",
            LlamaFtype::MostlyQ5_1 => "q5_1",
            LlamaFtype::MostlyQ8_0 => "q8_0",
            LlamaFtype::MostlyIQ4_NL => "iq4_nl",
            LlamaFtype::MostlyQ2_K => "q2_k",
            LlamaFtype::MostlyQ3_K_S => "q3_k_s",
            LlamaFtype::MostlyQ3_K_M => "q3_k_m",
            LlamaFtype::MostlyQ3_K_L => "q3_k_l",
            LlamaFtype::MostlyQ4_K_S => "q4_k_s",
            LlamaFtype::MostlyQ4_K_M => "q4_k_m",
            LlamaFtype::MostlyQ5_K_S => "q5_k_s",
            LlamaFtype::MostlyQ5_K_M => "q5_k_m",
            LlamaFtype::MostlyQ6_K => "q6_k",
        }
    }

    /// Inverse of `name()` — parse CLI string back to LlamaFtype.
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "f32" => LlamaFtype::AllF32,
            "f16" => LlamaFtype::MostlyF16,
            "bf16" => LlamaFtype::BF16,
            "q4_0" => LlamaFtype::MostlyQ4_0,
            "q4_1" => LlamaFtype::MostlyQ4_1,
            "q5_0" => LlamaFtype::MostlyQ5_0,
            "q5_1" => LlamaFtype::MostlyQ5_1,
            "q8_0" => LlamaFtype::MostlyQ8_0,
            "iq4_nl" => LlamaFtype::MostlyIQ4_NL,
            "q2_k" => LlamaFtype::MostlyQ2_K,
            "q3_k_s" => LlamaFtype::MostlyQ3_K_S,
            "q3_k_m" => LlamaFtype::MostlyQ3_K_M,
            "q3_k_l" => LlamaFtype::MostlyQ3_K_L,
            "q4_k_s" => LlamaFtype::MostlyQ4_K_S,
            "q4_k_m" => LlamaFtype::MostlyQ4_K_M,
            "q5_k_s" => LlamaFtype::MostlyQ5_K_S,
            "q5_k_m" => LlamaFtype::MostlyQ5_K_M,
            "q6_k" => LlamaFtype::MostlyQ6_K,
            _ => return None,
        })
    }
}

impl TryFrom<u32> for LlamaFtype {
    type Error = QuantizeError;
    fn try_from(v: u32) -> Result<Self, Self::Error> {
        Ok(match v {
            0 => LlamaFtype::AllF32,
            1 => LlamaFtype::MostlyF16,
            2 => LlamaFtype::MostlyQ4_0,
            3 => LlamaFtype::MostlyQ4_1,
            7 => LlamaFtype::MostlyQ8_0,
            8 => LlamaFtype::MostlyQ5_0,
            9 => LlamaFtype::MostlyQ5_1,
            10 => LlamaFtype::MostlyQ2_K,
            11 => LlamaFtype::MostlyQ3_K_S,
            12 => LlamaFtype::MostlyQ3_K_M,
            13 => LlamaFtype::MostlyQ3_K_L,
            14 => LlamaFtype::MostlyQ4_K_S,
            15 => LlamaFtype::MostlyQ4_K_M,
            16 => LlamaFtype::MostlyQ5_K_S,
            17 => LlamaFtype::MostlyQ5_K_M,
            18 => LlamaFtype::MostlyQ6_K,
            25 => LlamaFtype::MostlyIQ4_NL,
            32 => LlamaFtype::BF16,
            other => return Err(QuantizeError::UnknownLlamaFtype(other)),
        })
    }
}

impl From<LlamaFtype> for u32 {
    fn from(t: LlamaFtype) -> u32 {
        t as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_type_matches_canonical_pick() {
        assert_eq!(LlamaFtype::MostlyQ4_0.primary_type(), GgmlType::Q4_0);
        assert_eq!(LlamaFtype::MostlyQ5_K_M.primary_type(), GgmlType::Q5_K);
        assert_eq!(LlamaFtype::MostlyQ8_0.primary_type(), GgmlType::Q8_0);
        assert_eq!(LlamaFtype::MostlyIQ4_NL.primary_type(), GgmlType::IQ4_NL);
        assert_eq!(LlamaFtype::AllF32.primary_type(), GgmlType::F32);
        assert_eq!(LlamaFtype::BF16.primary_type(), GgmlType::BF16);
    }

    #[test]
    fn name_round_trip() {
        let cases = &[
            LlamaFtype::AllF32,
            LlamaFtype::MostlyQ4_0,
            LlamaFtype::MostlyQ4_K_M,
            LlamaFtype::MostlyQ5_K_M,
            LlamaFtype::MostlyQ6_K,
            LlamaFtype::MostlyIQ4_NL,
            LlamaFtype::BF16,
        ];
        for &f in cases {
            assert_eq!(LlamaFtype::from_name(f.name()), Some(f));
        }
    }

    #[test]
    fn u32_round_trip() {
        for v in [0u32, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 25, 32] {
            let f = LlamaFtype::try_from(v).unwrap();
            assert_eq!(u32::from(f), v);
        }
    }

    #[test]
    fn unknown_u32_errors() {
        assert!(LlamaFtype::try_from(4).is_err());  // reserved hole
        assert!(LlamaFtype::try_from(19).is_err()); // reserved (IQ family)
        assert!(LlamaFtype::try_from(99).is_err());
    }
}
