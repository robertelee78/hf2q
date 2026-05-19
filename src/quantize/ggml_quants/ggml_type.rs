//! `GgmlType` — Rust enum mirroring llama.cpp's `enum ggml_type`
//! (`/opt/llama.cpp/ggml/include/ggml.h:389`) at the literal numeric
//! values so the GGUF header bytes match byte-for-byte across the
//! two pipelines.
//!
//! Per ADR-033 Decision §"Per-tensor IR (Decision §1 concrete)" and
//! the LlamaFtype mapping in §"LlamaFtype mapping (Decision §2 concrete)".
//!
//! Holes in the numeric space (4, 5, 16..19, 21..29) are llama.cpp
//! values currently out of v1 scope (TQ1_0/TQ2_0/IQ2_*/IQ3_*/IQ1_*).
//! Add only when the matching `Quantizer` impl ships.

use super::error::QuantizeError;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)] // mirrors llama.cpp's GGML_TYPE_Q4_0-style names verbatim
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // 4, 5 unused
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    // 16..23 + 29: IQ-family. Placeholders for target_for return-values
    // — `llama_tensor_get_type_impl` selects these for several
    // ftype/category combinations (e.g. IQ3_S on attn_v at IQ3_XXS).
    // The convert pipeline has no Quantizer impl for these in v1; if
    // target_for returns one of these, the downstream `quantizer_for`
    // call surfaces `QuantizeError::NoQuantizerForType` instead of
    // silently demoting. Per [[feedback-no-loop-suppression-2026-05-17]].
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
}

impl GgmlType {
    /// Number of f32 elements per disk-format block. From
    /// `/opt/llama.cpp/ggml/src/ggml-common.h` defines (`QK4_0=32`,
    /// `QK_K=256`, etc.).
    pub const fn block_size(self) -> usize {
        match self {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => 1,
            GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q8_1
            | GgmlType::IQ4_NL
            | GgmlType::MXFP4 => 32,
            GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_K
            | GgmlType::IQ2_XXS
            | GgmlType::IQ2_XS
            | GgmlType::IQ3_XXS
            | GgmlType::IQ1_S
            | GgmlType::IQ3_S
            | GgmlType::IQ2_S
            | GgmlType::IQ4_XS
            | GgmlType::IQ1_M
            | GgmlType::TQ1_0
            | GgmlType::TQ2_0 => 256,
        }
    }

    /// Bytes per disk-format block. From the `static_assert`s in
    /// `/opt/llama.cpp/ggml/src/ggml-common.h` (`sizeof(block_q4_0)`,
    /// etc.). Mirrored as a Rust const so byte-cmp regressions surface
    /// at compile time if a kernel's BLOCK_BYTES drifts from this.
    pub const fn type_size(self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::BF16 => 2,
            GgmlType::Q4_0 => 18,     // f16 d + 16 nibbles
            GgmlType::Q4_1 => 20,     // f16 d + f16 m + 16 nibbles
            GgmlType::Q5_0 => 22,     // f16 d + u32 qh + 16 nibbles
            GgmlType::Q5_1 => 24,     // f16 d + f16 m + u32 qh + 16 nibbles
            GgmlType::Q8_0 => 34,     // f16 d + 32 i8
            GgmlType::Q8_1 => 36,     // 2*f16 + 32 i8
            GgmlType::IQ4_NL => 18,   // f16 d + 16 nibbles (codebook lookup)
            GgmlType::Q2_K => 84,     // scales[16] + qs[64] + f16 d + f16 dmin
            GgmlType::Q3_K => 110,    // hmask[32] + qs[64] + scales[12] + f16 d
            GgmlType::Q4_K => 144,    // f16 d + f16 dmin + scales[12] + qs[128]
            GgmlType::Q5_K => 176,    // f16 d + f16 dmin + scales[12] + qh[32] + qs[128]
            GgmlType::Q6_K => 210,    // ql[128] + qh[64] + scales[16] (i8) + f16 d
            GgmlType::Q8_K => 292,    // f32 d + qs[256] + bsums[16] (only used internally by llama.cpp)
            // Placeholder IQ/TQ/MXFP4 types — no Quantizer impl in v1.
            // Sizes mirror llama.cpp's ggml-common.h for documentation
            // value; v1 pipeline never emits these rows (NoQuantizerForType).
            GgmlType::IQ2_XXS => 66,
            GgmlType::IQ2_XS => 74,
            GgmlType::IQ3_XXS => 98,
            GgmlType::IQ1_S => 50,
            GgmlType::IQ3_S => 110,
            GgmlType::IQ2_S => 82,
            GgmlType::IQ4_XS => 136,
            GgmlType::IQ1_M => 56,
            GgmlType::TQ1_0 => 54,
            GgmlType::TQ2_0 => 66,
            GgmlType::MXFP4 => 17, // u8 scale + 16 nibbles
        }
    }

    /// Total row size in bytes for `n_per_row` f32 elements. Mirrors
    /// `ggml_row_size` at `ggml/src/ggml.c`.
    pub const fn row_size(self, n_per_row: usize) -> usize {
        (n_per_row / self.block_size()) * self.type_size()
    }

    /// Whether the type structurally requires imatrix at quantize time
    /// (matches `ggml_quantize_requires_imatrix` at `ggml.c:7655`).
    ///
    /// For the v1 set, NONE of the supported types hard-require
    /// imatrix — only `IQ2_XXS / IQ2_XS / IQ1_S` do, and those are out
    /// of v1 scope.
    pub const fn requires_imatrix(self) -> bool {
        false
    }

    /// Canonical lowercase name used for `--quant <name>` CLI dispatch.
    pub const fn name(self) -> &'static str {
        match self {
            GgmlType::F32 => "f32",
            GgmlType::F16 => "f16",
            GgmlType::BF16 => "bf16",
            GgmlType::Q4_0 => "q4_0",
            GgmlType::Q4_1 => "q4_1",
            GgmlType::Q5_0 => "q5_0",
            GgmlType::Q5_1 => "q5_1",
            GgmlType::Q8_0 => "q8_0",
            GgmlType::Q8_1 => "q8_1",
            GgmlType::IQ4_NL => "iq4_nl",
            GgmlType::Q2_K => "q2_k",
            GgmlType::Q3_K => "q3_k",
            GgmlType::Q4_K => "q4_k",
            GgmlType::Q5_K => "q5_k",
            GgmlType::Q6_K => "q6_k",
            GgmlType::Q8_K => "q8_k",
            GgmlType::IQ2_XXS => "iq2_xxs",
            GgmlType::IQ2_XS => "iq2_xs",
            GgmlType::IQ3_XXS => "iq3_xxs",
            GgmlType::IQ1_S => "iq1_s",
            GgmlType::IQ3_S => "iq3_s",
            GgmlType::IQ2_S => "iq2_s",
            GgmlType::IQ4_XS => "iq4_xs",
            GgmlType::IQ1_M => "iq1_m",
            GgmlType::TQ1_0 => "tq1_0",
            GgmlType::TQ2_0 => "tq2_0",
            GgmlType::MXFP4 => "mxfp4",
        }
    }

    /// Case-insensitive parse of a `GgmlType` name.
    ///
    /// Accepts the canonical lowercase tokens from [`Self::name`] AND
    /// the uppercase variants emitted by `mudler/apex-quant`'s
    /// `configs/<model>_<tier>.txt` files (e.g. `Q5_K`, `Q6_K`, `Q8_0`).
    /// Used by the ADR-033 §9 mudler-config parser to lift the vendored
    /// per-tensor overlay into typed `GgmlType` values.
    ///
    /// Returns `None` for unknown strings; the caller (mudler parser)
    /// surfaces those as typed `ApexError::MudlerConfigParse`.
    pub fn from_name(name: &str) -> Option<Self> {
        // ASCII lowercase fold — every token we accept is ASCII; avoid
        // a `String` allocation for the common case.
        let mut buf = [0u8; 16];
        let bytes = name.as_bytes();
        if bytes.len() > buf.len() {
            return None;
        }
        for (i, &b) in bytes.iter().enumerate() {
            buf[i] = b.to_ascii_lowercase();
        }
        let lower = std::str::from_utf8(&buf[..bytes.len()]).ok()?;
        Some(match lower {
            "f32" => GgmlType::F32,
            "f16" => GgmlType::F16,
            "bf16" => GgmlType::BF16,
            "q4_0" => GgmlType::Q4_0,
            "q4_1" => GgmlType::Q4_1,
            "q5_0" => GgmlType::Q5_0,
            "q5_1" => GgmlType::Q5_1,
            "q8_0" => GgmlType::Q8_0,
            "q8_1" => GgmlType::Q8_1,
            "iq4_nl" => GgmlType::IQ4_NL,
            "q2_k" => GgmlType::Q2_K,
            "q3_k" => GgmlType::Q3_K,
            "q4_k" => GgmlType::Q4_K,
            "q5_k" => GgmlType::Q5_K,
            "q6_k" => GgmlType::Q6_K,
            "q8_k" => GgmlType::Q8_K,
            "iq2_xxs" => GgmlType::IQ2_XXS,
            "iq2_xs" => GgmlType::IQ2_XS,
            "iq3_xxs" => GgmlType::IQ3_XXS,
            "iq1_s" => GgmlType::IQ1_S,
            "iq3_s" => GgmlType::IQ3_S,
            "iq2_s" => GgmlType::IQ2_S,
            "iq4_xs" => GgmlType::IQ4_XS,
            "iq1_m" => GgmlType::IQ1_M,
            "tq1_0" => GgmlType::TQ1_0,
            "tq2_0" => GgmlType::TQ2_0,
            "mxfp4" => GgmlType::MXFP4,
            _ => return None,
        })
    }
}

impl TryFrom<u32> for GgmlType {
    type Error = QuantizeError;
    fn try_from(v: u32) -> Result<Self, Self::Error> {
        Ok(match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2_K,
            11 => GgmlType::Q3_K,
            12 => GgmlType::Q4_K,
            13 => GgmlType::Q5_K,
            14 => GgmlType::Q6_K,
            15 => GgmlType::Q8_K,
            16 => GgmlType::IQ2_XXS,
            17 => GgmlType::IQ2_XS,
            18 => GgmlType::IQ3_XXS,
            19 => GgmlType::IQ1_S,
            20 => GgmlType::IQ4_NL,
            21 => GgmlType::IQ3_S,
            22 => GgmlType::IQ2_S,
            23 => GgmlType::IQ4_XS,
            29 => GgmlType::IQ1_M,
            30 => GgmlType::BF16,
            34 => GgmlType::TQ1_0,
            35 => GgmlType::TQ2_0,
            39 => GgmlType::MXFP4,
            other => return Err(QuantizeError::UnknownGgmlType(other)),
        })
    }
}

impl From<GgmlType> for u32 {
    fn from(t: GgmlType) -> u32 {
        t as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-check our type_size() against each kernel's BLOCK_BYTES.
    /// If a kernel module's BLOCK_BYTES drifts from this table, byte-cmp
    /// tests would fail; this guard catches the drift earlier.
    #[test]
    fn type_size_matches_kernel_constants() {
        use super::super::*;
        assert_eq!(GgmlType::Q4_0.type_size(), q4_0::BLOCK_BYTES);
        assert_eq!(GgmlType::Q4_1.type_size(), q4_1::BLOCK_BYTES);
        assert_eq!(GgmlType::Q5_0.type_size(), q5_0::BLOCK_BYTES);
        assert_eq!(GgmlType::Q5_1.type_size(), q5_1::BLOCK_BYTES);
        assert_eq!(GgmlType::Q8_0.type_size(), q8_0::BLOCK_BYTES);
        assert_eq!(GgmlType::IQ4_NL.type_size(), iq4_nl::BLOCK_BYTES);
        assert_eq!(GgmlType::Q2_K.type_size(), q2_k::BLOCK_BYTES);
        assert_eq!(GgmlType::Q3_K.type_size(), q3_k::BLOCK_BYTES);
        assert_eq!(GgmlType::Q4_K.type_size(), q4_k::BLOCK_BYTES);
        assert_eq!(GgmlType::Q5_K.type_size(), q5_k::BLOCK_BYTES);
        assert_eq!(GgmlType::Q6_K.type_size(), q6_k::BLOCK_BYTES);
    }

    #[test]
    fn block_size_matches_kernel_constants() {
        use super::super::*;
        assert_eq!(GgmlType::Q4_0.block_size(), q4_0::QK4_0);
        assert_eq!(GgmlType::Q8_0.block_size(), q8_0::QK8_0);
        // K-family all use QK_K = 256
        assert_eq!(GgmlType::Q4_K.block_size(), 256);
        assert_eq!(GgmlType::Q5_K.block_size(), 256);
    }

    #[test]
    fn round_trip_u32() {
        for t in [
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::IQ4_NL,
        ] {
            let v: u32 = t.into();
            let back = GgmlType::try_from(v).unwrap();
            assert_eq!(t, back);
        }
    }

    #[test]
    fn unknown_u32_errors() {
        assert!(GgmlType::try_from(4).is_err()); // hole
        assert!(GgmlType::try_from(100).is_err());
    }

    #[test]
    fn row_size_legacy() {
        // 64 f32 elements at Q4_0 = 2 blocks × 18 bytes = 36 bytes
        assert_eq!(GgmlType::Q4_0.row_size(64), 36);
        // 32 elements at Q5_0 = 1 block × 22 bytes
        assert_eq!(GgmlType::Q5_0.row_size(32), 22);
    }

    /// Mudler's `configs/<model>_<tier>.txt` files mix lowercase
    /// (`q5_K`, `iq4_xs`) and uppercase (`Q5_K`, `Q8_0`) variants for
    /// the same logical type. `from_name` MUST fold both to the same
    /// `GgmlType`. Used by the ADR-033 §9 mudler-config parser.
    #[test]
    fn from_name_case_insensitive_for_mudler_configs() {
        // Canonical lowercase set.
        assert_eq!(GgmlType::from_name("q5_k"), Some(GgmlType::Q5_K));
        assert_eq!(GgmlType::from_name("q6_k"), Some(GgmlType::Q6_K));
        assert_eq!(GgmlType::from_name("q8_0"), Some(GgmlType::Q8_0));
        assert_eq!(GgmlType::from_name("iq4_xs"), Some(GgmlType::IQ4_XS));
        assert_eq!(GgmlType::from_name("iq2_s"), Some(GgmlType::IQ2_S));

        // Mixed-case as emitted by vendor configs.
        assert_eq!(GgmlType::from_name("Q5_K"), Some(GgmlType::Q5_K));
        assert_eq!(GgmlType::from_name("Q6_K"), Some(GgmlType::Q6_K));
        assert_eq!(GgmlType::from_name("Q8_0"), Some(GgmlType::Q8_0));
        assert_eq!(GgmlType::from_name("q5_K"), Some(GgmlType::Q5_K));

        // Unknown tokens surface as None.
        assert_eq!(GgmlType::from_name("notatype"), None);
        assert_eq!(GgmlType::from_name(""), None);
        // Over-length input (would overflow the 16-byte scratch buffer)
        // returns None instead of panicking.
        assert_eq!(GgmlType::from_name("very_long_unknown_type_name"), None);
    }

    #[test]
    fn row_size_k_family() {
        // 512 f32 elements at Q4_K = 2 super-blocks × 144 bytes = 288 bytes
        assert_eq!(GgmlType::Q4_K.row_size(512), 288);
        // 256 elements at Q6_K = 1 super-block × 210 bytes
        assert_eq!(GgmlType::Q6_K.row_size(256), 210);
    }
}
