//! `QuantSelector` — unified `--quant <name>` parser for `hf2q convert-v2`.
//!
//! Per ADR-033 Decision §6, `hf2q convert-v2 --quant <name>` accepts three
//! disjoint name-spaces:
//!
//! 1. **Standard llama.cpp ftypes** — `f32`, `f16`, `bf16`, `q4_0`,
//!    `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q2_k`, `q3_k_s/m/l`, `q4_k_s/m`,
//!    `q5_k_s/m`, `q6_k`, `iq4_nl`, etc. — parsed via
//!    [`LlamaFtype::from_name`].
//! 2. **Apex algorithmic tiers** — `apex-quality`, `apex-i-quality`,
//!    `apex-balanced`, `apex-i-balanced`, `apex-compact`, `apex-i-compact`,
//!    `apex-mini`. Resolved to [`ApexTier`]; the driver pairs each tier
//!    with an `ApexPolicy` built from the source model's `n_layers` and
//!    `n_expert`.
//! 3. **Apex custom** — `apex-custom` carries an operator-supplied
//!    `--tensor-type-file <path>` (out of v1 scope for the convert-v2
//!    driver — typed error stub only).
//!
//! Per [[feedback-no-loop-suppression-2026-05-17]] + ADR §"reserved-name
//! typed-error stubs": every unsupported / reserved name surfaces as a
//! typed [`QuantSelectorError`] — never silent fallback. Reserved names
//! (`dwq`, bare `apex`, `tq1_0`, `tq2_0`) are explicitly rejected with
//! diagnostic errors.
//!
//! Per [[feedback-no-backwards-compat-2026-05-18]]: no compat aliases for
//! legacy `--quant` values.

use std::path::PathBuf;

use crate::quantize::ggml_quants::apex::{
    ApexTier, SUPPORTED_APEX_TIERS,
};
use crate::quantize::ggml_quants::LlamaFtype;

/// One resolved `--quant <name>` selector.
///
/// The CLI layer parses the operator-supplied string into this enum via
/// [`QuantSelector::from_name`]; the convert-v2 driver then branches on
/// the variant to pick the right policy (StandardPolicy vs ApexPolicy
/// vs operator-supplied tensor-type file).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantSelector {
    /// Standard llama.cpp file-type. The driver builds the orchestrator
    /// with `StandardPolicy` and emits `general.file_type` = the
    /// underlying [`LlamaFtype`] discriminant.
    Standard(LlamaFtype),
    /// Apex algorithmic tier. The driver builds an `ApexPolicy { tier,
    /// n_layers, n_expert }` from the source model's `config.json`.
    /// `general.file_type` carries the closest standard LlamaFtype
    /// approximation (see [`approximate_for_apex`]).
    Apex(ApexTier),
    /// Operator-supplied `apex-custom --tensor-type-file <path>`. Out of
    /// v1 scope for the convert-v2 driver; placeholder for the future
    /// per-tensor override path described in ADR §"Per-model APEX config
    /// override".
    ApexCustom(PathBuf),
}

/// Typed errors raised by [`QuantSelector::from_name`].
///
/// Per [[feedback-no-loop-suppression-2026-05-17]]: every unsupported
/// `--quant` value is rejected with a structured error; the driver never
/// falls back to a default.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum QuantSelectorError {
    /// The name didn't match any standard `LlamaFtype` or apex prefix.
    #[error("unknown --quant value `{name}` (no LlamaFtype / Apex tier mapping)")]
    UnknownQuant { name: String },

    /// `--quant apex-<X>` where `<X>` isn't in the v1 supported tier
    /// set. v1 tiers per ADR §6:
    /// `quality / i-quality / balanced / i-balanced / compact / i-compact / mini`.
    #[error(
        "unknown Apex tier `apex-{tier}`; v1 supports {supported:?}"
    )]
    UnknownApexTier {
        tier: String,
        supported: &'static [&'static str],
    },

    /// Mudler's experimental tiers (`nano / i-nano / micro / i-micro`)
    /// were dropped from v1's surface (per ADR §"Decision §6"). They are
    /// reachable only via `--quant apex-custom --tensor-type-file <file>`
    /// with the vendored per-model config.
    #[error(
        "Apex tier `apex-{tier}` is out of v1 scope (mudler's experimental tiers); \
         use `--quant apex-custom --tensor-type-file <vendored-config>` instead"
    )]
    ApexTierOutOfScope { tier: String },

    /// `apex-custom` requires `--tensor-type-file <path>`. The selector
    /// parses the `--quant` flag in isolation; the tensor-type-file
    /// resolution happens at a higher CLI layer when it's wired.
    #[error("--quant apex-custom requires --tensor-type-file <path>")]
    ApexCustomRequiresTensorTypeFile,

    /// `--quant dwq` is reserved for a future training-loop entry point
    /// (ADR-020 follow-up). Surface a typed error rather than silently
    /// treating it as an unknown quant.
    #[error(
        "--quant dwq is reserved for the future DWQ-train pipeline; \
         not implemented in convert-v2"
    )]
    DwqReserved,

    /// Bare `apex` without a tier suffix. Per ADR §6 the operator MUST
    /// pick a tier explicitly — there is no implicit Apex default.
    #[error(
        "--quant apex is unqualified; use one of {supported:?} or `apex-custom`"
    )]
    ApexUnqualified {
        supported: &'static [&'static str],
    },

    /// TQ1_0 / TQ2_0 are valid `LlamaFtype` variants but out of v1
    /// convert-v2 scope (no quantizer implementation — see
    /// `quantizer.rs`). We surface a more diagnostic error than the
    /// generic UnknownQuant.
    #[error(
        "--quant {name} is a recognized ftype but out of v1 convert-v2 scope \
         (no Quantizer impl)"
    )]
    TqOutOfV1Scope { name: String },
}

impl QuantSelector {
    /// Parse a `--quant <name>` string into the matching selector.
    ///
    /// Resolution order:
    /// 1. Try [`LlamaFtype::from_name`] first — the most common case.
    /// 2. Try the `apex-<tier>` prefix.
    /// 3. Match reserved-name typed errors (`dwq`, bare `apex`,
    ///    `tq1_0`, `tq2_0`).
    /// 4. Anything else: [`QuantSelectorError::UnknownQuant`].
    ///
    /// Per [[feedback-no-backwards-compat-2026-05-18]]: no compat aliases
    /// for legacy `--quant` values.
    pub fn from_name(s: &str) -> Result<Self, QuantSelectorError> {
        // 1. Standard llama.cpp ftypes. TQ1_0 / TQ2_0 are members of
        //    LlamaFtype but the convert-v2 pipeline has no Quantizer impl
        //    for them today — surface a more diagnostic error rather than
        //    letting them slip through and panic at quantize time.
        match s {
            "tq1_0" | "tq2_0" => {
                return Err(QuantSelectorError::TqOutOfV1Scope {
                    name: s.to_string(),
                });
            }
            _ => {}
        }
        if let Some(ftype) = LlamaFtype::from_name(s) {
            return Ok(QuantSelector::Standard(ftype));
        }

        // 2. Apex tier names.
        if let Some(rest) = s.strip_prefix("apex-") {
            let tier = match rest {
                "quality" => ApexTier::Quality,
                "i-quality" => ApexTier::IQuality,
                "balanced" => ApexTier::Balanced,
                "i-balanced" => ApexTier::IBalanced,
                "compact" => ApexTier::Compact,
                "i-compact" => ApexTier::ICompact,
                "mini" => ApexTier::Mini,
                "custom" => {
                    return Err(QuantSelectorError::ApexCustomRequiresTensorTypeFile);
                }
                // Mudler experimental tiers dropped from v1 surface per
                // ADR §6 — reachable only via apex-custom.
                "nano" | "i-nano" | "micro" | "i-micro" => {
                    return Err(QuantSelectorError::ApexTierOutOfScope {
                        tier: rest.to_string(),
                    });
                }
                _ => {
                    return Err(QuantSelectorError::UnknownApexTier {
                        tier: rest.to_string(),
                        supported: SUPPORTED_APEX_TIERS,
                    });
                }
            };
            return Ok(QuantSelector::Apex(tier));
        }

        // 3. Reserved-name typed errors.
        match s {
            "dwq" => Err(QuantSelectorError::DwqReserved),
            "apex" => Err(QuantSelectorError::ApexUnqualified {
                supported: SUPPORTED_APEX_TIERS,
            }),
            _ => Err(QuantSelectorError::UnknownQuant {
                name: s.to_string(),
            }),
        }
    }
}

/// Closest standard [`LlamaFtype`] for an [`ApexTier`].
///
/// Apex tiers are mixed-precision recipes — no single LlamaFtype is a
/// faithful encoding. We pick the closest "headline" ftype so the GGUF
/// `general.file_type` byte at least clues operators / inspectors into
/// the tier's bit-budget class:
///
/// | Apex tier              | Approximate LlamaFtype |
/// |------------------------|------------------------|
/// | Quality / IQuality     | MostlyQ6_K (18)        |
/// | Balanced / IBalanced   | MostlyQ5_K_M (17)      |
/// | Compact / ICompact     | MostlyQ4_K_M (15)      |
/// | Mini                   | MostlyQ3_K_S (11)      |
///
/// The rationale is the dominant mid-band expert quant for each tier:
/// quality's `mid_exp=IQ4_XS` floats up to `mid_attn=Q6_K`; balanced is
/// `mid_exp=Q5_K`; compact is `mid_exp=Q3_K + mid_attn=Q4_K`; mini is
/// `mid_exp=IQ2_S + mid_attn=Q3_K`. **This is purely a
/// header-metadata pick** — actual per-tensor types are decided by
/// `ApexPolicy::target_for` and recorded on each tensor's own
/// `ggml_type` field, not via the file-type byte.
pub const fn approximate_for_apex(tier: ApexTier) -> LlamaFtype {
    match tier {
        ApexTier::Quality | ApexTier::IQuality => LlamaFtype::MostlyQ6_K,
        ApexTier::Balanced | ApexTier::IBalanced => LlamaFtype::MostlyQ5_K_M,
        ApexTier::Compact | ApexTier::ICompact => LlamaFtype::MostlyQ4_K_M,
        ApexTier::Mini => LlamaFtype::MostlyQ3_K_S,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_quant_selector_standard_round_trip() {
        // Spot-check a handful of standard ftype names — full coverage
        // lives in `LlamaFtype::name_round_trip` already.
        assert_eq!(
            QuantSelector::from_name("q5_k_m").unwrap(),
            QuantSelector::Standard(LlamaFtype::MostlyQ5_K_M)
        );
        assert_eq!(
            QuantSelector::from_name("q8_0").unwrap(),
            QuantSelector::Standard(LlamaFtype::MostlyQ8_0)
        );
        assert_eq!(
            QuantSelector::from_name("f16").unwrap(),
            QuantSelector::Standard(LlamaFtype::MostlyF16)
        );
        assert_eq!(
            QuantSelector::from_name("iq4_nl").unwrap(),
            QuantSelector::Standard(LlamaFtype::MostlyIQ4_NL)
        );
    }

    #[test]
    fn parse_quant_selector_apex_round_trip() {
        assert_eq!(
            QuantSelector::from_name("apex-balanced").unwrap(),
            QuantSelector::Apex(ApexTier::Balanced)
        );
        assert_eq!(
            QuantSelector::from_name("apex-quality").unwrap(),
            QuantSelector::Apex(ApexTier::Quality)
        );
        assert_eq!(
            QuantSelector::from_name("apex-compact").unwrap(),
            QuantSelector::Apex(ApexTier::Compact)
        );
        assert_eq!(
            QuantSelector::from_name("apex-mini").unwrap(),
            QuantSelector::Apex(ApexTier::Mini)
        );
    }

    #[test]
    fn parse_quant_selector_apex_i_variant() {
        assert_eq!(
            QuantSelector::from_name("apex-i-quality").unwrap(),
            QuantSelector::Apex(ApexTier::IQuality)
        );
        assert_eq!(
            QuantSelector::from_name("apex-i-balanced").unwrap(),
            QuantSelector::Apex(ApexTier::IBalanced)
        );
        assert_eq!(
            QuantSelector::from_name("apex-i-compact").unwrap(),
            QuantSelector::Apex(ApexTier::ICompact)
        );
    }

    #[test]
    fn parse_quant_selector_apex_custom_errors() {
        let err = QuantSelector::from_name("apex-custom").unwrap_err();
        assert!(matches!(
            err,
            QuantSelectorError::ApexCustomRequiresTensorTypeFile
        ));
    }

    #[test]
    fn parse_quant_selector_dwq_reserved() {
        let err = QuantSelector::from_name("dwq").unwrap_err();
        assert!(matches!(err, QuantSelectorError::DwqReserved));
    }

    #[test]
    fn parse_quant_selector_apex_nano_out_of_scope() {
        let err = QuantSelector::from_name("apex-nano").unwrap_err();
        match err {
            QuantSelectorError::ApexTierOutOfScope { tier } => {
                assert_eq!(tier, "nano");
            }
            other => panic!("expected ApexTierOutOfScope, got {other:?}"),
        }

        // Same for i-nano / micro / i-micro.
        for v in ["apex-i-nano", "apex-micro", "apex-i-micro"] {
            let err = QuantSelector::from_name(v).unwrap_err();
            assert!(
                matches!(err, QuantSelectorError::ApexTierOutOfScope { .. }),
                "{v} should be ApexTierOutOfScope, got {err:?}"
            );
        }
    }

    #[test]
    fn parse_quant_selector_bare_apex_unqualified() {
        let err = QuantSelector::from_name("apex").unwrap_err();
        match err {
            QuantSelectorError::ApexUnqualified { supported } => {
                assert!(supported.contains(&"balanced"));
                assert!(supported.contains(&"quality"));
                assert!(supported.contains(&"mini"));
            }
            other => panic!("expected ApexUnqualified, got {other:?}"),
        }
    }

    #[test]
    fn parse_quant_selector_unknown_apex_tier_errors() {
        let err = QuantSelector::from_name("apex-bogus").unwrap_err();
        match err {
            QuantSelectorError::UnknownApexTier { tier, supported } => {
                assert_eq!(tier, "bogus");
                assert!(supported.contains(&"balanced"));
            }
            other => panic!("expected UnknownApexTier, got {other:?}"),
        }
    }

    #[test]
    fn parse_quant_selector_tq_out_of_v1_scope() {
        let err = QuantSelector::from_name("tq1_0").unwrap_err();
        assert!(matches!(err, QuantSelectorError::TqOutOfV1Scope { .. }));
        let err = QuantSelector::from_name("tq2_0").unwrap_err();
        assert!(matches!(err, QuantSelectorError::TqOutOfV1Scope { .. }));
    }

    #[test]
    fn parse_quant_selector_unknown_quant_errors() {
        let err = QuantSelector::from_name("garbage").unwrap_err();
        match err {
            QuantSelectorError::UnknownQuant { name } => assert_eq!(name, "garbage"),
            other => panic!("expected UnknownQuant, got {other:?}"),
        }
    }

    // ============================================================================
    // ADR-033 §P7 AC#3 — every typed-error variant's MESSAGE contains
    // the actionable hint the operator needs to recover (the supported
    // alternative / the tracking issue / the future-ADR reference).
    //
    // Variant-only matching is not enough: a future refactor that
    // removes the hint from the `#[error(...)]` template would not be
    // caught by `matches!(err, Variant)`. These tests `.to_string()`
    // the error and assert substring presence, so the user-facing
    // diagnostic stays informative.
    // ============================================================================

    #[test]
    fn p7_ac3_hint_dwq_reserved() {
        let msg = QuantSelectorError::DwqReserved.to_string();
        assert!(msg.contains("dwq"), "msg should name the rejected flag: {msg}");
        assert!(
            msg.contains("reserved") || msg.contains("future"),
            "msg should hint at the reserved/future-pipeline status: {msg}"
        );
    }

    #[test]
    fn p7_ac3_hint_apex_unqualified() {
        let err = QuantSelectorError::ApexUnqualified {
            supported: SUPPORTED_APEX_TIERS,
        };
        let msg = err.to_string();
        assert!(msg.contains("apex"), "msg should name the flag: {msg}");
        // Must enumerate the supported tier names so the operator
        // can immediately pick one.
        assert!(msg.contains("balanced"), "msg should list `balanced`: {msg}");
        assert!(msg.contains("mini"), "msg should list `mini`: {msg}");
        assert!(
            msg.contains("apex-custom"),
            "msg should mention the apex-custom escape hatch: {msg}"
        );
    }

    #[test]
    fn p7_ac3_hint_tq_out_of_v1_scope() {
        let msg = QuantSelectorError::TqOutOfV1Scope {
            name: "tq1_0".to_string(),
        }
        .to_string();
        assert!(msg.contains("tq1_0"), "msg should echo the rejected name: {msg}");
        assert!(
            msg.contains("out of v1") || msg.contains("scope"),
            "msg should hint at the scope reason: {msg}"
        );
        assert!(
            msg.contains("Quantizer"),
            "msg should reference the missing Quantizer impl: {msg}"
        );
    }

    #[test]
    fn p7_ac3_hint_unknown_apex_tier_lists_supported() {
        let msg = QuantSelectorError::UnknownApexTier {
            tier: "bogus".to_string(),
            supported: SUPPORTED_APEX_TIERS,
        }
        .to_string();
        assert!(msg.contains("bogus"), "msg should echo the bad tier: {msg}");
        assert!(
            msg.contains("balanced"),
            "msg should list the supported tiers (e.g. `balanced`): {msg}"
        );
    }

    #[test]
    fn p7_ac3_hint_apex_custom_requires_tensor_type_file() {
        let msg = QuantSelectorError::ApexCustomRequiresTensorTypeFile.to_string();
        assert!(
            msg.contains("apex-custom") || msg.contains("--tensor-type-file"),
            "msg should name the missing flag the operator must supply: {msg}"
        );
    }

    #[test]
    fn p7_ac3_hint_apex_tier_out_of_scope() {
        let msg = QuantSelectorError::ApexTierOutOfScope {
            tier: "nano".to_string(),
        }
        .to_string();
        assert!(msg.contains("nano"), "msg should echo the rejected tier: {msg}");
        assert!(
            msg.contains("apex-custom") || msg.contains("scope"),
            "msg should hint at the escape hatch or scope reason: {msg}"
        );
    }

    #[test]
    fn p7_ac3_hint_unknown_quant() {
        let msg = QuantSelectorError::UnknownQuant {
            name: "garbage".to_string(),
        }
        .to_string();
        assert!(msg.contains("garbage"), "msg should echo the bad name: {msg}");
    }

    #[test]
    fn approximate_for_apex_table() {
        assert_eq!(approximate_for_apex(ApexTier::Quality), LlamaFtype::MostlyQ6_K);
        assert_eq!(approximate_for_apex(ApexTier::IQuality), LlamaFtype::MostlyQ6_K);
        assert_eq!(
            approximate_for_apex(ApexTier::Balanced),
            LlamaFtype::MostlyQ5_K_M
        );
        assert_eq!(
            approximate_for_apex(ApexTier::IBalanced),
            LlamaFtype::MostlyQ5_K_M
        );
        assert_eq!(
            approximate_for_apex(ApexTier::Compact),
            LlamaFtype::MostlyQ4_K_M
        );
        assert_eq!(
            approximate_for_apex(ApexTier::ICompact),
            LlamaFtype::MostlyQ4_K_M
        );
        assert_eq!(approximate_for_apex(ApexTier::Mini), LlamaFtype::MostlyQ3_K_S);
    }
}
