//! `ApexError` — typed errors for `ApexPolicy::target_for`.
//!
//! Per ADR-033 §"Plan" / Pa exit gate, every failure mode is typed at
//! the policy boundary. Per [[feedback-no-loop-suppression-2026-05-17]]:
//! unsupported arch / tier / missing hparam → typed `Err`, never a
//! silent demotion to F16 or some other quant.
//!
//! `ApexPolicy::target_for` cannot return `Ok(F16)` for a fallback.
//! Vision-tensor F16 emission is handled upstream at the convert
//! dispatcher (it checks `is_vision_tensor_pattern` BEFORE calling
//! the policy).

use thiserror::Error;

use super::super::tensor_ref::ArchName;

/// Typed errors produced by `ApexPolicy::target_for` and friends.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ApexError {
    /// The source model's architecture isn't in v1 Apex's supported
    /// set. Per ADR-033 Decision §6 the v1 set is
    /// `{Qwen35Moe, Gemma4 (if MoE), MiniMaxM2}`. The error carries
    /// the supported-set list so the caller's user-facing message can
    /// reproduce it without hardcoding.
    #[error(
        "Apex tier requires a supported MoE arch; got `{arch}`. \
         Supported v1 arches: {supported:?}"
    )]
    UnsupportedArch {
        arch: &'static str,
        supported: &'static [&'static str],
    },

    /// The model is configured as dense (`n_expert <= 1`) but the
    /// caller requested an Apex tier. v1 Apex is MoE-only by design
    /// (mudler's published recipe is MoE-specific; the routed/shared
    /// expert split is what gives Apex its bit-budget advantage).
    #[error(
        "Apex tier requires an MoE model (n_expert > 1); got `{arch}` with \
         n_expert={n_expert}. Use `--quant q5_k_m` or similar for dense models"
    )]
    DenseModelNotSupported {
        arch: &'static str,
        n_expert: u32,
    },

    /// The named tier string didn't match any known `ApexTier`
    /// variant. v1 supported tiers per ADR §"Decision §6":
    /// `quality / i-quality / balanced / i-balanced / compact / i-compact / mini`.
    /// `nano / i-nano / micro / i-micro` were dropped from mudler's
    /// surface for v1; they're routable via `apex-custom` with an
    /// explicit `--tensor-type-file`.
    #[error(
        "unknown Apex tier `{tier}`; v1 supports {supported:?}. \
         (mudler's `nano/i-nano/micro/i-micro` dropped from v1 surface; \
         use `--quant apex-custom --tensor-type-file ...` if needed)"
    )]
    UnsupportedTier {
        tier: String,
        supported: &'static [&'static str],
    },

    /// `apex-custom` was requested without a `--tensor-type-file`.
    /// Per ADR Decision §6, `apex-custom` is the operator-supplied
    /// per-tensor override path; it has no algorithmic rules.
    #[error("--quant apex-custom requires --tensor-type-file <path>")]
    CustomRequiresTensorTypeFile,

    /// The source `config.json` is missing the `num_hidden_layers`
    /// field (or it's not a positive integer). Apex's per-layer
    /// gradient is keyed on the model's layer count; without it the
    /// EDGE/NEAR/MID partition can't be computed.
    #[error("source config.json missing or invalid `{hparam}`")]
    MissingHParam { hparam: &'static str },

    /// A tensor name reached `target_for` without a parseable
    /// `blk.<N>.` layer index. Per `TensorRef::layer_index = None` is
    /// reserved for global tensors (token_embd / output / output_norm
    /// etc.); per-block tensors must carry `Some(i)`. The convert
    /// dispatcher upstream is responsible for parsing the layer index;
    /// this error fires if it forgot.
    #[error(
        "tensor `{name}` reached Apex policy without a layer_index; \
         expected `blk.<N>.*` but dispatcher passed layer_index=None"
    )]
    MissingLayerIndex { name: String },

    /// `layer_index >= n_layers`. Apex's per-layer table runs from
    /// `0..n_layers`; an out-of-range layer index is a bug at the
    /// dispatcher.
    #[error(
        "tensor `{name}` has layer_index={layer_index} but n_layers={n_layers}"
    )]
    LayerIndexOutOfRange {
        name: String,
        layer_index: usize,
        n_layers: u32,
    },

    /// ADR-033 §9: the manifest matched the source model's
    /// fingerprint, but the referenced `mudler_config_path` is not
    /// present in the compile-time vendor-content bake
    /// (`fingerprint::VENDOR_CONFIGS`). Hard error per the
    /// no-silent-fallback rule — a matched fingerprint MUST resolve
    /// to a usable config, else the operator's manifest is
    /// inconsistent with the baked vendor surface.
    ///
    /// In practice this fires only if a vendor regen updates the
    /// manifest JSON without also updating the `include_str!` table
    /// in `fingerprint.rs`. The unit test
    /// `every_manifest_entry_has_baked_vendor_content` catches it at
    /// build time.
    #[error(
        "ADR-033 §9: manifest matched fingerprint `{fingerprint}` → \
         `{mudler_config_path}`, but that config is not in the \
         compile-time vendor bake; add it to fingerprint::VENDOR_CONFIGS"
    )]
    FingerprintConfigMissing {
        fingerprint: String,
        mudler_config_path: String,
    },

    /// A mudler config file failed to parse — either a malformed
    /// line (missing `=`) or an unknown `GgmlType` token. Carries the
    /// source-relative line number for easy bisection.
    #[error(
        "mudler config parse error at {source_path}:{line_number}: {detail}"
    )]
    MudlerConfigParse {
        source_path: String,
        line_number: usize,
        detail: String,
    },

    /// ADR-033 §9 + the no-silent-fallback rule: a fingerprint match
    /// resolved to a mudler per-model config, but the tensor name
    /// hasn't been assigned a `GgmlType` in that config. We do NOT
    /// silently fall back to the algorithmic generator — per ADR
    /// "the vendored config's rules win over the algorithmic
    /// generator's output". A missing tensor surfaces as this typed
    /// error so the operator can fix the manifest (add the tensor) or
    /// switch off the per-model override.
    #[error(
        "tensor `{tensor_name}` not present in mudler config {source_path}; \
         per-model override is authoritative (no silent fallback to algorithmic generator)"
    )]
    TensorNotInMudlerConfig {
        source_path: String,
        tensor_name: String,
    },

    /// ADR-033 §Pi: an `apex-i-*` tier was requested but no per-row
    /// imatrix data was supplied. Per the no-silent-fallback rule we
    /// reject with a typed error rather than silently producing non-I
    /// bytes (the current `tier_rules` map identical TierRules for
    /// {Quality, IQuality}, {Balanced, IBalanced}, {Compact, ICompact}
    /// pairs — so running an I-tier without imatrix would emit bytes
    /// byte-identical to the non-I sibling, defeating the purpose of
    /// asking for I).
    ///
    /// `supported_for_imatrix` is the set of arches that can consume
    /// imatrix data for I-tier policy dispatch. Operators reaching
    /// this error should:
    ///   - run `hf2q convert ... --imatrix-corpus cdv3` (Stage 3.0
    ///     drives in-tree generation for Gemma 4); OR
    ///   - run `llama-imatrix -m <gguf> -f cdv3.txt -o <out>` and
    ///     pass `<out>` via `--imatrix <path>` (works on any
    ///     supported arch).
    #[error(
        "Apex tier `{tier}` requires per-row imatrix data (ADR-033 §Pi). \
         No imatrix was supplied for arch `{arch}`; \
         supported_for_imatrix arches: {supported_for_imatrix:?}. \
         Pass `--imatrix-corpus cdv3` (Gemma 4 in-tree driver) \
         OR `--imatrix <path>` (pre-computed `.imatrix.gguf` from \
         stock `llama-imatrix`)."
    )]
    ImatrixRequiresInference {
        tier: &'static str,
        arch: &'static str,
        supported_for_imatrix: &'static [&'static str],
    },
}

impl ApexError {
    /// Helper to build an `UnsupportedArch` from an `ArchName` with the
    /// canonical v1 supported list.
    pub fn unsupported_arch(arch: ArchName) -> Self {
        ApexError::UnsupportedArch {
            arch: arch.name(),
            supported: super::arches::SUPPORTED_APEX_ARCHES,
        }
    }
}

#[cfg(test)]
mod p7_ac3_hint_tests {
    //! ADR-033 §P7 AC#3 — every typed error's MESSAGE must contain
    //! the actionable hint the operator needs to recover. Variant-only
    //! matching via `matches!(...)` doesn't catch hint regressions; the
    //! tests below `.to_string()` each variant and substring-match the
    //! hint text.

    use super::*;
    use super::super::arches::SUPPORTED_APEX_ARCHES;

    #[test]
    fn p7_ac3_unsupported_arch_lists_supported() {
        let msg = ApexError::UnsupportedArch {
            arch: "falcon",
            supported: SUPPORTED_APEX_ARCHES,
        }
        .to_string();
        assert!(msg.contains("falcon"), "msg should echo bad arch: {msg}");
        assert!(
            msg.contains("gemma4") || msg.contains("qwen"),
            "msg should list at least one supported arch: {msg}"
        );
    }

    #[test]
    fn p7_ac3_dense_model_not_supported() {
        let msg = ApexError::DenseModelNotSupported {
            arch: "llama",
            n_expert: 1,
        }
        .to_string();
        assert!(msg.contains("MoE"), "msg should explain MoE requirement: {msg}");
        assert!(msg.contains("q5_k_m"), "msg should suggest dense fallback: {msg}");
    }

    #[test]
    fn p7_ac3_unsupported_tier_lists_supported() {
        let msg = ApexError::UnsupportedTier {
            tier: "nano".to_string(),
            supported: super::super::rules::SUPPORTED_APEX_TIERS,
        }
        .to_string();
        assert!(msg.contains("nano"), "msg should echo bad tier: {msg}");
        assert!(msg.contains("balanced"), "msg should list supported tiers: {msg}");
        assert!(
            msg.contains("apex-custom"),
            "msg should mention apex-custom escape hatch: {msg}"
        );
    }

    #[test]
    fn p7_ac3_custom_requires_tensor_type_file() {
        let msg = ApexError::CustomRequiresTensorTypeFile.to_string();
        assert!(
            msg.contains("apex-custom") || msg.contains("--tensor-type-file"),
            "msg should name the missing flag: {msg}"
        );
    }

    #[test]
    fn p7_ac3_missing_hparam_names_field() {
        let msg = ApexError::MissingHParam {
            hparam: "num_hidden_layers",
        }
        .to_string();
        assert!(
            msg.contains("num_hidden_layers"),
            "msg should name the missing field: {msg}"
        );
    }

    #[test]
    fn p7_ac3_missing_layer_index_names_tensor() {
        let msg = ApexError::MissingLayerIndex {
            name: "blk.5.attn_q.weight".to_string(),
        }
        .to_string();
        assert!(
            msg.contains("blk.5.attn_q.weight"),
            "msg should echo offending tensor name: {msg}"
        );
        assert!(
            msg.contains("layer_index"),
            "msg should reference the missing field: {msg}"
        );
    }

    #[test]
    fn p7_ac3_layer_index_out_of_range_carries_values() {
        let msg = ApexError::LayerIndexOutOfRange {
            name: "blk.40.attn_q.weight".to_string(),
            layer_index: 40,
            n_layers: 30,
        }
        .to_string();
        assert!(msg.contains("40"), "msg should include layer_index: {msg}");
        assert!(msg.contains("30"), "msg should include n_layers: {msg}");
        assert!(
            msg.contains("blk.40.attn_q.weight"),
            "msg should echo the tensor: {msg}"
        );
    }

    #[test]
    fn p7_ac3_fingerprint_config_missing_names_path() {
        let msg = ApexError::FingerprintConfigMissing {
            fingerprint: "deadbeef".to_string(),
            mudler_config_path: "vendor/apex-quant/configs/x.txt".to_string(),
        }
        .to_string();
        assert!(msg.contains("vendor/apex-quant/configs/x.txt"));
        assert!(
            msg.contains("VENDOR_CONFIGS"),
            "msg should point at the fix site: {msg}"
        );
    }

    #[test]
    fn p7_ac3_mudler_config_parse_locates() {
        let msg = ApexError::MudlerConfigParse {
            source_path: "foo.txt".to_string(),
            line_number: 42,
            detail: "missing `=` separator".to_string(),
        }
        .to_string();
        assert!(msg.contains("foo.txt:42"), "msg should locate at file:line: {msg}");
        assert!(msg.contains("missing `=`"), "msg should carry detail: {msg}");
    }

    #[test]
    fn p7_ac3_tensor_not_in_mudler_config_names_both() {
        let msg = ApexError::TensorNotInMudlerConfig {
            source_path: "x.txt".to_string(),
            tensor_name: "blk.0.attn_q.weight".to_string(),
        }
        .to_string();
        assert!(msg.contains("blk.0.attn_q.weight"));
        assert!(msg.contains("x.txt"));
        assert!(
            msg.contains("authoritative") || msg.contains("no silent fallback"),
            "msg should explain why this errors instead of fallback: {msg}"
        );
    }

    /// **Critical post-Stage-3c hint test**: the operator command in
    /// the error message MUST point at the now-shipped paths
    /// (`--imatrix-corpus cdv3` for Gemma 4 in-tree, or `--imatrix
    /// <path>` for pre-computed). Earlier message said "Pi is not yet
    /// shipped" which was stale after commit 9fd50afa.
    #[test]
    fn p7_ac3_imatrix_requires_inference_post_stage_3_hints() {
        let msg = ApexError::ImatrixRequiresInference {
            tier: "i-balanced",
            arch: "gemma4",
            supported_for_imatrix: &["qwen3moe", "gemma4"],
        }
        .to_string();
        assert!(msg.contains("i-balanced"), "msg should echo bad tier: {msg}");
        assert!(msg.contains("gemma4"), "msg should echo arch: {msg}");
        assert!(
            msg.contains("--imatrix-corpus") && msg.contains("cdv3"),
            "msg should advertise the in-tree driver path (Stage 3c): {msg}"
        );
        assert!(
            msg.contains("--imatrix") && msg.contains("llama-imatrix"),
            "msg should advertise the pre-computed file path: {msg}"
        );
        assert!(
            !msg.contains("not yet shipped") && !msg.contains("pending Pi"),
            "msg must NOT claim Pi is unshipped (post-Stage-3c regression check): {msg}"
        );
    }
}
