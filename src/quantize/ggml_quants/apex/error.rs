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
