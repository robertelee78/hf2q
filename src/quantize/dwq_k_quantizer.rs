//! `DwqKQuantizer` — per-tensor K-quant target dispatcher for the
//! DWQ-class variants (ADR-014 P11-prereq Iter B).
//!
//! ## Why this layer exists
//!
//! Today's DWQ shipping path (`src/calibrate/dwq.rs:267-272`) constructs
//! a `MixedBitQuantizer` with `preset = "mixed-{base}-{sensitive}"` and
//! the activation-derived `sensitive_layers: Vec<RangeInclusive<usize>>`.
//! `MixedBitQuantizer` then routes each tensor through `StaticQuantizer`
//! at the matching bit width (q4/q6/q8/q2) — the legacy IR-quantize
//! format with raw `i8` quants + separate scales/biases that the GGUF
//! backend later round-trips via `repack_q*_*` (the very dance ADR-014
//! is replacing).
//!
//! `DwqKQuantizer` is the algorithmic replacement: it preserves the
//! *exact same* per-tensor base-vs-sensitive dispatch policy
//! (`extract_layer_index(tensor_name) ∈ sensitive_indices`) but routes
//! through `KQuantCodecQuantizer` with a `KQuantTarget` chosen per
//! variant.  Output bytes are final GGUF block bytes carrying the
//! `METHOD_K_QUANT_CODEC_DIRECT` sentinel, which Iter A's consumer-side
//! fast-path in `gguf.rs` recognises and emits unchanged with the
//! correct K-quant header type code.
//!
//! ## Variants
//!
//! | Variant       | Base    | Sensitive |
//! |---------------|---------|-----------|
//! | `DwqKVariant::P46` | Q4_K | Q6_K |
//! | `DwqKVariant::P48` | Q4_K | Q8_0 |
//! | `DwqKVariant::P68` | Q6_K | Q8_0 |
//! | `DwqKVariant::P28` | Q2_K | Q8_0 |
//!
//! `P28` is the abliterated-apex 2-bit/8-bit DWQ preset.  `KQuantTarget`
//! does not yet expose a `Q2_K` variant (the Q2_K codec port is pending
//! a separate iter), so `P28` returns a typed
//! `QuantizeError::TensorQuantizeFailed` from `quantize_tensor` for
//! base-target tensors — **no panics, no silent fall-back**.  The error
//! message points the caller at the deferred Q2_K codec land.  Sensitive-
//! target tensors (Q8_0) still quantize correctly under `P28`; the
//! restriction only fires when the base target is requested.
//!
//! ## What we drop vs `MixedBitQuantizer` + DWQ scale-cal
//!
//! The legacy DWQ scale-cal step (`src/calibrate/dwq.rs:298-356`) walks
//! every quantized tensor post-hoc and recomputes the optimal
//! per-group scalar scale via the closed-form
//! `optimal_scale = dot(W, Q) / dot(Q, Q)`.  This is geometrically
//! incompatible with K-quant emit:
//!
//!   * Q4_K stores a per-super-block 16-bit `d` AND `dmin`, plus 8
//!     6-bit-packed sub-block scales/mins.  There is no scalar-per-group
//!     scale in the format that a closed-form re-fit could even target.
//!   * Q4_K's codebook search (`make_qkx2_quants` / `quantize_row_q4_K`)
//!     already optimises (sub-block scale, sub-block min) jointly via
//!     a search over candidate inverse-scales and per-element re-binning;
//!     a post-hoc scalar scale tweak would degrade that joint optimum.
//!
//! Iter C (the wiring iter) drops the scale-cal step from the DWQ
//! orchestrator when `DwqKQuantizer` is the active codec.  This iter
//! introduces the codec; it does not modify the orchestrator (file fence).
//!
//! ## Sensitivity → Imatrix
//!
//! `DwqConfig` today exposes only a *layer-scalar* sensitivity (consumed
//! upstream to derive `sensitive_layers: Vec<RangeInclusive<usize>>` —
//! see `src/calibrate/sensitivity.rs::LayerSensitivity.score`).  The
//! K-quant codec's `CalibrationData::Imatrix` payload, by contrast,
//! expects a *per-column* importance vector (`Vec<f32>` of length
//! `row_len`) that drives `quantize_row_q4_k_imatrix_to_bytes`'s
//! per-element-weighted codebook search.  These two shapes are
//! dimensionally incompatible — a scalar cannot be losslessly broadcast
//! into a per-column vector.
//!
//! Bridging requires either (a) a separate imatrix calibration step run
//! alongside DWQ that produces real per-column vectors, or (b) an upstream
//! decision to broadcast the scalar uniformly across all columns
//! (degenerates to `_ref` since uniform weights yield the same codebook
//! search as no weights).  Both are call-site decisions belonging to
//! Iter C.  This iter accepts an `Option<CalibrationData>` constructor
//! parameter so Iter C can wire (a) without churning the API; passing
//! `None` falls back to the `_ref` codebook search (the same default
//! `KQuantCodecQuantizer::new(... CalibrationData::None)` provides).

use std::collections::HashSet;
use std::ops::RangeInclusive;

use crate::calibrate::calibrator::CalibrationData;
use crate::ir::{QuantizedTensor, TensorRef};
use crate::quantize::k_quant_codec::KQuantTarget;
use crate::quantize::k_quant_codec_quantizer::{f16_passthrough, KQuantCodecQuantizer};
use crate::quantize::layer_mix::should_emit_f16_for_kquant;
use crate::quantize::{LayerQuantConfig, Quantizer, QuantizeError};

/// DWQ-class K-quant variant.  Maps to a (base, sensitive) pair of
/// [`KQuantTarget`]s.  Names mirror the existing dwq46/dwq48/dwq68/dwq28
/// dispatch surface in `cmd_convert`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DwqKVariant {
    /// `dwq46` — base `Q4_K`, sensitive `Q6_K`.
    P46,
    /// `dwq48` — base `Q4_K`, sensitive `Q8_0`.
    P48,
    /// `dwq68` — base `Q6_K`, sensitive `Q8_0`.
    P68,
    /// `dwq28` — base `Q2_K` (codec land pending), sensitive `Q8_0`.
    /// Sensitive-target tensors quantize correctly; base-target tensors
    /// surface a typed error pointing at the deferred Q2_K codec.
    P28,
}

impl DwqKVariant {
    /// The K-quant target for *non-sensitive* tensors under this variant.
    /// Returns `None` for `P28` (Q2_K codec deferred).
    pub fn base_target(&self) -> Option<KQuantTarget> {
        match self {
            Self::P46 | Self::P48 => Some(KQuantTarget::Q4K),
            Self::P68 => Some(KQuantTarget::Q6K),
            // Q2_K codec not yet available in `KQuantTarget`.  Kept as
            // `None` rather than a panicking sentinel so the
            // `Quantizer::quantize_tensor` impl can surface a typed error.
            Self::P28 => None,
        }
    }

    /// The K-quant target for *sensitive* tensors under this variant.
    pub fn sensitive_target(&self) -> KQuantTarget {
        match self {
            Self::P46 => KQuantTarget::Q6K,
            Self::P48 | Self::P68 | Self::P28 => KQuantTarget::Q8Legacy,
        }
    }

    /// Human-readable name used by the `Quantizer::name()` impl and by
    /// the eventual cmd_convert dispatch table (Iter C).
    pub fn name(&self) -> &'static str {
        match self {
            Self::P46 => "dwq-k-mixed-4-6",
            Self::P48 => "dwq-k-mixed-4-8",
            Self::P68 => "dwq-k-mixed-6-8",
            Self::P28 => "dwq-k-mixed-2-8",
        }
    }
}

/// Per-tensor K-quant target dispatcher for DWQ-class variants.
///
/// Mirrors `MixedBitQuantizer`'s base-vs-sensitive split exactly:
///   * `sensitive_layers: Vec<RangeInclusive<usize>>` is expanded into
///     a `HashSet<usize>` of layer indices at construction time.
///   * `extract_layer_index(tensor_name)` parses the canonical HF
///     `model.layers.<N>.…` and `model.language_model.layers.<N>.…`
///     patterns (re-using the helper from `mixed.rs` semantics —
///     re-implemented here to keep `mixed.rs` read-only per file fence).
///   * Tensors whose layer index is in the sensitive set route to the
///     variant's `sensitive_target`; all others route to `base_target`.
///   * Non-block tensors (`output.weight`, `token_embd.weight`, norms)
///     do not match the layers pattern and therefore fall through to
///     `base_target` — same behaviour as today's DWQ shipping path.
pub struct DwqKQuantizer {
    variant: DwqKVariant,
    sensitive_indices: HashSet<usize>,
    /// Optional calibration payload threaded into the codec.  See module
    /// doc "Sensitivity → Imatrix" — `None` is the supported value for
    /// this iter; Iter C lands the bridge.
    calibration: CalibrationData,
}

impl DwqKQuantizer {
    /// Construct.  `sensitive_layers` mirrors `DwqConfig.sensitive_layers`
    /// (`Vec<RangeInclusive<usize>>`); each range is expanded inclusively
    /// into the per-layer `HashSet<usize>` used at dispatch.
    ///
    /// `calibration`'s only currently-supported value is
    /// [`CalibrationData::None`] — see module doc for the bridging
    /// constraints.  The other variants (`Imatrix`, `ImatrixWithStats`,
    /// `Dwq`) are accepted by the API for forward-compat but documented
    /// as Iter-C call-site decisions.
    pub fn new(
        variant: DwqKVariant,
        sensitive_layers: &[RangeInclusive<usize>],
        calibration: CalibrationData,
    ) -> Self {
        let mut sensitive_indices = HashSet::new();
        for range in sensitive_layers {
            // Mirror `MixedBitQuantizer::new`: silently clamp inverted
            // ranges to empty so this constructor stays infallible —
            // upstream `MixedBitQuantizer::new` returns an error on
            // inverted ranges, but its callers (cmd_convert + DWQ
            // activation) construct ranges programmatically and never
            // invert, so the error path is untaken in practice.  Our
            // constructor's contract is "accept what DWQ produces"; the
            // upstream validation is preserved by Iter C calling sites.
            if range.start() <= range.end() {
                for idx in range.clone() {
                    sensitive_indices.insert(idx);
                }
            }
        }
        Self {
            variant,
            sensitive_indices,
            calibration,
        }
    }

    /// The variant this quantizer dispatches on.
    pub fn variant(&self) -> DwqKVariant {
        self.variant
    }

    /// Whether this tensor name parses to a layer index in the sensitive
    /// set.  Mirrors `MixedBitQuantizer::is_sensitive_tensor`.
    fn is_sensitive_tensor(&self, tensor_name: &str) -> bool {
        if self.sensitive_indices.is_empty() {
            return false;
        }
        extract_layer_index(tensor_name)
            .map(|idx| self.sensitive_indices.contains(&idx))
            .unwrap_or(false)
    }

    /// Pick the K-quant target for `tensor_name`.  Returns `None` only
    /// when the variant's base target is `P28`'s Q2_K (codec deferred)
    /// AND the tensor falls into the base bucket.
    fn target_for(&self, tensor_name: &str) -> Option<KQuantTarget> {
        if self.is_sensitive_tensor(tensor_name) {
            Some(self.variant.sensitive_target())
        } else {
            self.variant.base_target()
        }
    }
}

impl Quantizer for DwqKQuantizer {
    fn name(&self) -> &str {
        self.variant.name()
    }

    fn requires_calibration(&self) -> bool {
        // The DWQ activation-capture / sensitivity-derivation step
        // happens upstream, before `DwqKQuantizer::new` is invoked.
        // From the codec's POV this quantizer needs no further forward
        // pass at quantize_tensor time.  Mirrors `KQuantCodecQuantizer`.
        false
    }

    fn quantize_tensor(
        &self,
        tensor: &TensorRef,
        config: &LayerQuantConfig,
    ) -> Result<QuantizedTensor, QuantizeError> {
        // ADR-014 P11-prereq Iter D (2026-04-27): vision-tensor +
        // non-256-multiple skip → F16 passthrough.  The check fires
        // BEFORE per-tensor target dispatch so DWQ's base/sensitive
        // routing never sees a tensor the K-quant codec cannot encode
        // (e.g. Qwen3.6-27B vision-attn proj at row_len = 1152, the
        // LIVE blocker for the four DWQ re-emits documented in
        // ADR-014 § "P11-prereq Iter D" 2026-04-27).  Plumbed here
        // (not just at the inner `KQuantCodecQuantizer` site) so the
        // skip signal is logged with the DWQ variant context AND so a
        // future refactor that swaps the inner codec cannot regress
        // the predicate on the DWQ path.
        //
        // **Ordering note**: `config.preserve` wins (handled by the
        // inner `KQuantCodecQuantizer` at delegation time).  Iter D
        // skip fires only when the caller did NOT opt into preserve
        // — symmetric with the in-source ordering at
        // `KQuantCodecQuantizer::quantize_tensor` and
        // `VariantKQuantizer::quantize_tensor`.
        let row_len = tensor.shape.last().copied().unwrap_or(0);
        if !config.preserve && should_emit_f16_for_kquant(&tensor.name, row_len) {
            tracing::info!(
                tensor = %tensor.name,
                row_len,
                variant = self.variant.name(),
                "DWQ K-quant skip → F16 passthrough (vision or non-256-multiple)"
            );
            return f16_passthrough(tensor, &tensor.name);
        }

        let target = match self.target_for(&tensor.name) {
            Some(t) => t,
            None => {
                // Only reachable for `DwqKVariant::P28` base bucket
                // until the Q2_K codec lands in `KQuantTarget`.  Surface
                // a typed error pointing at the deferred codec land —
                // never panic, never silently degrade to a different
                // target.
                return Err(QuantizeError::TensorQuantizeFailed {
                    tensor: tensor.name.clone(),
                    reason: format!(
                        "dwq-k-quantizer ({}): base target Q2_K is not yet \
                         available in KQuantTarget — codec land pending. \
                         Sensitive-layer tensors still quantize to Q8_0; \
                         base-layer tensors block on the Q2_K codec port.",
                        self.variant.name(),
                    ),
                });
            }
        };

        // Delegate to the codec-direct quantizer.  Constructing per-call
        // is cheap (pure data move; no allocations beyond the cached
        // calibration ref) and keeps the per-tensor target dispatch
        // localised here.  The codec handles preserve / passthrough,
        // F32/F16/BF16 input conversion, row-major 2D layout, and
        // sets the `METHOD_K_QUANT_CODEC_DIRECT` sentinel that Iter A's
        // gguf.rs fast-path recognises.
        let codec_name = format!("{}-{}", self.variant.name(), kquant_target_short_name(target));
        let codec = KQuantCodecQuantizer::new(codec_name, target, self.calibration.clone());
        codec.quantize_tensor(tensor, config)
    }
}

/// Short name of a `KQuantTarget` for use in derived codec names.
/// Kept here (rather than re-using `target_to_ggml_name` which is
/// module-private to `k_quant_codec_quantizer.rs`) to honour the
/// read-only fence on that file.
fn kquant_target_short_name(target: KQuantTarget) -> &'static str {
    match target {
        KQuantTarget::Q4K => "q4_k",
        KQuantTarget::Q5K => "q5_k",
        KQuantTarget::Q6K => "q6_k",
        KQuantTarget::Q4Legacy => "q4_0",
        KQuantTarget::Q4Legacy1 => "q4_1",
        KQuantTarget::Q5Legacy0 => "q5_0",
        KQuantTarget::Q5Legacy1 => "q5_1",
        KQuantTarget::Q8Legacy => "q8_0",
    }
}

/// Extract the layer index from a tensor name following the canonical
/// HF `model.layers.<N>.…` (or `model.language_model.layers.<N>.…`)
/// pattern.  Returns `None` for non-layer tensors (output, embeddings,
/// norms without a layer prefix).
///
/// Re-implementation of the same helper in `mixed.rs::extract_layer_index`
/// (read-only per file fence).  Same semantics — exercised by
/// `tests/dwq_k_quantizer.rs::layer_index_dispatch_matches_mixed_quantizer`.
fn extract_layer_index(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "layers" {
            if let Some(num_str) = parts.get(i + 1) {
                return num_str.parse::<usize>().ok();
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::DType;
    use crate::quantize::k_quant::BLOCK_Q4_K_SIZE;
    use crate::quantize::k_quant_codec_quantizer::METHOD_K_QUANT_CODEC_DIRECT;

    /// Single Q4_K super-block size.
    const QK_K: usize = 256;

    fn make_block_aligned_f32_tensor(name: &str) -> TensorRef {
        let values: Vec<f32> = (0..QK_K)
            .map(|i| (i as f32 - (QK_K as f32 / 2.0)) / (QK_K as f32))
            .collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        TensorRef {
            name: name.to_string(),
            shape: vec![QK_K],
            dtype: DType::F32,
            data,
        }
    }

    fn default_layer_config() -> LayerQuantConfig {
        // bits + group_size are sentinels (the codec ignores them —
        // block layout is implicit in the target).  preserve = false
        // routes through the codec rather than the passthrough fast-path.
        LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: false,
        }
    }

    #[test]
    fn extract_layer_index_canonical() {
        assert_eq!(
            extract_layer_index("model.layers.13.self_attn.q_proj.weight"),
            Some(13)
        );
        assert_eq!(
            extract_layer_index("model.language_model.layers.5.mlp.down_proj.weight"),
            Some(5)
        );
        assert_eq!(extract_layer_index("model.embed_tokens.weight"), None);
        assert_eq!(extract_layer_index("lm_head.weight"), None);
    }

    #[test]
    fn variant_targets_p46() {
        assert_eq!(DwqKVariant::P46.base_target(), Some(KQuantTarget::Q4K));
        assert_eq!(DwqKVariant::P46.sensitive_target(), KQuantTarget::Q6K);
    }

    #[test]
    fn variant_targets_p48() {
        assert_eq!(DwqKVariant::P48.base_target(), Some(KQuantTarget::Q4K));
        assert_eq!(DwqKVariant::P48.sensitive_target(), KQuantTarget::Q8Legacy);
    }

    #[test]
    fn variant_targets_p68() {
        assert_eq!(DwqKVariant::P68.base_target(), Some(KQuantTarget::Q6K));
        assert_eq!(DwqKVariant::P68.sensitive_target(), KQuantTarget::Q8Legacy);
    }

    #[test]
    fn variant_targets_p28_base_unavailable() {
        // Q2_K codec deferred; base target is None until it lands.
        assert_eq!(DwqKVariant::P28.base_target(), None);
        // Sensitive target is Q8_0 — already supported by KQuantTarget.
        assert_eq!(DwqKVariant::P28.sensitive_target(), KQuantTarget::Q8Legacy);
    }

    #[test]
    fn variant_names_match_dwq_naming() {
        assert_eq!(DwqKVariant::P46.name(), "dwq-k-mixed-4-6");
        assert_eq!(DwqKVariant::P48.name(), "dwq-k-mixed-4-8");
        assert_eq!(DwqKVariant::P68.name(), "dwq-k-mixed-6-8");
        assert_eq!(DwqKVariant::P28.name(), "dwq-k-mixed-2-8");
    }

    #[test]
    fn quantizer_name_introspection() {
        let q = DwqKQuantizer::new(DwqKVariant::P46, &[], CalibrationData::None);
        assert_eq!(q.name(), "dwq-k-mixed-4-6");
        assert!(!q.requires_calibration());
        assert_eq!(q.variant(), DwqKVariant::P46);
    }

    #[test]
    fn sensitive_layers_expansion_inclusive_ranges() {
        let q = DwqKQuantizer::new(
            DwqKVariant::P46,
            &[5..=7, 12..=12],
            CalibrationData::None,
        );
        // Sanity: layers 5, 6, 7, 12 → sensitive; 4, 8, 11, 13 → base.
        assert!(q.is_sensitive_tensor("model.layers.5.self_attn.q_proj.weight"));
        assert!(q.is_sensitive_tensor("model.layers.6.self_attn.q_proj.weight"));
        assert!(q.is_sensitive_tensor("model.layers.7.self_attn.q_proj.weight"));
        assert!(q.is_sensitive_tensor("model.layers.12.mlp.down_proj.weight"));
        assert!(!q.is_sensitive_tensor("model.layers.4.self_attn.q_proj.weight"));
        assert!(!q.is_sensitive_tensor("model.layers.8.self_attn.q_proj.weight"));
        assert!(!q.is_sensitive_tensor("model.layers.11.mlp.down_proj.weight"));
        assert!(!q.is_sensitive_tensor("model.layers.13.mlp.down_proj.weight"));
        // Non-layer tensors never match.
        assert!(!q.is_sensitive_tensor("model.embed_tokens.weight"));
        assert!(!q.is_sensitive_tensor("lm_head.weight"));
    }

    #[test]
    fn empty_sensitive_layers_routes_everything_to_base() {
        let q = DwqKQuantizer::new(DwqKVariant::P46, &[], CalibrationData::None);
        assert!(!q.is_sensitive_tensor("model.layers.0.self_attn.q_proj.weight"));
        assert!(!q.is_sensitive_tensor("model.layers.99.mlp.down_proj.weight"));
        // target_for returns base target for everything.
        assert_eq!(
            q.target_for("model.layers.0.self_attn.q_proj.weight"),
            Some(KQuantTarget::Q4K)
        );
    }

    #[test]
    fn inverted_range_silently_dropped() {
        // Spec: constructor mirrors MixedBitQuantizer's leniency on
        // inverted ranges — drop them rather than refuse construction.
        let q = DwqKQuantizer::new(
            DwqKVariant::P46,
            &[10..=5], // inverted
            CalibrationData::None,
        );
        // No layers ended up sensitive.
        assert!(!q.is_sensitive_tensor("model.layers.5.self_attn.q_proj.weight"));
        assert!(!q.is_sensitive_tensor("model.layers.10.self_attn.q_proj.weight"));
    }

    #[test]
    fn target_for_p28_base_layer_returns_none() {
        // P28 + base bucket → None (caller surfaces typed error).
        let q = DwqKQuantizer::new(DwqKVariant::P28, &[3..=3], CalibrationData::None);
        assert_eq!(
            q.target_for("model.layers.0.self_attn.q_proj.weight"),
            None
        );
        // Sensitive layer still routes to Q8_0.
        assert_eq!(
            q.target_for("model.layers.3.self_attn.q_proj.weight"),
            Some(KQuantTarget::Q8Legacy)
        );
    }

    // ─── End-to-end `quantize_tensor` tests (Iter B integration surface) ───

    #[test]
    fn end_to_end_p46_base_layer_routes_to_q4_k_bytes() {
        // P46 base = Q4_K. Empty sensitive_layers → every layer hits base.
        // Drive the full Quantizer trait surface end-to-end and lock both
        // metadata + byte-format contracts.
        let q = DwqKQuantizer::new(DwqKVariant::P46, &[], CalibrationData::None);
        let tensor =
            make_block_aligned_f32_tensor("model.layers.0.self_attn.q_proj.weight");
        let out = q.quantize_tensor(&tensor, &default_layer_config()).unwrap();

        assert_eq!(out.quant_info.method, METHOD_K_QUANT_CODEC_DIRECT);
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q4_K"));
        assert_eq!(out.data.len(), KQuantTarget::Q4K.bytes_per_block());
        assert_eq!(out.data.len(), BLOCK_Q4_K_SIZE); // cross-check raw constant
        assert!(!out.quant_info.preserved);
        assert!(out.quant_info.scales.is_none());
        assert!(out.quant_info.biases.is_none());
        assert_eq!(out.quant_info.bits, 0); // sentinel from codec
        assert_eq!(out.quant_info.group_size, 0); // sentinel from codec
    }

    #[test]
    fn end_to_end_p46_sensitive_layer_routes_to_q6_k_bytes() {
        // P46 sensitive = Q6_K.  Layer 7 marked sensitive.
        let q = DwqKQuantizer::new(
            DwqKVariant::P46,
            &[7..=7],
            CalibrationData::None,
        );
        let tensor =
            make_block_aligned_f32_tensor("model.layers.7.self_attn.v_proj.weight");
        let out = q.quantize_tensor(&tensor, &default_layer_config()).unwrap();

        assert_eq!(out.quant_info.method, METHOD_K_QUANT_CODEC_DIRECT);
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q6_K"));
        assert_eq!(out.data.len(), KQuantTarget::Q6K.bytes_per_block());
    }

    #[test]
    fn end_to_end_p48_sensitive_routes_to_q8_0_bytes() {
        let q = DwqKQuantizer::new(
            DwqKVariant::P48,
            &[3..=3],
            CalibrationData::None,
        );
        let tensor =
            make_block_aligned_f32_tensor("model.layers.3.mlp.down_proj.weight");
        let out = q.quantize_tensor(&tensor, &default_layer_config()).unwrap();

        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q8_0"));
        // Q8_0 = 32-elem-per-block legacy.  256 / 32 = 8 blocks per row.
        let blocks = QK_K / KQuantTarget::Q8Legacy.elements_per_block();
        assert_eq!(
            out.data.len(),
            blocks * KQuantTarget::Q8Legacy.bytes_per_block()
        );
    }

    #[test]
    fn end_to_end_p48_base_routes_to_q4_k_bytes() {
        let q = DwqKQuantizer::new(
            DwqKVariant::P48,
            &[3..=3],
            CalibrationData::None,
        );
        // Layer 4 NOT in sensitive set → base = Q4_K.
        let tensor =
            make_block_aligned_f32_tensor("model.layers.4.self_attn.q_proj.weight");
        let out = q.quantize_tensor(&tensor, &default_layer_config()).unwrap();
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q4_K"));
        assert_eq!(out.data.len(), KQuantTarget::Q4K.bytes_per_block());
    }

    #[test]
    fn end_to_end_p68_base_routes_to_q6_k_bytes() {
        let q = DwqKQuantizer::new(DwqKVariant::P68, &[], CalibrationData::None);
        let tensor =
            make_block_aligned_f32_tensor("model.layers.2.self_attn.q_proj.weight");
        let out = q.quantize_tensor(&tensor, &default_layer_config()).unwrap();
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q6_K"));
        assert_eq!(out.data.len(), KQuantTarget::Q6K.bytes_per_block());
    }

    #[test]
    fn end_to_end_p68_sensitive_routes_to_q8_0_bytes() {
        let q = DwqKQuantizer::new(
            DwqKVariant::P68,
            &[5..=5],
            CalibrationData::None,
        );
        let tensor =
            make_block_aligned_f32_tensor("model.layers.5.self_attn.v_proj.weight");
        let out = q.quantize_tensor(&tensor, &default_layer_config()).unwrap();
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q8_0"));
    }

    #[test]
    fn end_to_end_p28_base_returns_typed_error_no_panic() {
        let q = DwqKQuantizer::new(
            DwqKVariant::P28,
            &[3..=3],
            CalibrationData::None,
        );
        // Layer 0 NOT in sensitive set → base = Q2_K (deferred) → typed error.
        let tensor =
            make_block_aligned_f32_tensor("model.layers.0.self_attn.q_proj.weight");
        let err = q
            .quantize_tensor(&tensor, &default_layer_config())
            .unwrap_err();
        match err {
            QuantizeError::TensorQuantizeFailed { tensor: name, reason } => {
                assert_eq!(name, "model.layers.0.self_attn.q_proj.weight");
                assert!(
                    reason.contains("Q2_K") && reason.contains("dwq-k-quantizer"),
                    "expected error to mention Q2_K + dwq-k-quantizer, got: {reason}"
                );
                assert!(
                    reason.contains("codec land pending"),
                    "expected error to point at deferred codec land, got: {reason}"
                );
            }
            other => panic!("expected TensorQuantizeFailed, got {other:?}"),
        }
    }

    #[test]
    fn end_to_end_p28_sensitive_still_quantizes_to_q8_0_bytes() {
        // P28's sensitive bucket is Q8_0 (legacy, codec-supported); only
        // the base bucket blocks on Q2_K codec land.  Sensitive-tensor
        // emit must still work end-to-end — the typed error MUST be
        // narrowly scoped to base-target tensors.
        let q = DwqKQuantizer::new(
            DwqKVariant::P28,
            &[3..=3],
            CalibrationData::None,
        );
        let tensor =
            make_block_aligned_f32_tensor("model.layers.3.self_attn.v_proj.weight");
        let out = q.quantize_tensor(&tensor, &default_layer_config()).unwrap();
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q8_0"));
        assert_eq!(out.quant_info.method, METHOD_K_QUANT_CODEC_DIRECT);
    }

    #[test]
    fn end_to_end_byte_format_is_k_quant_super_block_size() {
        // Cross-check the codec emits exactly BLOCK_Q4_K_SIZE bytes for
        // one super-block under P46 base (catches drift between
        // KQuantTarget::Q4K.bytes_per_block() and the raw constant).
        let q = DwqKQuantizer::new(DwqKVariant::P46, &[], CalibrationData::None);
        let tensor =
            make_block_aligned_f32_tensor("model.layers.0.self_attn.q_proj.weight");
        let out = q.quantize_tensor(&tensor, &default_layer_config()).unwrap();
        assert_eq!(out.data.len(), BLOCK_Q4_K_SIZE);
        assert_eq!(BLOCK_Q4_K_SIZE, 144);
        assert_eq!(BLOCK_Q4_K_SIZE, KQuantTarget::Q4K.bytes_per_block());
    }

    #[test]
    fn end_to_end_non_layer_tensor_falls_through_to_base() {
        // `output.weight` does not match `model.layers.<N>.…` → falls
        // through to base bucket — even when "every" plausible layer is
        // marked sensitive.  Mirrors today's MixedBitQuantizer + DWQ
        // shipping behaviour (Chesterton's fence: contract Iter C wires).
        let q = DwqKQuantizer::new(
            DwqKVariant::P46,
            &[0..=99],
            CalibrationData::None,
        );
        let tensor = make_block_aligned_f32_tensor("output.weight");
        let out = q.quantize_tensor(&tensor, &default_layer_config()).unwrap();
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q4_K"));
    }

    #[test]
    fn end_to_end_preserve_path_returns_passthrough() {
        // `preserve = true` short-circuits inside `KQuantCodecQuantizer`.
        // Locks the per-tensor preserve-flag contract through the
        // DwqKQuantizer wrapper (norms, biases, vision tensors all hit
        // this path in production).
        let q = DwqKQuantizer::new(DwqKVariant::P46, &[], CalibrationData::None);
        let tensor = TensorRef {
            name: "model.layers.0.input_layernorm.weight".to_string(),
            shape: vec![16],
            dtype: DType::F32,
            data: vec![0u8; 16 * 4],
        };
        let cfg = LayerQuantConfig {
            bits: 0,
            group_size: 0,
            preserve: true,
        };
        let out = q.quantize_tensor(&tensor, &cfg).unwrap();
        assert_eq!(out.quant_info.method, "passthrough");
        assert!(out.quant_info.preserved);
        assert!(out.quant_info.ggml_type.is_none());
        assert_eq!(out.data, tensor.data);
    }

    #[test]
    fn end_to_end_layer_index_dispatch_matches_mixed_quantizer_semantics() {
        // Cross-check against `MixedBitQuantizer::extract_layer_index`
        // semantics (the contract this Iter is replacing).  Verifies the
        // re-implemented helper in this module didn't drift from the
        // canonical version in `mixed.rs`.
        let q = DwqKQuantizer::new(
            DwqKVariant::P46,
            &[5..=5, 13..=13],
            CalibrationData::None,
        );
        let cfg = default_layer_config();

        // Canonical: model.layers.<N> matched.
        let t = make_block_aligned_f32_tensor("model.layers.5.self_attn.q_proj.weight");
        assert_eq!(
            q.quantize_tensor(&t, &cfg).unwrap().quant_info.ggml_type.as_deref(),
            Some("Q6_K")
        );
        // language_model. prefix variant.
        let t = make_block_aligned_f32_tensor(
            "model.language_model.layers.5.mlp.down_proj.weight",
        );
        assert_eq!(
            q.quantize_tensor(&t, &cfg).unwrap().quant_info.ggml_type.as_deref(),
            Some("Q6_K")
        );
        // Outside sensitive set.
        let t = make_block_aligned_f32_tensor("model.layers.4.self_attn.q_proj.weight");
        assert_eq!(
            q.quantize_tensor(&t, &cfg).unwrap().quant_info.ggml_type.as_deref(),
            Some("Q4_K")
        );
        // No layer prefix → base.
        let t = make_block_aligned_f32_tensor("lm_head.weight");
        assert_eq!(
            q.quantize_tensor(&t, &cfg).unwrap().quant_info.ggml_type.as_deref(),
            Some("Q4_K")
        );
    }

    #[test]
    fn end_to_end_2d_tensor_emits_per_row_bytes() {
        // Multi-row weight matrix [4, 256] → 4 × 144 = 576 bytes.
        // Locks the row-iteration semantics inherited from the codec.
        let n_rows = 4;
        let mut values = Vec::with_capacity(n_rows * QK_K);
        for r in 0..n_rows {
            for j in 0..QK_K {
                values.push(((r * QK_K + j) as f32 - 512.0) / 512.0);
            }
        }
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensor = TensorRef {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            shape: vec![n_rows, QK_K],
            dtype: DType::F32,
            data,
        };

        let q = DwqKQuantizer::new(DwqKVariant::P46, &[], CalibrationData::None);
        let out = q.quantize_tensor(&tensor, &default_layer_config()).unwrap();
        assert_eq!(out.shape, vec![n_rows, QK_K]);
        assert_eq!(out.data.len(), n_rows * BLOCK_Q4_K_SIZE);
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q4_K"));
    }

    /// **Iter D**: vision-tensor blocker shape (1152-dim
    /// `model.visual.…`) emits F16 passthrough through the DWQ
    /// dispatcher — pre-empts the codec's row-alignment rejection.
    /// Matches the LIVE blocker that killed the first P11 re-emit
    /// attempt at `commit base 87c6242` (qwen3.6-27B → dwq46).
    #[test]
    fn dwq_p46_vision_tensor_blocker_shape_emits_f16() {
        let row: Vec<f32> = (0..1152).map(|i| (i as f32 - 576.0) / 576.0).collect();
        let data: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensor = TensorRef {
            name: "model.visual.blocks.0.attn.proj.weight".to_string(),
            shape: vec![1152],
            dtype: DType::F32,
            data,
        };
        let q = DwqKQuantizer::new(DwqKVariant::P46, &[], CalibrationData::None);
        let out = q.quantize_tensor(&tensor, &default_layer_config()).unwrap();
        assert_eq!(out.quant_info.method, "f16");
        assert_eq!(out.quant_info.bits, 16);
        assert!(out.quant_info.preserved);
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("F16"));
        assert_eq!(out.data.len(), 1152 * 2);
    }

    /// **Iter D**: same vision tensor under DWQ-P28 (the variant whose
    /// base bucket would otherwise return the typed Q2_K-deferred
    /// error).  The Iter D skip predicate fires BEFORE the
    /// base-target dispatch, so the path is F16 passthrough — NOT
    /// the typed Q2_K error.  Locks the ordering: vision skip runs
    /// before per-variant target lookup.
    #[test]
    fn dwq_p28_vision_tensor_emits_f16_not_q2k_deferred_error() {
        let row: Vec<f32> = (0..1152).map(|i| (i as f32) / 1152.0).collect();
        let data: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensor = TensorRef {
            name: "model.visual.blocks.0.attn.proj.weight".to_string(),
            shape: vec![1152],
            dtype: DType::F32,
            data,
        };
        let q = DwqKQuantizer::new(DwqKVariant::P28, &[], CalibrationData::None);
        let out = q.quantize_tensor(&tensor, &default_layer_config()).unwrap();
        assert_eq!(out.quant_info.method, "f16");
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("F16"));
    }

    /// **Iter D**: aligned non-vision tensor still routes through the
    /// codec under DWQ — proves the skip is narrowly scoped.
    #[test]
    fn dwq_p46_aligned_language_tensor_still_routes_through_codec() {
        let tensor = make_block_aligned_f32_tensor(
            "model.layers.0.self_attn.q_proj.weight",
        );
        let q = DwqKQuantizer::new(DwqKVariant::P46, &[], CalibrationData::None);
        let out = q.quantize_tensor(&tensor, &default_layer_config()).unwrap();
        assert_eq!(out.quant_info.method, METHOD_K_QUANT_CODEC_DIRECT);
        assert_eq!(out.quant_info.ggml_type.as_deref(), Some("Q4_K"));
    }

    /// **Iter D**: defensive non-256-multiple arm fires for a
    /// synthetic non-vision name on the DWQ path (independent of the
    /// vision substring match).
    #[test]
    fn dwq_p46_non_256_multiple_emits_f16() {
        let row: Vec<f32> = (0..1153).map(|i| (i as f32) / 1153.0).collect();
        let data: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensor = TensorRef {
            name: "model.layers.0.attn_weird.weight".to_string(),
            shape: vec![1153],
            dtype: DType::F32,
            data,
        };
        let q = DwqKQuantizer::new(DwqKVariant::P46, &[], CalibrationData::None);
        let out = q.quantize_tensor(&tensor, &default_layer_config()).unwrap();
        assert_eq!(out.quant_info.method, "f16");
        assert_eq!(out.data.len(), 1153 * 2);
    }

    #[test]
    fn kquant_target_short_name_covers_all() {
        assert_eq!(kquant_target_short_name(KQuantTarget::Q4K), "q4_k");
        assert_eq!(kquant_target_short_name(KQuantTarget::Q5K), "q5_k");
        assert_eq!(kquant_target_short_name(KQuantTarget::Q6K), "q6_k");
        assert_eq!(kquant_target_short_name(KQuantTarget::Q4Legacy), "q4_0");
        assert_eq!(kquant_target_short_name(KQuantTarget::Q4Legacy1), "q4_1");
        assert_eq!(kquant_target_short_name(KQuantTarget::Q5Legacy0), "q5_0");
        assert_eq!(kquant_target_short_name(KQuantTarget::Q5Legacy1), "q5_1");
        assert_eq!(kquant_target_short_name(KQuantTarget::Q8Legacy), "q8_0");
    }
}
