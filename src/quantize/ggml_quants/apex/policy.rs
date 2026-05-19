//! `ApexPolicy` — pure-Rust port of `mudler/apex-quant`'s per-tier
//! tensor-type-file rules, dispatched through the same
//! `QuantPolicy::target_for(TensorRef) -> Result<GgmlType, _>`
//! contract as `StandardPolicy`.
//!
//! Per ADR-033 §"Plan" / Pa. Vendored mudler reference:
//! `/opt/hf2q/vendor/apex-quant/` @ pinned SHA
//! `63c5048b7dc9ff230f2397d7bc445ca28894b769`.
//!
//! Resolution order (per ADR Decision §3 + §9):
//!
//! 1. **Vision/audio gate** — handled UPSTREAM at the convert
//!    dispatcher (`is_vision_tensor_pattern` / `is_audio_tensor_pattern`
//!    in `src/quantize/ggml_quants/vision.rs`). `ApexPolicy::target_for`
//!    is **not called** for those tensors; modality-side weights emit
//!    F16 outside the policy.
//!
//! 2. **Fingerprint match** (Decision §9) — performed UPSTREAM by the
//!    Apex driver against `data/apex-references/manifest.json`. If a
//!    per-model override matched, `target_for` is bypassed for that
//!    tensor in favor of the vendored config's verbatim assignment.
//!    `ApexPolicy::target_for` is the **algorithmic-generator path**
//!    only. (Per-model override dispatch lives in P4a's CLI driver,
//!    not in this file — Pa is rules + classifier scaffolding only.)
//!
//! 3. **Role classification** — `classify_moe_tensor(arch, name)`
//!    returns a `MoeTensorRole`.
//!
//! 4. **Layer-region partition** — `exp_region / shared_region /
//!    attn_region` for routed/shared/attention tensors respectively.
//!    Asymmetric: EXP/SHARED use a 5-wide edge band; ATTN uses 3-wide.
//!
//! 5. **Picker** — `tier_rules(tier)` returns the 7-tuple; we pick
//!    the slot indexed by `(role, region)`. Globals (`token_embd`,
//!    `output`) and structural tensors (`Norm`, `RouterGate`) have
//!    hard-coded picks not in the per-tier table.
//!
//! Per [[feedback-no-loop-suppression-2026-05-17]]: unsupported
//! arch / tier / dense model / missing hparam → typed `Err`. Never
//! silent F16 demotion.

use super::super::ggml_type::GgmlType;
use super::super::tensor_ref::{ArchName, TensorRef};
use super::arches::{classify_moe_tensor, is_apex_supported_arch, MoeTensorRole};
use super::error::ApexError;
use super::mudler_config::MudlerConfig;
use super::rules::{
    attn_region, exp_region, shared_region, tier_rules, ApexTier, AttnRegion, ExpRegion,
    SharedRegion,
};

/// `ApexPolicy` — per-tier algorithmic quant-target picker for MoE
/// arches. Constructed once per `convert` invocation; consumed by the
/// per-tensor dispatch loop.
///
/// Construction validates that:
///   - `arch` is in the v1 Apex supported set
///     (`Qwen35Moe / Gemma4 / MiniMaxM2`)
///   - `n_expert > 1` (dense models error early; Apex is MoE-only
///     by design)
///   - `n_layers > 0`
///
/// Post-construction, `target_for` is infallible w.r.t. these
/// preconditions; the only runtime errors are `MissingLayerIndex` /
/// `LayerIndexOutOfRange` (dispatcher bugs).
///
/// ADR §9 per-model override: when the CLI driver's fingerprint
/// dispatch matches a manifest entry, it wraps the matched
/// `MudlerConfig` via [`ApexPolicy::with_mudler_override`]. The
/// override path lifts every `target_for` query through the vendored
/// per-tensor map and surfaces missing tensors as
/// [`ApexError::TensorNotInMudlerConfig`] — no silent fall-through to
/// the algorithmic generator (ADR §9 line 102: "the vendored config's
/// rules win over the algorithmic generator's output").
#[derive(Debug, Clone, Copy)]
pub struct ApexPolicy {
    /// The selected Apex tier.
    pub tier: ApexTier,
    /// Source model's `config.json::num_hidden_layers` (auto-detected
    /// at convert time; no env-var override per ADR §"Per-model APEX
    /// config override").
    pub n_layers: u32,
    /// Source model's `config.json::num_experts` (Qwen35Moe naming) /
    /// `num_local_experts` (Mixtral naming) / equivalent per-arch
    /// field. Used at construction to gate the dense-model error;
    /// `target_for` doesn't re-check it.
    pub n_expert: u32,
    /// Arch for tensor classification.
    pub arch: ArchName,
    /// ADR §9 per-model override. When `Some`, `target_for` consults
    /// the vendored per-tensor map first; algorithmic generator is
    /// bypassed entirely. `'static` because the cache lives in the
    /// process-wide `mudler_config::cache_slot`. Defaults to `None`
    /// (algorithmic).
    pub mudler_override: Option<&'static MudlerConfig>,
}

impl ApexPolicy {
    /// Construct an `ApexPolicy`, validating preconditions up-front.
    /// Errors:
    ///   - `UnsupportedArch` if the arch isn't in the Apex v1 set.
    ///   - `DenseModelNotSupported` if `n_expert <= 1` (Apex is
    ///     MoE-only).
    ///   - `MissingHParam` if `n_layers == 0`.
    pub fn new(
        tier: ApexTier,
        arch: ArchName,
        n_layers: u32,
        n_expert: u32,
    ) -> Result<Self, ApexError> {
        if !is_apex_supported_arch(arch) {
            return Err(ApexError::unsupported_arch(arch));
        }
        if n_layers == 0 {
            return Err(ApexError::MissingHParam {
                hparam: "num_hidden_layers",
            });
        }
        if n_expert <= 1 {
            return Err(ApexError::DenseModelNotSupported {
                arch: arch.name(),
                n_expert,
            });
        }
        // ADR-033 §Pi: I-tier variants require per-row imatrix data.
        // v1's `tier_rules` maps {Quality,IQuality}, {Balanced,IBalanced},
        // {Compact,ICompact} to identical TierRules — so absent imatrix
        // data the I-tier silently produces bytes byte-identical to its
        // non-I sibling, defeating the operator's intent. Reject upfront
        // until Pi ships an inference-side imatrix driver.
        // SUPPORTED_FOR_IMATRIX is empty in v1; once Pi lands for a given
        // arch, append the arch's `name()` slot here.
        if tier.requires_imatrix() {
            const SUPPORTED_FOR_IMATRIX: &[&str] = &[];
            return Err(ApexError::ImatrixRequiresInference {
                tier: tier.cli_name(),
                arch: arch.name(),
                supported_for_imatrix: SUPPORTED_FOR_IMATRIX,
            });
        }
        Ok(Self {
            tier,
            n_layers,
            n_expert,
            arch,
            mudler_override: None,
        })
    }

    /// Attach an ADR §9 mudler per-model override to the policy.
    ///
    /// The override wins over `target_for`'s algorithmic generator for
    /// every tensor name present in the parsed [`MudlerConfig`]; a
    /// missing tensor surfaces as
    /// [`ApexError::TensorNotInMudlerConfig`] rather than fall
    /// through. Caller (CLI driver) is responsible for logging the
    /// match for debug transparency — see ADR §9 line 104's "surprise
    /// risk" mitigation.
    pub fn with_mudler_override(mut self, mudler: &'static MudlerConfig) -> Self {
        self.mudler_override = Some(mudler);
        self
    }

    /// Decide the disk-format `GgmlType` for one tensor.
    ///
    /// Resolution order per ADR Decision §3 (vision-gate and §9
    /// fingerprint lookup are upstream of this function):
    ///   1. Classify via `classify_moe_tensor(arch, name)`.
    ///   2. For role ∈ {RoutedExpert, SharedExpert, Attention, Ssm,
    ///      Other}: compute the layer-region, index into the per-tier
    ///      7-tuple.
    ///   3. For role ∈ {TokenEmbd, Output, RouterGate, Norm}: return
    ///      the hardcoded picks (Q6_K / Q5_0 / F32).
    pub fn target_for(&self, tensor: &TensorRef) -> Result<GgmlType, ApexError> {
        // ADR §9 per-model override: if the CLI driver attached a
        // mudler config, IT IS authoritative (line 102). We do NOT
        // fall through to the algorithmic generator on a tensor miss
        // — `mudler_config::MudlerConfig::target_for` surfaces a
        // typed `TensorNotInMudlerConfig` so the no-silent-fallback
        // rule holds.
        //
        // Exception: structural tensors that mudler's files do NOT
        // enumerate (norms, token_embd, output, router gate). For
        // those, the algorithmic generator's hardcoded picks (F32 /
        // Q6_K / Q5_0) apply — they're not part of mudler's per-tier
        // surface by design (mudler's `generate_config.sh` only emits
        // the routed/shared/attn/ssm lines). To keep the override
        // strict on the tensors mudler DOES enumerate (and silent on
        // the ones it doesn't), we apply override-first for the
        // four MoE roles below; the structural arms below fall
        // through to the algorithmic hardcodes.
        if let Some(mudler) = self.mudler_override {
            if mudler.contains_match(tensor.name) {
                return mudler.target_for(tensor.name);
            }
            // Structural tensors mudler doesn't enumerate: token_embd,
            // output, output_norm, blk.N.{attn,ffn}_norm, router gate
            // (ffn_gate_inp). Fall through to the algorithmic
            // hardcodes below — this preserves the override's
            // strictness on the enumerated tensors (per-layer attn /
            // exps / shexp / ssm) while letting the small structural
            // set use llama.cpp's defaults, exactly mirroring stock
            // `llama-quantize --tensor-type-file`'s semantics.
            let role = classify_moe_tensor(self.arch, tensor.name);
            match role {
                MoeTensorRole::TokenEmbd
                | MoeTensorRole::Output
                | MoeTensorRole::RouterGate
                | MoeTensorRole::Norm => {
                    // OK — fall through to algorithmic below.
                }
                MoeTensorRole::RoutedExpert
                | MoeTensorRole::SharedExpert
                | MoeTensorRole::Attention
                | MoeTensorRole::Ssm
                | MoeTensorRole::Other => {
                    // Mudler's surface SHOULD enumerate these. Missing
                    // entry → typed error per the strict-override rule.
                    return Err(ApexError::TensorNotInMudlerConfig {
                        source_path: mudler.source_path.to_string(),
                        tensor_name: tensor.name.to_string(),
                    });
                }
            }
        }

        let role = classify_moe_tensor(self.arch, tensor.name);

        // Per-tier rule table for `RoutedExpert / SharedExpert / Attention / Ssm`.
        let rules = tier_rules(self.tier);

        match role {
            // --- Global / structural tensors (no layer index needed) ---
            // Mudler convention: token_embd and output stay at Q6_K
            // (implicit — llama.cpp's quantize-tool default for these
            // when no `--token-embedding-type` / `--output-tensor-type`
            // override is given is Q6_K). We mirror that explicitly
            // so `target_for`'s contract stays "policy chooses
            // everything".
            MoeTensorRole::TokenEmbd => Ok(GgmlType::Q6_K),
            MoeTensorRole::Output => Ok(GgmlType::Q6_K),
            // Router gate is a small per-token expert selector. Mudler
            // doesn't list it in `generate_config.sh`, so llama.cpp's
            // default fires — preserved at Q5_0 (small, perf-critical).
            MoeTensorRole::RouterGate => Ok(GgmlType::Q5_0),
            // Norms are never quantized. F32 always.
            MoeTensorRole::Norm => Ok(GgmlType::F32),

            // --- Per-block tensors (layer-indexed) ---
            MoeTensorRole::RoutedExpert => {
                let layer = self.require_layer_index(tensor)?;
                Ok(match exp_region(layer, self.n_layers) {
                    ExpRegion::Edge => rules.edge_exp,
                    ExpRegion::Near => rules.near_exp,
                    ExpRegion::Mid => rules.mid_exp,
                })
            }
            MoeTensorRole::SharedExpert => {
                let layer = self.require_layer_index(tensor)?;
                Ok(match shared_region(layer, self.n_layers) {
                    SharedRegion::Edge => rules.edge_shared,
                    SharedRegion::Mid => rules.mid_shared,
                })
            }
            MoeTensorRole::Attention => {
                let layer = self.require_layer_index(tensor)?;
                Ok(match attn_region(layer, self.n_layers) {
                    AttnRegion::Edge => rules.edge_attn,
                    AttnRegion::Mid => rules.mid_attn,
                })
            }
            MoeTensorRole::Ssm => {
                // SSM tensors pair with attention's `attn_type` per
                // `generate_config.sh:189-192`.
                let layer = self.require_layer_index(tensor)?;
                Ok(match attn_region(layer, self.n_layers) {
                    AttnRegion::Edge => rules.edge_attn,
                    AttnRegion::Mid => rules.mid_attn,
                })
            }
            MoeTensorRole::Other => {
                // Catch-all: route to attention-region quant for the
                // active layer. Mudler's bash doesn't enumerate
                // "everything else"; this is hf2q's choice to keep
                // the policy total (no silent escape).
                //
                // For a global "Other" tensor (no layer_index — e.g.
                // a future un-classified global) we fall back to
                // mid_attn since we have no layer to index.
                let region = match tensor.layer_index {
                    Some(l) => attn_region(l, self.n_layers),
                    None => AttnRegion::Mid,
                };
                Ok(match region {
                    AttnRegion::Edge => rules.edge_attn,
                    AttnRegion::Mid => rules.mid_attn,
                })
            }
        }
    }

    /// Helper: require a layer index for per-block tensors. The
    /// convert dispatcher is responsible for parsing `blk.<i>.` and
    /// setting `TensorRef::layer_index`; this errors if it forgot.
    fn require_layer_index(&self, tensor: &TensorRef) -> Result<usize, ApexError> {
        let layer = tensor.layer_index.ok_or_else(|| ApexError::MissingLayerIndex {
            name: tensor.name.to_string(),
        })?;
        if (layer as u64) >= (self.n_layers as u64) {
            return Err(ApexError::LayerIndexOutOfRange {
                name: tensor.name.to_string(),
                layer_index: layer,
                n_layers: self.n_layers,
            });
        }
        Ok(layer)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::tensor_ref::SourceDtype;
    use super::*;

    /// Build a `TensorRef` for tests with a 4096×N shape.
    fn tref<'a>(name: &'a str, arch: ArchName, layer: Option<usize>) -> TensorRef<'a> {
        // `shape` outlives the call via 'static.
        static SHAPE: [usize; 2] = [4096, 4096];
        TensorRef {
            name,
            shape: &SHAPE,
            source_dtype: SourceDtype::BF16,
            arch,
            layer_index: layer,
        }
    }

    /// Constructor rejects unsupported arches with the typed error
    /// and supported-set list intact.
    #[test]
    fn apex_policy_unsupported_arch_errors() {
        let err = ApexPolicy::new(ApexTier::Quality, ArchName::Llama3, 32, 0).unwrap_err();
        match err {
            ApexError::UnsupportedArch { arch, supported } => {
                assert_eq!(arch, "llama");
                assert!(supported.contains(&"qwen3moe"));
                assert!(supported.contains(&"gemma4"));
                assert!(supported.contains(&"minimax-m2"));
            }
            other => panic!("expected UnsupportedArch, got {other:?}"),
        }

        // Bert is dense embedding-only — also UnsupportedArch.
        let err = ApexPolicy::new(ApexTier::Quality, ArchName::Bert, 24, 0).unwrap_err();
        assert!(matches!(err, ApexError::UnsupportedArch { .. }));
    }

    /// Constructor rejects dense MoE-capable arches with the typed
    /// error (Gemma 4 with n_expert=0 is treated as dense).
    #[test]
    fn apex_policy_dense_model_errors() {
        let err = ApexPolicy::new(ApexTier::Quality, ArchName::Gemma4, 30, 0).unwrap_err();
        assert!(matches!(err, ApexError::DenseModelNotSupported { .. }));

        let err = ApexPolicy::new(ApexTier::Quality, ArchName::Gemma4, 30, 1).unwrap_err();
        assert!(matches!(err, ApexError::DenseModelNotSupported { .. }));
    }

    /// At tier=quality, 40-layer Qwen35Moe:
    ///   blk.0.ffn_gate_exps.weight → Q6_K (edge_exp_quality)
    #[test]
    fn apex_policy_routed_expert_edge_layer_0() {
        let p = ApexPolicy::new(ApexTier::Quality, ArchName::Qwen35Moe, 40, 128).unwrap();
        let t = tref("blk.0.ffn_gate_exps.weight", ArchName::Qwen35Moe, Some(0));
        assert_eq!(p.target_for(&t).unwrap(), GgmlType::Q6_K);
    }

    /// At tier=quality, 40-layer Qwen35Moe:
    ///   blk.20.ffn_gate_exps.weight → IQ4_XS (mid_exp_quality)
    #[test]
    fn apex_policy_routed_expert_mid_layer_20() {
        let p = ApexPolicy::new(ApexTier::Quality, ArchName::Qwen35Moe, 40, 128).unwrap();
        let t = tref("blk.20.ffn_gate_exps.weight", ArchName::Qwen35Moe, Some(20));
        assert_eq!(p.target_for(&t).unwrap(), GgmlType::IQ4_XS);
    }

    /// Tier=mini surfaces edge_attn=Q4_K vs mid_attn=Q3_K asymmetry
    /// for the SAME tensor name at different layer indices.
    /// On a 30-layer model (gemma4-26b), layer 2 is ATTN_EDGE
    /// (`i <= 2`) and layer 3 is ATTN_MID.
    #[test]
    fn apex_policy_attention_edge_vs_mid() {
        let p = ApexPolicy::new(ApexTier::Mini, ArchName::Gemma4, 30, 8).unwrap();

        let t_edge = tref("blk.2.attn_q.weight", ArchName::Gemma4, Some(2));
        assert_eq!(p.target_for(&t_edge).unwrap(), GgmlType::Q4_K);

        let t_mid = tref("blk.3.attn_q.weight", ArchName::Gemma4, Some(3));
        assert_eq!(p.target_for(&t_mid).unwrap(), GgmlType::Q3_K);
    }

    /// Router gate is Q5_0 regardless of tier.
    #[test]
    fn apex_policy_router_gate_q5_0() {
        for tier in [
            ApexTier::Quality,
            ApexTier::Balanced,
            ApexTier::Compact,
            ApexTier::Mini,
        ] {
            let p = ApexPolicy::new(tier, ArchName::Qwen35Moe, 40, 128).unwrap();
            let t = tref(
                "blk.5.ffn_gate_inp.weight",
                ArchName::Qwen35Moe,
                Some(5),
            );
            assert_eq!(
                p.target_for(&t).unwrap(),
                GgmlType::Q5_0,
                "tier {tier:?}"
            );
        }
    }

    /// Norms are F32 regardless of tier.
    #[test]
    fn apex_policy_norm_f32() {
        for tier in [
            ApexTier::Quality,
            ApexTier::Balanced,
            ApexTier::Compact,
            ApexTier::Mini,
        ] {
            let p = ApexPolicy::new(tier, ArchName::Qwen35Moe, 40, 128).unwrap();
            for name in [
                "blk.5.attn_norm.weight",
                "blk.5.ffn_norm.weight",
                "blk.5.attn_q_norm.weight",
                "output_norm.weight",
            ] {
                let layer = if name.starts_with("blk.") { Some(5) } else { None };
                let t = tref(name, ArchName::Qwen35Moe, layer);
                assert_eq!(
                    p.target_for(&t).unwrap(),
                    GgmlType::F32,
                    "tier {tier:?} name {name}"
                );
            }
        }
    }

    /// `token_embd.weight` → Q6_K regardless of tier.
    #[test]
    fn apex_policy_token_embd_q6_k() {
        for tier in [
            ApexTier::Quality,
            ApexTier::Balanced,
            ApexTier::Compact,
            ApexTier::Mini,
        ] {
            let p = ApexPolicy::new(tier, ArchName::Qwen35Moe, 40, 128).unwrap();
            let t = tref("token_embd.weight", ArchName::Qwen35Moe, None);
            assert_eq!(p.target_for(&t).unwrap(), GgmlType::Q6_K, "tier {tier:?}");
        }
    }

    /// Per-block tensor without `layer_index` → `MissingLayerIndex`
    /// (dispatcher contract bug).
    #[test]
    fn apex_policy_missing_layer_index_errors() {
        let p = ApexPolicy::new(ApexTier::Quality, ArchName::Qwen35Moe, 40, 128).unwrap();
        let t = tref("blk.0.ffn_gate_exps.weight", ArchName::Qwen35Moe, None);
        let err = p.target_for(&t).unwrap_err();
        assert!(matches!(err, ApexError::MissingLayerIndex { .. }));
    }

    /// `layer_index >= n_layers` → `LayerIndexOutOfRange`.
    #[test]
    fn apex_policy_layer_index_out_of_range_errors() {
        let p = ApexPolicy::new(ApexTier::Quality, ArchName::Qwen35Moe, 40, 128).unwrap();
        let t = tref("blk.40.ffn_gate_exps.weight", ArchName::Qwen35Moe, Some(40));
        let err = p.target_for(&t).unwrap_err();
        assert!(matches!(err, ApexError::LayerIndexOutOfRange { .. }));
    }

    /// Cross-check against the vendored verbatim config
    /// `vendor/apex-quant/configs/carnice_qwen36_mtp_quality.txt` —
    /// the operator's qwen3.6 41-layer production model class. At
    /// layer 5, ffn_gate_exps=Q5_K (NEAR_EXP_QUALITY), shared=Q8_0
    /// (EDGE/MID_SHARED_QUALITY both Q8_0), attn=Q6_K.
    #[test]
    fn apex_policy_matches_carnice_qwen36_layer_5() {
        // 41 layers (40 + 1 MTP per the vendored config's max blk.40).
        let p = ApexPolicy::new(ApexTier::Quality, ArchName::Qwen35Moe, 41, 128).unwrap();

        let exp = tref("blk.5.ffn_gate_exps.weight", ArchName::Qwen35Moe, Some(5));
        assert_eq!(p.target_for(&exp).unwrap(), GgmlType::Q5_K);

        let shexp = tref("blk.5.ffn_gate_shexp.weight", ArchName::Qwen35Moe, Some(5));
        assert_eq!(p.target_for(&shexp).unwrap(), GgmlType::Q8_0);

        let attn = tref("blk.5.attn_q.weight", ArchName::Qwen35Moe, Some(5));
        assert_eq!(p.target_for(&attn).unwrap(), GgmlType::Q6_K);
    }

    /// At mini tier on a 30-layer Gemma4 MoE, layer 10 routed-expert
    /// should be IQ2_S (mid_exp_mini), shared should be Q4_K
    /// (mid_shared_mini), attn should be Q3_K (mid_attn_mini).
    /// Cross-validated against `vendor/apex-quant/configs/gemma4_26b_mini.txt`.
    #[test]
    fn apex_policy_matches_gemma4_mini_layer_10() {
        let p = ApexPolicy::new(ApexTier::Mini, ArchName::Gemma4, 30, 8).unwrap();

        let exp = tref("blk.10.ffn_gate_exps.weight", ArchName::Gemma4, Some(10));
        assert_eq!(p.target_for(&exp).unwrap(), GgmlType::IQ2_S);

        let shexp = tref("blk.10.ffn_gate_shexp.weight", ArchName::Gemma4, Some(10));
        assert_eq!(p.target_for(&shexp).unwrap(), GgmlType::Q4_K);

        let attn = tref("blk.10.attn_q.weight", ArchName::Gemma4, Some(10));
        assert_eq!(p.target_for(&attn).unwrap(), GgmlType::Q3_K);
    }

    /// ADR §9 per-model override end-to-end smoke. Build an
    /// ApexPolicy at the algorithmic level, then attach the
    /// gemma4_26b_balanced.txt mudler config. The override MUST win
    /// for enumerated tensors and the structural fall-through MUST
    /// still produce Q5_0 / Q6_K / F32 for router-gate / token-embd
    /// / norms (mudler doesn't enumerate those).
    #[test]
    fn apex_policy_with_mudler_override_wins_for_enumerated_tensors() {
        use super::super::fingerprint::vendor_config_content;
        use super::super::mudler_config::MudlerConfig;

        // 30-layer Gemma4 MoE @ balanced — same arch as
        // gemma4_26b_balanced.txt. The vendored file sets
        // `blk.0.ffn_gate_exps=Q6_K` at edge layers and
        // `blk.5.ffn_gate_exps=Q5_K` at the near band.
        let content =
            vendor_config_content("vendor/apex-quant/configs/gemma4_26b_balanced.txt")
                .unwrap();
        // The override leaks; that's intentional for the
        // process-wide cache. For this test we leak a fresh copy so
        // the `&'static` lifetime is satisfied without polluting the
        // production cache slot.
        let mudler: &'static MudlerConfig = Box::leak(Box::new(
            MudlerConfig::parse(content, "test/gemma4_26b_balanced.txt").unwrap(),
        ));
        let p = ApexPolicy::new(ApexTier::Balanced, ArchName::Gemma4, 30, 128)
            .unwrap()
            .with_mudler_override(mudler);

        // Override wins for enumerated routed-expert tensors.
        let exp_0 = tref("blk.0.ffn_gate_exps.weight", ArchName::Gemma4, Some(0));
        assert_eq!(p.target_for(&exp_0).unwrap(), GgmlType::Q6_K);
        let exp_5 = tref("blk.5.ffn_gate_exps.weight", ArchName::Gemma4, Some(5));
        assert_eq!(p.target_for(&exp_5).unwrap(), GgmlType::Q5_K);

        // Structural fall-through: router gate stays Q5_0
        // (algorithmic hardcode; mudler doesn't enumerate it).
        let rg = tref("blk.5.ffn_gate_inp.weight", ArchName::Gemma4, Some(5));
        assert_eq!(p.target_for(&rg).unwrap(), GgmlType::Q5_0);

        // token_embd is structural; falls through to algorithmic
        // Q6_K hardcode.
        let te = tref("token_embd.weight", ArchName::Gemma4, None);
        assert_eq!(p.target_for(&te).unwrap(), GgmlType::Q6_K);

        // Norms are structural; F32.
        let nm = tref("blk.5.attn_norm.weight", ArchName::Gemma4, Some(5));
        assert_eq!(p.target_for(&nm).unwrap(), GgmlType::F32);
    }

    /// ADR-033 §Pi gate: requesting any `i-*` tier without imatrix
    /// support shipped MUST hard-error with `ImatrixRequiresInference`,
    /// not silently degrade to the non-I sibling's bytes.
    /// `SUPPORTED_FOR_IMATRIX` is empty in v1 (Pi pending), so all three
    /// I-tier variants reject on every arch.
    #[test]
    fn apex_policy_rejects_i_tier_without_imatrix() {
        for tier in [
            ApexTier::IQuality,
            ApexTier::IBalanced,
            ApexTier::ICompact,
        ] {
            let err = ApexPolicy::new(tier, ArchName::Gemma4, 30, 128).unwrap_err();
            match err {
                ApexError::ImatrixRequiresInference {
                    tier: t,
                    arch,
                    supported_for_imatrix,
                } => {
                    assert_eq!(t, tier.cli_name());
                    assert_eq!(arch, "gemma4");
                    assert_eq!(supported_for_imatrix, &[] as &[&str]);
                }
                other => panic!(
                    "expected ImatrixRequiresInference for {tier:?}, got {other:?}"
                ),
            }
        }
    }

    /// The non-I siblings (`Quality`, `Balanced`, `Compact`, `Mini`)
    /// MUST still construct cleanly — the §Pi gate is I-tier-only.
    #[test]
    fn apex_policy_accepts_non_i_tiers() {
        for tier in [
            ApexTier::Quality,
            ApexTier::Balanced,
            ApexTier::Compact,
            ApexTier::Mini,
        ] {
            ApexPolicy::new(tier, ArchName::Gemma4, 30, 128)
                .unwrap_or_else(|e| panic!("non-I tier {tier:?} rejected: {e}"));
        }
    }
}
