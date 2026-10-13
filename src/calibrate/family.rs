//! ADR-054 per-family capture (#300): the family abstraction `hf2q
//! calibrate` dispatches on. The pipeline skeleton (corpus, contrastive
//! arms, distill, export, canary, register report) is family-agnostic and
//! lives in `mod.rs`; each family adapter supplies the parts that genuinely
//! differ — capture read arms, tokenizer, forward plumbing, the canary bind
//! contract, and the lexical register.
//!
//! The calibrate family set is the set of families with a capture adapter
//! here. Since #299 (2026-10-13) that is NO LONGER identical to the
//! serve-side GLP family set: gemma4 now serves `--glp`
//! (`residual_stream_post_layer` apply arm) but has no calibrate capture
//! adapter, so it is refused HERE only — an operator can steer a gemma4
//! model with a published or externally calibrated vector, and calibrate
//! names the implemented adapters when it refuses.

use anyhow::{bail, Result};
use mlx_native::gguf::GgufFile;

use crate::inference::glp::bind::BoundGlp;
use crate::inference::glp::reader::{GlpHookPoint, GlpVector};

use super::deepseek4::Deepseek4Family;
use super::qwen35::Qwen35Family;

/// The spec hook sites a family can CAPTURE at — a subset of the sites its
/// engine can APPLY at (capture additionally needs a read arm in the
/// forward path). An adapter lists only what it has a real read arm for,
/// so a future site cannot be silently routed through a wrong-site capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureSite {
    /// The complete post-layer residual state.
    ResidualStreamPostLayer,
    /// The FFN writer output, pre-fold (`ffn_output = moe + shared`,
    /// immediately before the residual/hyper-connection fold).
    FfnOutPreResidual,
}

impl CaptureSite {
    /// Spec-spelled site name — the exact string exported as
    /// `glp.derived_at` (and accepted for `glp.hook_point`).
    pub fn as_spec_str(self) -> &'static str {
        match self {
            Self::ResidualStreamPostLayer => "residual_stream_post_layer",
            Self::FfnOutPreResidual => "ffn_out_pre_residual",
        }
    }

    /// The matching spec hook point (capture sites are a subset of hook
    /// sites, same spelling).
    pub fn as_hook_point(self) -> GlpHookPoint {
        match self {
            Self::ResidualStreamPostLayer => GlpHookPoint::ResidualStreamPostLayer,
            Self::FfnOutPreResidual => GlpHookPoint::FfnOutPreResidual,
        }
    }

    /// CLI spelling (`--capture-site`/`--hook-point` values).
    pub fn parse_cli(value: &str) -> Result<Self> {
        match value {
            "residual" => Ok(Self::ResidualStreamPostLayer),
            "writer" => Ok(Self::FfnOutPreResidual),
            other => bail!(
                "unknown site {other:?}; expected `residual` or `writer` \
                 (residual_stream_post_layer / ffn_out_pre_residual)"
            ),
        }
    }
}

/// One captured committed state (last sequence position).
///
/// `Dense([hidden])` — the Qwen post-layer residual, or any family's
/// FFN-writer site. `MhcStreams([hc * hidden])` — the DeepSeek-V4
/// post-layer complete hyper-connection state, stream-major.
#[derive(Debug)]
pub enum CapturedState {
    Dense(Vec<f32>),
    MhcStreams(Vec<f32>),
}

impl CapturedState {
    /// Flat stream-major view `[hc][hidden]` (`hc = 1` for a dense
    /// capture; the stream count comes from `FamilyFacts`). `ArmSums`
    /// consumes exactly this shape, so the shared distill path is
    /// capture-shape-generic.
    pub fn as_flat(&self) -> &[f32] {
        match self {
            Self::Dense(values) => values.as_slice(),
            Self::MhcStreams(values) => values.as_slice(),
        }
    }
}

/// Per-family model facts, read from the GGUF (never hardcoded): shapes,
/// the implemented capture sites, and the family's default export pair.
/// `Clone` so the pipeline can snapshot the facts and release the
/// adapter borrow before the mutable capture/encode calls.
#[derive(Clone, Debug)]
pub struct FamilyFacts {
    /// `"deepseek4"` | `"qwen35"` | `"qwen35moe"` — the GGUF
    /// `general.architecture` verbatim.
    pub family: &'static str,
    pub num_layers: u32,
    pub hidden: u32,
    /// `Some(k)` for deepseek4 only (the hyper-connection stream count).
    pub hc: Option<u32>,
    /// Only the sites this family's forward path can actually read.
    pub capture_sites: &'static [CaptureSite],
    /// Default export apply site (`glp.hook_point`).
    pub default_hook: GlpHookPoint,
    /// Default capture site, spec-spelled — the `--capture-site` default
    /// (deepseek4 keeps the residual capture; qwen captures same-site at
    /// its published hook class).
    pub default_derived_at: &'static str,
    /// Per-family capture-layer default (deepseek4: 29 — the measured
    /// export layer; qwen: the GLP-49 span midpoint, clamped to the
    /// artifact's layer count at load).
    pub default_layer: usize,
}

impl FamilyFacts {
    /// The capture's stream count at `site`: the post-layer residual is the
    /// family's complete state (deepseek4: `[hc, hidden]` mHC; qwen: dense),
    /// while the FFN-writer site is a single dense `[hidden]` row for every
    /// family — the writer buffer has no per-stream structure pre-fold.
    /// `ArmSums` and the fail-closed capture-shape check consume exactly
    /// this per-site shape, never the family's residual-stream count.
    pub fn capture_streams(&self, site: CaptureSite) -> usize {
        match site {
            CaptureSite::ResidualStreamPostLayer => {
                self.hc.map(|k| k as usize).unwrap_or(1)
            }
            CaptureSite::FfnOutPreResidual => 1,
        }
    }

    /// Fail-closed site support check naming the implemented sites.
    pub fn ensure_capture_site(&self, site: CaptureSite) -> Result<()> {
        if self.capture_sites.contains(&site) {
            return Ok(());
        }
        let supported = self
            .capture_sites
            .iter()
            .map(|site| site.as_spec_str())
            .collect::<Vec<_>>()
            .join(", ");
        bail!(
            "capture site {} is not implemented for family {} \
             (implemented capture sites: {supported}); a site is never \
             silently routed through another family's capture",
            site.as_spec_str(),
            self.family
        );
    }
}

/// The per-family calibration adapter. The two engine types share no
/// inference trait, and the repo's discipline is explicit family behavior
/// — so the abstraction is a trait the two adapters implement, not a
/// dyn-forward path and not an approximately-compatible loader shortcut.
/// `mod.rs` holds the loaded family as a boxed adapter and never names a
/// model type.
pub trait CalibrationFamily {
    fn facts(&self) -> &FamilyFacts;

    /// Teacher-forced encode through the family tokenizer.
    fn encode(&mut self, text: &str) -> Result<Vec<u32>>;

    /// Prefill-only capture of the committed state at `layer`, reading the
    /// named site. Fresh cache per prompt (one capture = one cache, the
    /// same discipline as the DeepSeek path today). Fails closed if the
    /// layer is out of range or the site is not in `facts().capture_sites`.
    fn capture(&mut self, tokens: &[u32], site: CaptureSite, layer: usize)
        -> Result<CapturedState>;

    /// Fixed-probe logits with no GLP bound (canary baseline) — one
    /// prefill + head read.
    fn probe_logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;

    /// Bind the exported vector for the live-dose canary, at the family's
    /// supported hook set and residual-site contract (deepseek4 →
    /// `GlpResidualSite::MhcStreams`; qwen → `Dense`). Exercising the exact
    /// serve-side bind here means a calibrate output that would be refused
    /// at serve time is refused at calibrate time.
    fn bind_glp(&mut self, vector: GlpVector, alpha: Option<f32>) -> Result<()>;

    /// Unbind any bound GLP (restores the unsteered baseline).
    fn clear_glp(&mut self);

    /// Greedy decode of the register probes through the family chat
    /// template (the deployment register, not raw-completion register).
    fn generate_openings(&mut self, probes: &[String], max_tokens: usize) -> Result<Vec<String>>;
}

/// Which adapter an architecture dispatches to (the dispatch matrix).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FamilyKind {
    Deepseek4,
    /// One adapter covers both `qwen35` (dense) and `qwen35moe` (MoE) —
    /// `Qwen35Model` loads both variants and `cfg.variant` records which.
    Qwen35,
}

/// Pure architecture → adapter dispatch, fail-closed. The implemented
/// adapters cover qwen35/qwen35moe/deepseek4; gemma4 serves `--glp` since
/// #299 but has no calibrate capture adapter, so it is refused HERE only
/// (the serve-side refusal set is `GLP_SERVING_ARCHITECTURES` — a superset
/// since gemma4 joined it; the two surfaces can no longer be described as
/// one set, and this message names the honest split).
pub fn family_kind_for_architecture(architecture: &str) -> Result<FamilyKind> {
    match architecture {
        "deepseek4" => Ok(FamilyKind::Deepseek4),
        "qwen35" | "qwen35moe" => Ok(FamilyKind::Qwen35),
        other => bail!(
            "hf2q calibrate is not supported for `{other}` models; it is \
             available for qwen35, qwen35moe, deepseek4 — the families with \
             capture adapters (gemma4 serves --glp since #299 but has no \
             calibrate capture adapter yet; it is refused here only, not at \
             serve time)",
        ),
    }
}

/// Load the calibration adapter for a model GGUF, dispatching on
/// `general.architecture` (the same key the serve runtime uses). The
/// refusal fires before any config parse or weight load.
pub fn load_family(gguf: &GgufFile) -> Result<Box<dyn CalibrationFamily>> {
    let architecture = gguf
        .metadata_string("general.architecture")
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("GGUF is missing required `general.architecture`"))?;
    match family_kind_for_architecture(architecture)? {
        FamilyKind::Deepseek4 => Ok(Box::new(Deepseek4Family::load(gguf)?)),
        FamilyKind::Qwen35 => Ok(Box::new(Qwen35Family::load(gguf)?)),
    }
}

/// Config-only family facts for the same dispatch — the cheap fail-fast
/// probe: family refusal, site support, and layer-range validation all
/// run against this BEFORE the (potentially very large) weight load,
/// mirroring the pre-#300 pipeline's config-first validation order.
pub fn family_facts(gguf: &GgufFile) -> Result<FamilyFacts> {
    let architecture = gguf
        .metadata_string("general.architecture")
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("GGUF is missing required `general.architecture`"))?;
    match family_kind_for_architecture(architecture)? {
        FamilyKind::Deepseek4 => Deepseek4Family::facts_from_gguf(gguf),
        FamilyKind::Qwen35 => Qwen35Family::facts_from_gguf(gguf),
    }
}

/// The family bind surface both adapters share: every GLP family supports
/// both spec hook sites (ADR-053 dual hook sites).
pub(crate) const FAMILY_HOOK_SET: [GlpHookPoint; 2] = [
    GlpHookPoint::ResidualStreamPostLayer,
    GlpHookPoint::FfnOutPreResidual,
];

/// Post-bind debug summary shared by both adapters (the historical
/// `[calibrate] debug: bound …` line; the per-layer `present` term became
/// the full bound-layer list when binding moved behind the family trait —
/// the line is plumbing diagnostics, not part of any export contract).
pub(crate) fn bound_glp_debug_line(bound: &BoundGlp) -> String {
    format!(
        "[calibrate] debug: bound alpha={} layers={:?}",
        bound.alpha,
        bound.device_directions.keys().collect::<Vec<_>>()
    )
}
