//! Actual typed load options only. No packet hash/argv inference or admission.
use crate::serve::api::engine::LoadOptions;
use anyhow::{ensure, Result};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(crate) enum TokenizerOrigin { EmbeddedGguf, ExternalOverride }
impl TokenizerOrigin {
    pub(crate) fn from_options(opts: &LoadOptions) -> Self {
        if opts.tokenizer_path.is_some() { Self::ExternalOverride } else { Self::EmbeddedGguf }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RuntimeRoutes { mmap_raw: bool, kernel_profile: bool, general_profile: bool, parallel_encode: bool }
impl RuntimeRoutes {
    fn actual() -> Self {
        let env = &crate::debug::investigation_env::INVESTIGATION_ENV;
        Self { mmap_raw: crate::inference::models::deepseek4::residency::raw_weights_mapped(),
            kernel_profile: env.mlx_kernel_profile, general_profile: env.mlx_profile,
            parallel_encode: env.parallel_encode_enabled() }
    }
}
/// Private fields/no Deserialize: this can only be derived from real options.
/// Validated means the stated LOAD options subset, not a complete runtime profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(super) struct SelectedLoadProfile {
    schema: &'static str,
    authority: &'static str,
    tokenizer_origin: TokenizerOrigin,
    effective_context: u32,
    external_config_option_present: bool,
    dwq_overlay_option_present: bool,
    persistence_option_present: bool,
    persistence_budget_bytes: u64,
    glp_option_present: bool,
    glp_alpha_option_present: bool,
    graft_option_present: bool,
    mmap_raw_weights: bool,
    kernel_profile: bool,
    general_profile: bool,
    parallel_encode: bool,
    queue_capacity: &'static str,
    logical_slots: &'static str,
    output_limit: &'static str,
    request_sampling: &'static str,
    request_logprobs_vision: &'static str,
    other_investigation_features: &'static str,
    loaded_persistor_graft: &'static str,
    runtime_qualified: bool,
}
impl SelectedLoadProfile {
    pub(super) fn from_actual(opts: &LoadOptions, effective_context: Option<u32>) -> Result<Self> {
        Self::derive(opts, effective_context, RuntimeRoutes::actual())
    }
    fn derive(opts: &LoadOptions, context: Option<u32>, routes: RuntimeRoutes) -> Result<Self> {
        ensure!(opts.model_path.is_absolute() && opts.model_path.as_os_str().as_encoded_bytes().len()<=4096,
            "Q5 selected actual model path must be bounded absolute");
        let origin=TokenizerOrigin::from_options(opts);
        ensure!(origin==TokenizerOrigin::EmbeddedGguf,
            "Q5 selected external tokenizer CPU/source budget unsupported; option not disabled");
        ensure!(opts.config_path.is_none() && opts.dwq_overlay_path.is_none(),
            "Q5 selected external config/DWQ coactivation unsupported");
        ensure!(opts.kv_persist_dir.is_none(),
            "Q5 selected persistence Some requires separate full image/index/pending budget");
        ensure!(opts.glp_path.is_none() && opts.glp_alpha.is_none() && opts.kv_graft_path.is_none(),
            "Q5 selected GLP/graft coactivation unsupported");
        ensure!(context==Some(262_144),
            "Q5 actual effective context Unknown or unsupported; cannot infer from packet");
        ensure!(routes.mmap_raw && !routes.kernel_profile && !routes.general_profile && !routes.parallel_encode,
            "Q5 selected actual mmap/profiling/parallel route unsupported");
        Ok(Self { schema:"hf2q.q5.deepseek.actual-load-options.v1",
            authority:"ActualTypedLoadOptionsAndSelectedSourcePredicates_NotAdmission",
            tokenizer_origin:origin,effective_context:262_144,external_config_option_present:false,
            dwq_overlay_option_present:false,persistence_option_present:false,
            // Retain even when unused; no assumption that absence implies zero.
            persistence_budget_bytes:opts.kv_persist_budget_bytes,glp_option_present:false,
            glp_alpha_option_present:false,graft_option_present:false,mmap_raw_weights:routes.mmap_raw,
            kernel_profile:routes.kernel_profile,general_profile:routes.general_profile,parallel_encode:routes.parallel_encode,
            queue_capacity:"ExplicitUnknown_NotLoadOptions_RequiresActualSpawnJoin",
            logical_slots:"ExplicitUnknown_NotLoadOptions_RequiresActualProvisionJoin",
            output_limit:"ExplicitUnknown_NotLoadOptions_RequiresActualRequestJoin",
            request_sampling:"ExplicitUnknown_NotLoadOptions_RequiresActualRequestJoin",
            request_logprobs_vision:"ExplicitUnknown_NotLoadOptions_RequiresActualRequestJoin",
            other_investigation_features:"ExplicitUnknown_NotACompleteFeatureCensus",
            loaded_persistor_graft:"ExplicitUnknown_ModelNotConstructed_NoLoadedStateClaim",
            runtime_qualified:false })
    }
    pub(super) fn require_actual_options(&self, opts:&LoadOptions, context:Option<u32>) -> Result<()> {
        ensure!(*self==Self::from_actual(opts,context)?, "Q5 actual selected load profile changed");Ok(())
    }
    pub(super) fn require_current_routes(&self) -> Result<()> {
        let now=RuntimeRoutes::actual();
        ensure!(now==RuntimeRoutes { mmap_raw:self.mmap_raw_weights,kernel_profile:self.kernel_profile,
            general_profile:self.general_profile,parallel_encode:self.parallel_encode },
            "Q5 selected source route drift; no further transaction");Ok(())
    }
}
#[cfg(test)]
#[path="selected_profile_tests.rs"]
mod tests;
