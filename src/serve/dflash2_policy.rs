//! DFlash2 default policy (ADR-067 D6): which route a model serves on.
//!
//! `HF2Q_DFLASH2` is a three-state policy, resolved once per model load,
//! before any weight is read:
//!
//! - unset: **auto**. DFlash2 and its route profile
//!   ([`route_profile::SWITCHES`]) are selected when the loaded target is a
//!   pinned pair, or has the architecture and shapes of a pinned
//!   descriptor's target (a fine-tune or another quantization of the same
//!   base); anything else keeps the prior promoted route and says why. The
//!   profile is applied switch by switch: a switch whose kernels do not
//!   admit the artifact is left off and the drafter still serves with the
//!   rest. Auto never fails startup.
//! - `0`/`off`/`false`: **off**, the kill switch. No drafter, no profile: the
//!   route the server ran before DFlash2 was a default.
//! - `1`/`on`/`true`: **required**. A target that cannot be served with
//!   DFlash2 fails startup.
//!
//! The decision has to precede the weight load: several profile switches are
//! read while weights are uploaded, and a missing drafter is a prerequisite
//! that must be found before a large model is loaded.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use mlx_native::gguf::GgufFile;
use mlx_native::GgmlType;

use crate::core::provenance::Provenance;
use crate::inference::models::qwen35::Qwen35Config;
use crate::inference::route_host::{self, HostRoute, HostRoutes};
use crate::inference::spec_decode::dflash2::config::DFlash2TargetShape;
use crate::inference::spec_decode::dflash2::route_profile::{
    self, switch, turns_on, PlaneSet, ResolvedRoutes, RouteProfile, Switch, SwitchRow, SWITCHES,
};
use crate::inference::spec_decode::dflash2::schema::{
    DFlash2CheckpointDescriptor, DFlash2FamilyAdapter, QwenTargetGeometry,
    KNOWN_DFLASH2_CHECKPOINTS,
};
use crate::inference::spec_decode::dflash2::weights::DFlash2WeightsFile;
use crate::serve::api::engine::EngineMode;
use crate::serve::resident_memory::Commitment;

#[path = "dflash2_policy_memory.rs"]
mod allocation_memory;
use allocation_memory::DrafterMemoryBound;

/// The policy switch.
pub(crate) const DFLASH2_ENV: &str = "HF2Q_DFLASH2";
/// Explicit drafter configuration; used only together with the weights path.
pub(crate) const DFLASH2_CONFIG_ENV: &str = "HF2Q_DFLASH2_CONFIG";
/// Explicit drafter weights; used only together with the configuration path.
pub(crate) const DFLASH2_WEIGHTS_ENV: &str = "HF2Q_DFLASH2_WEIGHTS";

const CONFIG_FILE: &str = "config.json";
const WEIGHTS_FILE: &str = "model.safetensors";

/// Physical memory the memory rule leaves to the operating system and the
/// process's own non-weight state, beside the configured KV budget.
pub(crate) const MEMORY_RESERVE_BYTES: u64 = 8 << 30;

/// Widest SlotAware worker a DFlash2 runtime serves.
const MAX_DFLASH2_SLOTS: u32 = 4;

/// What `HF2Q_DFLASH2` asks for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PolicyRequest {
    Auto,
    Off,
    Required,
}

impl PolicyRequest {
    fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
            Self::Required => "required",
        }
    }
}

/// Parse `HF2Q_DFLASH2`; unset or blank is auto. Any other value than the
/// documented ones fails startup, as it did before the policy existed.
pub(crate) fn parse_policy(raw: Option<&str>) -> Result<PolicyRequest, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(PolicyRequest::Auto);
    };
    match raw.to_ascii_lowercase().as_str() {
        "0" | "off" | "false" => Ok(PolicyRequest::Off),
        "1" | "on" | "true" => Ok(PolicyRequest::Required),
        _ => Err(format!(
            "{DFLASH2_ENV} must be 1/on/true or 0/off/false, got {raw:?}"
        )),
    }
}

/// Where the drafter files came from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DrafterSource {
    /// `HF2Q_DFLASH2_CONFIG` and `HF2Q_DFLASH2_WEIGHTS`.
    Explicit,
    /// The Hugging Face cache snapshot at the pinned revision.
    Cache,
    /// Downloaded into that cache during this start.
    Downloaded,
}

impl DrafterSource {
    fn label(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::Cache => "cache",
            Self::Downloaded => "downloaded",
        }
    }
}

/// A located drafter checkpoint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DrafterFiles {
    pub(crate) config_path: PathBuf,
    pub(crate) weights_path: PathBuf,
    pub(crate) source: DrafterSource,
}

/// Why a target qualifies for a descriptor's drafter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QualifiedClass {
    /// hf2q provenance and the descriptor's exact target source identity.
    PinnedPair,
    /// The architecture and shapes of the descriptor's target: a fine-tune
    /// or another quantization of the same base, whatever its provenance
    /// (operator decision 2026-11-12).
    SameShapes,
}

impl QualifiedClass {
    fn label(self) -> &'static str {
        match self {
            Self::PinnedPair => "pinned_pair",
            Self::SameShapes => "same_shapes",
        }
    }
}

/// The drafter a model's worker provisions.
#[derive(Clone, Debug)]
pub(crate) struct DFlash2Plan {
    pub(crate) descriptor: &'static DFlash2CheckpointDescriptor,
    pub(crate) class: QualifiedClass,
    pub(crate) config_path: PathBuf,
    pub(crate) weights_path: PathBuf,
    pub(crate) source: DrafterSource,
}

/// One tensor of the target's GGUF directory.
#[derive(Clone, Debug)]
pub(crate) struct TensorFact {
    pub(crate) name: String,
    pub(crate) ggml_type: GgmlType,
    /// Outermost-first, as the GGUF reader reports it.
    pub(crate) shape: Vec<usize>,
}

/// What the policy reads from the target before its weights are loaded.
#[derive(Clone, Debug)]
pub(crate) struct TargetFacts {
    pub(crate) architecture: String,
    pub(crate) provenance: Provenance,
    /// Slots of a SlotAware worker; `None` for any other scheduler.
    pub(crate) slots: Option<u32>,
    /// Rows one slot's KV cache can hold: the context length the engine is
    /// loaded with. `None` when the header does not resolve one. A scratch
    /// bound that grows with the cache takes this capacity, never the rows a
    /// conversation happens to use.
    pub(crate) kv_rows: Option<usize>,
    pub(crate) model_file_bytes: u64,
    pub(crate) tensors: Vec<TensorFact>,
    /// Architecture, dimensions, bound tensor shapes and tokenizer identity;
    /// `None` when the header does not describe a Qwen target.
    pub(crate) geometry: Option<QwenTargetGeometry>,
    /// Parsed once from the target header; scratch bounds use the loader's
    /// full geometry rather than reconstructing it from a descriptor.
    pub(crate) qwen_config: Option<Qwen35Config>,
    /// Whether a DWQ overlay is applied on load.
    pub(crate) dwq_overlay: bool,
    /// Whether the chunk-scan prefill investigation route is on; the
    /// prepared prefill refuses to run beside it.
    pub(crate) chunk_scan_prefill: bool,
}

/// The geometry of a Qwen target, from its GGUF header alone.
fn read_geometry(
    gguf: &GgufFile,
    config: &Qwen35Config,
) -> Option<QwenTargetGeometry> {
    let architecture = gguf.metadata_string("general.architecture")?;
    let shape = |name: &str| -> Option<[usize; 2]> {
        <[usize; 2]>::try_from(gguf.tensor_info(name)?.shape.as_slice()).ok()
    };
    let moe = config.moe.as_ref();
    Some(QwenTargetGeometry {
        architecture: architecture.to_owned().into(),
        hidden_size: config.hidden_size,
        layers: config.num_hidden_layers,
        vocab_size: config.vocab_size,
        attention_heads: config.num_attention_heads,
        attention_kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        linear_key_heads: config.linear_num_key_heads,
        linear_value_heads: config.linear_num_value_heads,
        linear_key_head_dim: config.linear_key_head_dim,
        linear_value_head_dim: config.linear_value_head_dim,
        full_attention_interval: config.full_attention_interval,
        intermediate_size: config.intermediate_size.unwrap_or(0),
        moe_experts: moe.map_or(0, |moe| moe.num_experts),
        moe_top_k: moe.map_or(0, |moe| moe.num_experts_per_tok),
        moe_intermediate_size: moe.map_or(0, |moe| moe.moe_intermediate_size),
        moe_shared_intermediate_size: moe.map_or(0, |moe| moe.shared_expert_intermediate_size),
        embedding_shape: shape("token_embd.weight")?,
        // A target without `output.weight` ties its head to the embedding.
        head_shape: shape("output.weight").or_else(|| shape("token_embd.weight"))?,
        tokenizer_sha256: crate::inference::models::qwen35::tokenizer::tokenizer_identity(gguf)
            .ok()?
            .into(),
    })
}

impl TargetFacts {
    /// Read the facts from the GGUF header and the engine configuration.
    pub(crate) fn read(
        gguf: &GgufFile,
        path: &Path,
        engine_mode: EngineMode,
        dwq_overlay: bool,
        requested_context: Option<crate::serve::operator_settings::RequestedContext>,
    ) -> Self {
        let architecture = gguf
            .metadata_string("general.architecture")
            .unwrap_or_default()
            .to_owned();
        let qwen = matches!(architecture.as_str(), "qwen35" | "qwen35moe");
        let tensors = if qwen {
            gguf.tensor_names()
                .into_iter()
                .filter_map(|name| gguf.tensor_info(name))
                .map(|info| TensorFact {
                    name: info.name.clone(),
                    ggml_type: info.ggml_type,
                    shape: info.shape.clone(),
                })
                .collect()
        } else {
            Vec::new()
        };
        let qwen_config = qwen
            .then(|| Qwen35Config::from_gguf(gguf).ok())
            .flatten();
        let geometry = qwen_config
            .as_ref()
            .and_then(|config| read_geometry(gguf, config));
        Self {
            provenance: crate::core::provenance::detect(gguf),
            slots: match engine_mode {
                EngineMode::SlotAware { max_slots } => Some(max_slots),
                _ => None,
            },
            // The same resolution the load performs.
            kv_rows: crate::serve::operator_settings::resolve_context_for_gguf(
                gguf,
                requested_context,
            )
            .ok()
            .map(|context| context.effective_tokens as usize),
            model_file_bytes: std::fs::metadata(path).map_or(0, |metadata| metadata.len()),
            tensors,
            geometry,
            qwen_config,
            dwq_overlay,
            chunk_scan_prefill: crate::debug::INVESTIGATION_ENV.chunk_scan_prefill,
            architecture,
        }
    }

    /// The target's own source identity, for the audit record.
    fn source_identity(&self) -> String {
        match &self.provenance {
            Provenance::Hf2q { source_sha256, .. } => source_sha256.clone(),
            Provenance::External => "external".to_owned(),
        }
    }
}

/// What the policy reads from the host.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HostFacts {
    pub(crate) physical_bytes: u64,
    /// What the host reports available now: free, inactive and purgeable
    /// pages. Nobody holds these bytes, whatever the ledger says; what was
    /// promised and not yet taken still comes out of them.
    pub(crate) available_bytes: u64,
    /// The configured aggregate KV budget; zero when none is configured.
    pub(crate) kv_budget_bytes: u64,
    /// What the engines already resident in this process have committed
    /// (`serve::resident_memory`).
    pub(crate) resident_bytes: u64,
}

impl HostFacts {
    pub(crate) fn read(kv_budget_bytes: Option<u64>, resident_bytes: u64) -> Self {
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        Self {
            physical_bytes: system.total_memory(),
            available_bytes: system.available_memory(),
            kv_budget_bytes: kv_budget_bytes.unwrap_or(0),
            resident_bytes,
        }
    }
}

/// Locates a pinned drafter checkpoint.
pub(crate) trait DrafterStore {
    /// The Hugging Face cache snapshot of `descriptor`, if both files are
    /// there.
    fn cached(&self, descriptor: &DFlash2CheckpointDescriptor) -> Option<(PathBuf, PathBuf)>;
    /// Download both files at the pinned revision into that cache and verify
    /// them. The error is the reason, in words an operator can act on.
    fn download(
        &self,
        descriptor: &DFlash2CheckpointDescriptor,
    ) -> std::result::Result<(PathBuf, PathBuf), String>;
    /// Whether an explicitly named checkpoint can be used at all; the error
    /// is why not (a path that is not a file).
    fn explicit_usable(
        &self,
        config_path: &Path,
        weights_path: &Path,
    ) -> std::result::Result<(), String>;
    /// Bind the located checkpoint configuration to the target's shape with
    /// the check the target adapter itself runs
    /// ([`DFlash2Config::validate_target`], what
    /// `DFlash2TargetFamily::validate` calls on the loaded model).
    ///
    /// [`DFlash2Config::validate_target`]: crate::inference::spec_decode::dflash2::config::DFlash2Config::validate_target
    fn bind(
        &self,
        descriptor: &DFlash2CheckpointDescriptor,
        config_path: &Path,
        target: DFlash2TargetShape<'_>,
    ) -> std::result::Result<(), String>;
}

/// The standard Hugging Face cache and the Hub.
pub(crate) struct HubDrafterStore;

impl DrafterStore for HubDrafterStore {
    fn cached(&self, descriptor: &DFlash2CheckpointDescriptor) -> Option<(PathBuf, PathBuf)> {
        let file = |name| {
            crate::input::hf_download::cached_pinned_file(
                descriptor.repository,
                descriptor.revision,
                name,
            )
        };
        let config = file(CONFIG_FILE)?;
        let weights = file(WEIGHTS_FILE)?;
        // A partial or foreign file is not a cache hit: the cheap checks run
        // here, the weight digest where the checkpoint is opened.
        let config_bytes = std::fs::read(&config).ok()?;
        descriptor.parse_config(&config_bytes).ok()?;
        let weights_bytes = std::fs::metadata(&weights).ok()?.len();
        (weights_bytes == descriptor.weights_bytes).then_some((config, weights))
    }

    fn download(
        &self,
        descriptor: &DFlash2CheckpointDescriptor,
    ) -> std::result::Result<(PathBuf, PathBuf), String> {
        if std::env::var("HF_HUB_OFFLINE").is_ok_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "on" | "true" | "yes"
            )
        }) {
            return Err("HF_HUB_OFFLINE is set".to_owned());
        }
        let fetch = |name: &'static str, bytes: Option<u64>| {
            tracing::info!(
                repository = descriptor.repository,
                revision = descriptor.revision,
                file = name,
                "downloading the pinned DFlash2 drafter"
            );
            let mut last_decile = None;
            let mut report = |progress: crate::input::hf_download::HubTransferProgress| {
                if progress.total_bytes == 0 {
                    return;
                }
                let decile = progress.completed_bytes.saturating_mul(10) / progress.total_bytes;
                if last_decile != Some(decile) {
                    last_decile = Some(decile);
                    tracing::info!(
                        repository = descriptor.repository,
                        file = name,
                        completed_bytes = progress.completed_bytes,
                        total_bytes = progress.total_bytes,
                        elapsed_ms = progress.elapsed_ms,
                        "DFlash2 drafter download progress"
                    );
                }
            };
            crate::input::hf_download::download_pinned_file_with_progress(
                descriptor.repository,
                descriptor.revision,
                name,
                bytes,
                &mut report,
            )
            .map_err(|error| error.to_string())
        };
        let config = fetch(CONFIG_FILE, None)?;
        let weights = fetch(WEIGHTS_FILE, Some(descriptor.weights_bytes))?;
        let config_bytes = std::fs::read(&config)
            .map_err(|error| format!("read downloaded {CONFIG_FILE}: {error}"))?;
        descriptor
            .parse_config(&config_bytes)
            .map_err(|error| format!("downloaded {CONFIG_FILE}: {error}"))?;
        DFlash2WeightsFile::open_verified(&weights, descriptor)
            .map_err(|error| format!("downloaded {WEIGHTS_FILE}: {error}"))?;
        Ok((config, weights))
    }

    fn explicit_usable(
        &self,
        config_path: &Path,
        weights_path: &Path,
    ) -> std::result::Result<(), String> {
        for (name, path) in [
            (DFLASH2_CONFIG_ENV, config_path),
            (DFLASH2_WEIGHTS_ENV, weights_path),
        ] {
            if !path.is_file() {
                return Err(format!("{name}={} is not a file", path.display()));
            }
        }
        Ok(())
    }

    fn bind(
        &self,
        descriptor: &DFlash2CheckpointDescriptor,
        config_path: &Path,
        target: DFlash2TargetShape<'_>,
    ) -> std::result::Result<(), String> {
        let bytes = std::fs::read(config_path)
            .map_err(|error| format!("read {}: {error}", config_path.display()))?;
        descriptor
            .parse_config(&bytes)
            .map_err(|error| error.to_string())?
            .validate_target(target)
            .map_err(|error| error.to_string())
    }
}

/// The policy's decision for one model load.
#[derive(Clone, Debug)]
pub(crate) struct Selection {
    pub(crate) request: PolicyRequest,
    /// The drafter to provision; `None` keeps the prior route.
    pub(crate) plan: Option<DFlash2Plan>,
    /// Every table switch as this model reads it.
    pub(crate) routes: Arc<ResolvedRoutes>,
    /// Why DFlash2 was or was not selected. When selected it names the
    /// class that qualified the target.
    pub(crate) reason: String,
    /// The descriptor the target matched, selected or not.
    pub(crate) descriptor: Option<&'static DFlash2CheckpointDescriptor>,
    /// The target's own source identity (`external` without hf2q provenance).
    pub(crate) target_source: String,
    /// Conditions an operator should see at WARN.
    pub(crate) warnings: Vec<String>,
    /// Memory rule inputs and estimates, when the rule ran.
    pub(crate) memory: Option<MemoryEstimate>,
    /// What an engine loaded on this selection holds resident.
    pub(crate) commitment: Commitment,
    /// The class that qualified the target, when one did.
    pub(crate) class: Option<QualifiedClass>,
    /// This model without a drafter: what a load that gives the drafter up
    /// is served on.
    prior: PriorRoute,
    /// Rows one slot's KV cache can hold ([`TargetFacts::kv_rows`]).
    kv_rows: Option<usize>,
}

/// The route of a model without a drafter: every switch from the process
/// environment alone. A switch the operator set keeps its route whatever
/// `HF2Q_DFLASH2` says, and the planes that route prepares are resident all
/// the same, so they are part of the commitment.
#[derive(Clone, Debug)]
struct PriorRoute {
    routes: Arc<ResolvedRoutes>,
    commitment: Commitment,
}

impl PriorRoute {
    fn of(facts: &TargetFacts, host: HostFacts, env: &dyn Fn(&str) -> Option<String>) -> Self {
        let routes = ResolvedRoutes::resolve(None, &|_| None, env);
        Self {
            commitment: route_commitment(facts, host, &routes, None),
            routes: Arc::new(routes),
        }
    }
}

/// What an engine of the target holds resident on `routes`, with the drafter
/// of `drafter_bytes` when it has one.
fn route_commitment(
    facts: &TargetFacts,
    host: HostFacts,
    routes: &ResolvedRoutes,
    drafter_bytes: Option<u64>,
) -> Commitment {
    Commitment {
        model_file_bytes: facts.model_file_bytes,
        kv_budget_bytes: host.kv_budget_bytes,
        drafter_bytes: drafter_bytes.unwrap_or(0),
        plane_bytes: routed_plane_bytes(facts, routes, drafter_bytes.is_some()),
        scratch_bytes: route_scratch_bytes(facts, routes),
    }
}

impl Selection {
    fn prior_route(
        request: PolicyRequest,
        facts: &TargetFacts,
        prior: &PriorRoute,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            request,
            plan: None,
            routes: Arc::clone(&prior.routes),
            reason: reason.into(),
            descriptor: None,
            target_source: facts.source_identity(),
            warnings: Vec::new(),
            memory: None,
            commitment: prior.commitment,
            class: None,
            prior: prior.clone(),
            kv_rows: facts.kv_rows,
        }
    }

    /// Whether DFlash2 is selected.
    pub(crate) fn selected(&self) -> bool {
        self.plan.is_some()
    }

    /// Whether a load that fails on this selection is retried on the prior
    /// route: only auto selects anything that it may also give up.
    pub(crate) fn falls_back_on_load_error(&self) -> bool {
        self.request == PolicyRequest::Auto && self.selected()
    }

    /// This selection with DFlash2 and the profile withdrawn: the model on
    /// the routes its environment alone selects, with their commitment.
    pub(crate) fn without_dflash2(&self, reason: impl Into<String>) -> Self {
        Self {
            request: self.request,
            plan: None,
            routes: Arc::clone(&self.prior.routes),
            reason: reason.into(),
            descriptor: self.descriptor,
            target_source: self.target_source.clone(),
            warnings: Vec::new(),
            memory: None,
            commitment: self.prior.commitment,
            class: self.class,
            prior: self.prior.clone(),
            kv_rows: self.kv_rows,
        }
    }

    /// The decision as `/hf2q/v1/runtime` publishes it
    /// (`measurement_snapshot.active_controls.dflash2`), so that a gate
    /// receipt binds the route that ran and not the environment that asked.
    pub(crate) fn published(&self) -> serde_json::Value {
        let switches: serde_json::Map<String, serde_json::Value> = self
            .routes
            .entries()
            .map(|(name, value, source, withheld)| {
                (
                    name.to_owned(),
                    serde_json::json!({"value": value, "source": source, "withheld": withheld}),
                )
            })
            .collect();
        serde_json::json!({
            "policy": self.request.label(),
            "selected": self.selected(),
            "reason": self.reason,
            "descriptor": self.descriptor.map(|descriptor| descriptor.id),
            "qualification": self.plan.as_ref().map(|plan| plan.class.label()),
            "target_source": self.target_source,
            "drafter_source": self.plan.as_ref().map(|plan| plan.source.label()),
            "profile": self.profile_label(),
            "switches": switches,
            "memory": {
                "model_file_bytes": self.commitment.model_file_bytes,
                "kv_budget_bytes": self.commitment.kv_budget_bytes,
                "drafter_bytes": self.commitment.drafter_bytes,
                "drafter_load_peak_with_mapping_extent_bytes":
                    self.memory.map(|memory| memory.drafter_load_bytes),
                "drafter_run_payload_peak_bytes":
                    self.memory.map(|memory| memory.drafter_run_bytes),
                "drafter_accounting_scope": "requested_payload_and_load_mapping_extent",
                "process_reserve_bytes": MEMORY_RESERVE_BYTES,
                "prepared_plane_bytes": self.commitment.plane_bytes,
                "scratch_bytes": self.commitment.scratch_bytes,
                "headroom_bytes": self.memory.map(|memory| memory.headroom_bytes),
                "present_headroom_bytes":
                    self.memory.map(|memory| memory.present_headroom_bytes),
                "available_bytes": self.memory.map(|memory| memory.available_bytes),
                "resident_bytes": self.memory.map(|memory| memory.resident_bytes),
                "kv_rows": self.kv_rows,
            },
        })
    }

    /// `full`, `partial` (some profile values withheld) or `none`.
    fn profile_label(&self) -> &'static str {
        match (self.routes.profile(), self.routes.any_withheld()) {
            (None, _) => "none",
            (Some(_), false) => "full",
            (Some(_), true) => "partial",
        }
    }

    /// Emit the startup audit record and any warning.
    pub(crate) fn log(&self, model: &Path) {
        for warning in &self.warnings {
            tracing::warn!(model = %model.display(), "{warning}");
        }
        let memory = self.memory.unwrap_or_default();
        // What the engine holds on the routes as resolved: also the planes
        // of a prepared route the operator set beside the kill switch.
        let commitment = self.commitment;
        tracing::info!(
            model = %model.display(),
            policy = self.request.label(),
            selected = self.selected(),
            reason = %self.reason,
            descriptor = self.descriptor.map_or("none", |descriptor| descriptor.id),
            target_source = %self.target_source,
            drafter_source = self.plan.as_ref().map_or("none", |plan| plan.source.label()),
            profile = self.profile_label(),
            withheld = %self.routes.withheld_audit(),
            memory_headroom_bytes = memory.headroom_bytes,
            present_headroom_bytes = memory.present_headroom_bytes,
            available_bytes = memory.available_bytes,
            resident_bytes = memory.resident_bytes,
            drafter_bytes = commitment.drafter_bytes,
            drafter_load_peak_with_mapping_extent_bytes = memory.drafter_load_bytes,
            drafter_run_payload_peak_bytes = memory.drafter_run_bytes,
            prepared_plane_bytes = commitment.plane_bytes,
            scratch_bytes = commitment.scratch_bytes,
            switches = %self.routes.audit(),
            "DFlash2 policy resolved"
        );
    }
}

enum Unqualified {
    /// DFlash2 has no serving path here at all; `HF2Q_DFLASH2=1` was always
    /// ignored in this configuration and still is.
    NotServed(String),
    /// A served configuration that does not qualify; fails startup when
    /// DFlash2 is required.
    Refused(String),
}

/// The pinned descriptor `facts` qualifies for and the class that qualifies
/// it, or why there is none.
fn qualify(
    facts: &TargetFacts,
) -> std::result::Result<(&'static DFlash2CheckpointDescriptor, QualifiedClass), Unqualified> {
    if !matches!(facts.architecture.as_str(), "qwen35" | "qwen35moe") {
        return Err(Unqualified::NotServed(format!(
            "architecture {:?} has no qualified DFlash2 pair",
            facts.architecture
        )));
    }
    let Some(slots) = facts.slots else {
        return Err(Unqualified::NotServed(
            "the scheduler is not SlotAware".to_owned(),
        ));
    };
    if !(1..=MAX_DFLASH2_SLOTS).contains(&slots) {
        return Err(Unqualified::Refused(format!(
            "DFlash2 supports 1..={MAX_DFLASH2_SLOTS} physical slots, got {slots}"
        )));
    }
    let mut candidates = KNOWN_DFLASH2_CHECKPOINTS.iter().filter(|descriptor| {
        matches!(
            descriptor.family_adapter,
            DFlash2FamilyAdapter::QwenDense | DFlash2FamilyAdapter::QwenMoe
        ) && descriptor.target_architecture == facts.architecture
    });
    if let Provenance::Hf2q { source_sha256, .. } = &facts.provenance {
        if let Some(descriptor) = candidates
            .clone()
            .find(|descriptor| descriptor.target_source_sha256 == Some(source_sha256.as_str()))
        {
            return Ok((descriptor, QualifiedClass::PinnedPair));
        }
    }
    let Some(geometry) = facts.geometry.as_ref() else {
        return Err(Unqualified::Refused(
            "the GGUF header does not describe a complete Qwen target".to_owned(),
        ));
    };
    let mut nearest = None;
    let matched = candidates.find(|descriptor| {
        let Some(pinned) = descriptor.target_geometry else {
            return false;
        };
        match geometry.mismatch(pinned) {
            None => true,
            Some(difference) => {
                nearest = Some(format!("{}: {difference}", descriptor.id));
                false
            }
        }
    });
    match matched {
        Some(descriptor) => Ok((descriptor, QualifiedClass::SameShapes)),
        None => Err(Unqualified::Refused(format!(
            "the target has neither the source identity nor the shapes of a pinned DFlash2              target ({})",
            nearest.unwrap_or_else(|| "no descriptor for this architecture".to_owned())
        ))),
    }
}

fn k_quant(ggml_type: GgmlType) -> bool {
    matches!(ggml_type, GgmlType::Q4_K | GgmlType::Q6_K)
}

/// Why the batched MoE route does not admit this target, if it does not:
/// the conditions under which its layers would keep the per-row route (and
/// the staged routes would then fail the load) or the load itself fails.
fn moe_batched_refusal(facts: &TargetFacts) -> Option<String> {
    if facts.dwq_overlay {
        return Some("a DWQ overlay is applied to the MoE target".to_owned());
    }
    let layers = facts
        .geometry
        .as_ref()
        .map_or(0, |geometry| geometry.layers);
    for block in 0..layers {
        let tensor = |leaf: &str| {
            let name = format!("blk.{block}.{leaf}.weight");
            facts
                .tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .ok_or_else(|| format!("{name} is missing"))
        };
        let check = || -> std::result::Result<(), String> {
            for leaf in ["ffn_gate_inp", "ffn_gate_inp_shexp"] {
                let tensor = tensor(leaf)?;
                if tensor.ggml_type != GgmlType::F32 {
                    return Err(format!(
                        "{} is {:?}, not F32",
                        tensor.name, tensor.ggml_type
                    ));
                }
            }
            for leaf in ["ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"] {
                let tensor = tensor(leaf)?;
                if !k_quant(tensor.ggml_type) {
                    return Err(format!(
                        "{} is {:?}, not Q4_K or Q6_K",
                        tensor.name, tensor.ggml_type
                    ));
                }
            }
            if tensor("ffn_gate_shexp")?.ggml_type != tensor("ffn_up_shexp")?.ggml_type {
                return Err(format!(
                    "blk.{block} shared gate and up are stored differently"
                ));
            }
            Ok(())
        };
        if let Err(reason) = check() {
            return Some(reason);
        }
    }
    None
}

/// Why the staged MoE expert route does not admit this target, if it does
/// not: the conditions that fail a load when the switch is set by hand.
fn moe_staged_refusal(facts: &TargetFacts) -> Option<String> {
    let geometry = facts.geometry.as_ref()?;
    if !matches!(geometry.moe_top_k, 1 | 6 | 8) {
        return Some(format!(
            "needs 1, 6 or 8 routed experts per token, got {}",
            geometry.moe_top_k
        ));
    }
    for block in 0..geometry.layers {
        for leaf in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
            let name = format!("blk.{block}.{leaf}.weight");
            let Some(tensor) = facts.tensors.iter().find(|tensor| tensor.name == name) else {
                return Some(format!("{name} is missing"));
            };
            if !k_quant(tensor.ggml_type) {
                return Some(format!(
                    "{} is {:?}, not Q4_K or Q6_K",
                    tensor.name, tensor.ggml_type
                ));
            }
            let aligned = tensor.shape.len() >= 2
                && tensor.shape[tensor.shape.len() - 2..]
                    .iter()
                    .all(|extent| extent % 64 == 0);
            if !aligned {
                return Some(format!(
                    "{} shape {:?} is not 64-aligned",
                    tensor.name, tensor.shape
                ));
            }
        }
    }
    None
}

/// Why a row of the profile is not applied to this artifact, if it is not.
/// A row is withheld when its kernels would fail the load on this storage,
/// or when the weights its planes cover exist and none has storage they
/// take. A weight kind stored partly in admitted types keeps its row: the
/// kernels admit weight by weight.
pub(crate) fn switch_refusal(facts: &TargetFacts, row: &SwitchRow) -> Option<String> {
    match row.name {
        "HF2Q_QWEN_MOE_BATCHED" => return moe_batched_refusal(facts),
        // The staged experts run inside the batched layer: they need the
        // storage the batched route admits, not only its switch. A layer
        // that keeps the per-row route fails a staged load.
        "HF2Q_QWEN_MOE_STAGED_M8" => {
            return moe_batched_refusal(facts).or_else(|| moe_staged_refusal(facts))
        }
        "HF2Q_QWEN_PREFILL_PREPARED" if facts.chunk_scan_prefill => {
            return Some("the chunk-scan prefill investigation route is on".to_owned())
        }
        // The staged draft reads the head's planes only when it has them and
        // runs without them otherwise.
        "HF2Q_DFLASH2_DRAFT_STAGED" => return None,
        _ => {}
    }
    if row.planes.is_empty() {
        return None;
    }
    let covered = |tensor: &&TensorFact| {
        plane_set_of(&tensor.name).is_some_and(|set| row.planes.contains(&set))
    };
    let mut stored: Vec<String> = Vec::new();
    for tensor in facts.tensors.iter().filter(covered) {
        if prepared_plane_bytes(tensor.ggml_type, 256) > 0 {
            return None;
        }
        stored.push(format!("{:?}", tensor.ggml_type));
    }
    // A target without such weights (a MoE target has no dense FFN) leaves
    // the row inert, and other rows may require it.
    if stored.is_empty() {
        return None;
    }
    stored.sort();
    stored.dedup();
    Some(format!(
        "the prepared kernels take Q4_K and Q6_K; these weights are {}",
        stored.join("/")
    ))
}

/// Bytes of the prepared planes of one Q4_K or Q6_K weight: the layout of
/// mlx-native `prepare_repacked_gguf_weight`. Per 32 weights a Q4_K weight
/// keeps 16 bytes of codes and a Q6_K weight 24; per 256 weights both keep
/// scale metadata, 16 and 20 bytes. Other storage has no prepared planes.
pub(crate) fn prepared_plane_bytes(ggml_type: GgmlType, elements: u64) -> u64 {
    let elements = u128::from(elements);
    let bytes = match ggml_type {
        GgmlType::Q4_K => elements / 32 * 16 + elements / 256 * 16,
        GgmlType::Q6_K => elements / 32 * 24 + elements / 256 * 20,
        _ => 0,
    };
    u64::try_from(bytes).unwrap_or(u64::MAX)
}

fn plane_set_of(name: &str) -> Option<PlaneSet> {
    if name == "output.weight" {
        return Some(PlaneSet::Head);
    }
    if !name.starts_with("blk.") {
        return None;
    }
    let leaf = name.rsplit('.').nth(1)?;
    if !name.ends_with(".weight") || name.split('.').count() != 4 {
        return None;
    }
    match leaf {
        "ffn_gate" | "ffn_up" | "ffn_down" => Some(PlaneSet::DenseFfn),
        "attn_qkv" | "attn_gate" | "ssm_out" | "ssm_alpha" | "ssm_beta" | "attn_q" | "attn_k"
        | "attn_v" | "attn_output" => Some(PlaneSet::Projections),
        "ffn_gate_exps" | "ffn_up_exps" | "ffn_down_exps" => Some(PlaneSet::MoeExperts),
        "ffn_gate_shexp" | "ffn_up_shexp" | "ffn_down_shexp" => Some(PlaneSet::MoeShared),
        _ => None,
    }
}

/// Retained prepared planes and the model cache's single execution root.
/// The fixed execution capacity covers every compatible model matrix, not
/// just the selected plane sets. Charge it once if a selected product can
/// split; head multiplicity applies only to immutable planes. MM-ID storage
/// is separate. Alpha/beta share one padded matrix and one new raw source.
pub(crate) fn prepared_storage_bytes<'a, I>(tensors: I, sets: &[PlaneSet], head_copies: u64) -> u64
where
    I: IntoIterator<Item = &'a TensorFact>,
    I::IntoIter: Clone,
{
    if !sets.iter().any(|set| *set != PlaneSet::Head || head_copies != 0) {
        return 0;
    }
    let may_need_execution = sets.iter().any(|set| {
        matches!(set, PlaneSet::DenseFfn | PlaneSet::Projections)
            || (*set == PlaneSet::Head && head_copies != 0)
    });
    let tensors = tensors.into_iter();
    let tied_head = !tensors.clone().any(|tensor| tensor.name == "output.weight");
    let mut execution_maximum = 0;
    let mut selected_execution = false;
    let planes = tensors.clone().map(|tensor| {
        let set = if tied_head && tensor.name == "token_embd.weight" {
            Some(PlaneSet::Head)
        } else {
            plane_set_of(&tensor.name)
        };
        let Some(set) = set else { return 0 };
        let copies = if sets.contains(&set) {
            if set == PlaneSet::Head { head_copies } else { 1 }
        } else {
            0
        };
        if copies == 0 && !may_need_execution { return 0; }
        if !k_quant(tensor.ggml_type) {
            return 0;
        }
        if matches!(set, PlaneSet::MoeExperts | PlaneSet::MoeShared) {
            if copies == 0 { return 0; }
            return tensor.shape.iter().try_fold(1u64, |n, &d| n.checked_mul(d as u64))
                .map_or(u64::MAX, |n| prepared_plane_bytes(tensor.ggml_type, n));
        }
        let [n, k] = tensor.shape.as_slice() else { return u64::MAX };
        if tensor.name.ends_with(".ssm_beta.weight") {
            return 0;
        }
        let (rows, source) = if let Some(prefix) = tensor.name.strip_suffix("ssm_alpha.weight") {
            let beta = tensors.clone().find(|other| {
                other.name.strip_suffix("ssm_beta.weight") == Some(prefix)
            });
            if !beta.is_some_and(|beta| beta.ggml_type == tensor.ggml_type && beta.shape == tensor.shape) {
                return 0; // no fused prepared product can run
            }
            let Some(rows) = n.checked_mul(2).and_then(|n| n.checked_add(63)).map(|n| n / 64 * 64)
                else { return u64::MAX };
            let source = (rows as u64).checked_mul(*k as u64 / 256)
                .and_then(|n| n.checked_mul(u64::from(tensor.ggml_type.block_bytes())))
                .unwrap_or(u64::MAX);
            (rows, source)
        } else {
            (*n, 0)
        };
        let scratch = prepared_matrix_execution_bytes(rows, *k, tensor.ggml_type);
        execution_maximum = execution_maximum.max(scratch);
        selected_execution |= copies != 0 && scratch != 0;
        if copies == 0 { return 0; }
        prepared_matrix_bytes(rows, *k, tensor.ggml_type)
            .saturating_add(source)
            .saturating_mul(copies)
    }).fold(0, u64::saturating_add);
    planes.saturating_add(if selected_execution { execution_maximum } else { 0 })
}

fn prepared_matrix_bytes(n: usize, k: usize, ggml_type: GgmlType) -> u64 {
    // Ineligible weights keep the stock kernel and retain no prepared planes.
    if n % 64 != 0 || k % 256 != 0 {
        return 0;
    }
    let (Ok(n), Ok(k)) = (u32::try_from(n), u32::try_from(k)) else { return u64::MAX };
    mlx_native::ops::repacked_gguf_mm::repacked_gguf_weight_plane_bytes(n, k, ggml_type)
        .map_or(u64::MAX, |bytes| bytes as u64)
}

fn prepared_matrix_execution_bytes(n: usize, k: usize, ggml_type: GgmlType) -> u64 {
    if n % 64 != 0 || k % 256 != 0 {
        return 0;
    }
    let (Ok(n), Ok(k)) = (u32::try_from(n), u32::try_from(k)) else { return u64::MAX };
    use mlx_native::ops::repacked_gguf_mm::{
        repacked_gguf_mm_scratch_bytes, RepackedGgufMmExecution,
    };
    // Auto(1) is the maximum grid tile: every grid call uses N*32*splits*4;
    // larger serial and explicit OnePass products need no execution storage.
    repacked_gguf_mm_scratch_bytes(n, k, ggml_type, 1, RepackedGgufMmExecution::Auto)
        .map_or(u64::MAX, |bytes| bytes as u64)
}

/// Bytes of the prepared planes `routes` keep resident on this target.
/// `drafter` says whether a drafter is loaded beside the model.
///
/// The model's own planes follow its switches alone: a prepared route the
/// operator set keeps them with or without a drafter, also beside
/// `HF2Q_DFLASH2=0`. A plane set is prepared once however many switches read
/// it, with one exception: the output head. The staged draft (a drafter at
/// four lanes) and the prepared target head each prepare a copy of their own
/// unless `HF2Q_REPACKED_GGUF_SHARED_HEAD` makes the model own one for both.
/// Their dispatches share one model-cache execution root, even when head
/// planes are private. Its capacity also covers unselected compatible weights.
pub(crate) fn routed_plane_bytes(
    facts: &TargetFacts,
    routes: &ResolvedRoutes,
    drafter: bool,
) -> u64 {
    const STAGED: Switch = switch("HF2Q_DFLASH2_DRAFT_STAGED");
    const TARGET_HEAD: Switch = switch("HF2Q_REPACKED_GGUF_M32_HEAD");
    const SHARED_HEAD: Switch = switch("HF2Q_REPACKED_GGUF_SHARED_HEAD");
    let mut sets = routes.plane_sets();
    // The staged draft projects its vocabulary on the prepared tile only
    // at 32 draft rows: four lanes.
    let consumers = u64::from(drafter && routes.is_on(STAGED) && facts.slots == Some(4))
        + u64::from(routes.is_on(TARGET_HEAD));
    let copies = if routes.is_on(SHARED_HEAD) {
        consumers.min(1)
    } else {
        consumers
    };
    if copies != 0 && !sets.contains(&PlaneSet::Head) {
        sets.push(PlaneSet::Head);
    }
    // Target preparation walks model.layers only. MTP is loaded separately
    // from native blocks and never receives these prepared target products.
    // An unknown layer count cannot justify excluding any header weight.
    let target_layers = facts.qwen_config.as_ref().map(|config| config.num_hidden_layers);
    let tensors = facts.tensors.iter().filter(|tensor| {
        let layer = tensor.name.strip_prefix("blk.")
            .and_then(|name| name.split_once('.'))
            .and_then(|(index, _)| index.parse::<u32>().ok());
        !matches!((target_layers, layer), (Some(count), Some(index)) if index >= count)
    });
    prepared_storage_bytes(tensors, &sets, copies)
}

/// Worker content quantum; a folded cue is accounted separately by its owner.
fn prefill_content_rows(facts: &TargetFacts) -> Option<usize> {
    match facts.slots {
        Some(slots) => Some(
            crate::serve::api::engine::qwen35_slot_prefill_chunk_tokens(slots.max(1) as usize)
                as usize,
        ),
        None => facts.kv_rows,
    }
}

/// Bytes of scratch the kernels of the routes `routes` selects retain
/// between steps on the target `facts` describes.
///
/// The one place a route's retained scratch enters memory admission: the
/// memory rule, the commitment, the pool's charge and the published
/// decision carry what this returns, with or without a drafter. It takes the
/// resolved routes as a whole, not one switch at a time, because routes can
/// share a pool: their scratch is then one figure, not a sum. A bound that
/// grows with a slot's KV cache takes the cache's capacity
/// ([`TargetFacts::kv_rows`]) and the slot count ([`TargetFacts::slots`]),
/// never the rows in use. Charge the maximum legal ordinary-solo, folded-solo
/// and aggregate-joint lifetime with one prior-route TLS high-water. The solo
/// scheduler may append its bounded cue beyond the worker's content quantum;
/// the joint scheduler instead includes that cue in its aggregate quantum.
pub(crate) fn route_scratch_bytes(facts: &TargetFacts, routes: &ResolvedRoutes) -> u64 {
    if !matches!(facts.architecture.as_str(), "qwen35" | "qwen35moe") {
        return 0;
    }
    let slots = facts.slots.unwrap_or(1).max(1) as usize;
    let joint = slots > 1 && routes.is_on(switch("HF2Q_QWEN_PREFILL_JOINT"));
    let fold_cue = routes.is_on(switch("HF2Q_QWEN_PREFILL_FOLD_CUE"));
    let Ok(wait_layers) =
        crate::inference::models::qwen35::prefill_prepared::parse_wait_layers(
            routes.value(switch("HF2Q_QWEN_PREFILL_WAIT_LAYERS")),
        )
    else {
        return u64::MAX;
    };
    let has_mtp = facts.qwen_config.as_ref().is_some_and(|config| config.mtp_num_hidden_layers > 0);
    if !joint && !fold_cue && wait_layers == 1 && !has_mtp {
        return 0;
    }
    let (Some(config), Some(kv_rows)) = (facts.qwen_config.as_ref(), facts.kv_rows) else {
        // An unknown bound must never admit a large model as zero scratch.
        return u64::MAX;
    };
    let Some(forward_rows) = prefill_content_rows(facts) else {
        return u64::MAX;
    };
    use crate::inference::models::qwen35::{prefill_memory, resume_attn_fast};
    // A switch alone does not prove its shape/type gate admits every
    // projection. Otherwise a valid stock-kernel fallback would be charged
    // the smaller prepared request bound.
    let prepared = routes.is_on(switch("HF2Q_QWEN_PREFILL_PREPARED"))
        && !facts.chunk_scan_prefill && !facts.dwq_overlay
        && crate::inference::models::qwen35::forward_gpu::layer_scratch_resets()
        && crate::inference::models::qwen35::decode_pool::default_pool()
        // A forced switch does not make the loader admit legacy expert storage.
        && (config.moe.is_none()
            || (routes.is_on(switch("HF2Q_QWEN_MOE_BATCHED"))
                && moe_batched_refusal(facts).is_none()))
        && config.linear_key_head_dim % 32 == 0 && config.linear_value_head_dim % 32 == 0
        && facts.tensors.iter().filter(|tensor| {
            matches!(plane_set_of(&tensor.name), Some(PlaneSet::DenseFfn | PlaneSet::Projections))
        }).all(|tensor| {
            let [n, k] = tensor.shape.as_slice() else { return false };
            let alpha_beta = tensor.name.ends_with(".ssm_alpha.weight")
                || tensor.name.ends_with(".ssm_beta.weight");
            if alpha_beta {
                let (from, to) = if tensor.name.ends_with(".ssm_alpha.weight") {
                    ("ssm_alpha.weight", "ssm_beta.weight")
                } else {
                    ("ssm_beta.weight", "ssm_alpha.weight")
                };
                let prefix = tensor.name.strip_suffix(from).unwrap();
                if !facts.tensors.iter().any(|candidate| candidate.name.strip_suffix(to) == Some(prefix)
                    && candidate.ggml_type == tensor.ggml_type && candidate.shape == tensor.shape)
                {
                    return false;
                }
            }
            k_quant(tensor.ggml_type) && *k % 256 == 0 && (alpha_beta || *n % 64 == 0)
        });
    let route = prefill_memory::PrefillMemoryRoute {
        joint, wait_layers, prepared,
        short_wait_layers: prefill_memory::short_wait_layers(),
        tq_only: crate::serve::api::tq_packed_descriptor::is_tq_active_mode(),
        resume_fast: resume_attn_fast::resume_attn_fast_enabled(),
    };
    let Ok(forward) = prefill_memory::envelope(
        config, forward_rows, slots, kv_rows, route, fold_cue,
    ) else {
        return u64::MAX;
    };
    if !has_mtp {
        return forward.total();
    }
    // MTP cannot fold a cue into its target transaction. Prior pooled roots
    // may still come from any earlier legal solo/folded/joint transaction.
    let mtp_forward = if fold_cue {
        match prefill_memory::envelope(config, forward_rows, slots, kv_rows, route, false) {
            Ok(bound) => bound.total(),
            Err(_) => return u64::MAX,
        }
    } else {
        forward.total()
    };
    if !crate::inference::models::qwen35::decode_pool::default_pool() {
        return u64::MAX;
    }
    let retained_cap =
        crate::inference::models::qwen35::prefill_prepared::PREFILL_RETAINED_SCRATCH_BYTES as u64;
    let prior_pool = if !joint && !fold_cue && wait_layers == 1 {
        // The ordinary target keeps its existing incremental baseline. This
        // is a DIFFERENT closed owner: serving resets/caps its pool after
        // the checked target drain and before any catch-up gather.
        retained_cap
    } else {
        match prefill_memory::retained_pool_bound_bytes(
            config, forward_rows, slots, kv_rows, route, fold_cue,
        ) {
            Ok(pool) => pool.min(retained_cap),
            Err(_) => return u64::MAX,
        }
    };
    use crate::inference::models::qwen35::carry_memory;
    let retained = carry_memory::retained_bound(config, forward_rows, slots);
    let carry = carry_memory::CatchupPlan::new(
        config, forward_rows, config.mtp_use_dedicated_embeddings,
    );
    // Forward bounds exclude the normalized parent; compose charges its
    // actual overlap with forward owners and with the complete catch-up.
    retained.saturating_add(forward.total())
        .max(carry.compose(mtp_forward, retained, prior_pool))
        .saturating_add(carry_memory::rope_cache_bound(forward_rows))
}

/// The memory rule's numbers.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MemoryEstimate {
    /// What the load could take: the smaller of the budget and the present
    /// bound ([`memory_verdict`]).
    pub(crate) headroom_bytes: u64,
    /// The present bound: what the host reported available, minus every
    /// claim on it ([`memory_verdict`]).
    pub(crate) present_headroom_bytes: u64,
    /// What the host reported available at the decision.
    pub(crate) available_bytes: u64,
    /// What the engines already charged have committed.
    pub(crate) resident_bytes: u64,
    /// Increment beside this route's target scratch, after whole-lifetime max.
    pub(crate) drafter_bytes: u64,
    pub(crate) drafter_load_bytes: u64,
    pub(crate) drafter_run_bytes: u64,
    /// Prepared planes of the routes as resolved.
    pub(crate) plane_bytes: u64,
    /// Retained scratch of the routes as resolved.
    pub(crate) scratch_bytes: u64,
}

/// What the memory rule allows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MemoryVerdict {
    /// Drafter and every profile plane fit.
    Full,
    /// Only the drafter fits: the profile's plane-carrying switches go.
    DropPlanes,
    /// Not even the drafter fits.
    NoDrafter,
}

/// The memory a load may take for its drafter, planes and scratch: the
/// smaller of two bounds (`serve::resident_memory` says why every term is in
/// both).
///
/// - present: `available now - charged engines - model file - KV budget -
///   reserve`;
/// - budget: `physical - charged engines - model file - KV budget - reserve`.
///
/// What is available now is still owed to everything that was promised
/// memory and has not taken it: the incoming model's file and KV budget, and
/// whatever the charged engines have not allocated of their commitments.
/// How much of a commitment an engine has allocated is not known, so the
/// whole of it is subtracted.
///
/// Compare complete route lifetimes, not a shared checkpoint-file proxy.
pub(crate) fn memory_verdict(
    host: HostFacts,
    model_file_bytes: u64,
    full_route_bytes: u64,
    drafter_only_route_bytes: u64,
) -> (MemoryVerdict, Headroom) {
    let after_claims = |bytes: u64| {
        bytes
            .saturating_sub(host.resident_bytes)
            .saturating_sub(model_file_bytes)
            .saturating_sub(host.kv_budget_bytes)
            .saturating_sub(MEMORY_RESERVE_BYTES)
    };
    let headroom = Headroom {
        budget_bytes: after_claims(host.physical_bytes),
        present_bytes: after_claims(host.available_bytes),
    };
    let verdict = if full_route_bytes <= headroom.bytes() {
        MemoryVerdict::Full
    } else if drafter_only_route_bytes <= headroom.bytes() {
        MemoryVerdict::DropPlanes
    } else {
        MemoryVerdict::NoDrafter
    };
    (verdict, headroom)
}

/// The two bounds of an admission.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct Headroom {
    /// Of the host's physical memory, what no claim is on.
    pub(crate) budget_bytes: u64,
    /// Of what the host has available now, what no claim is on.
    pub(crate) present_bytes: u64,
}

impl Headroom {
    /// What the load may take: within both bounds.
    pub(crate) fn bytes(self) -> u64 {
        self.budget_bytes.min(self.present_bytes)
    }
}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1_u64 << 30) as f64)
}

const MEMORY_REFUSAL: &str = "its prepared planes do not fit in memory";

/// Why this host cannot run a table switch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HostRefusal {
    /// The reason, for a profile row left off (`withheld=` in the audit).
    pub(crate) reason: String,
    /// What an operator who set the switch on by hand reads when the load is
    /// refused.
    pub(crate) explicit: String,
}

/// What this host's GPU can run (`inference::route_host`), asked before any
/// weight is read. The error is why the host could not be asked.
pub(crate) fn host_routes() -> std::result::Result<HostRoutes, String> {
    let device =
        mlx_native::MlxDevice::new().map_err(|error| format!("no Metal device: {error}"))?;
    route_host::probe(&device).map_err(|error| format!("{error:#}"))
}

/// Why `host` cannot run the kernels of the table switch `name`, if it
/// cannot: the one place the policy asks the host about a route, for every
/// row of the table before it is turned on. A host that could not be asked
/// runs no route that needs a tensor-only kernel family.
pub(crate) fn host_switch_refusal(
    host: &std::result::Result<HostRoutes, String>,
    name: &str,
) -> Option<HostRefusal> {
    match host {
        Ok(routes) => match routes.switch(name)? {
            HostRoute::Runs => None,
            route @ HostRoute::Unavailable(_) => Some(HostRefusal {
                reason: route.reason()?,
                explicit: routes.explicit_refusal(name, route),
            }),
        },
        Err(error) => {
            let row = route_host::SWITCHES.iter().find(|row| row.name == name)?;
            (!row.families.is_empty()).then(|| HostRefusal {
                reason: format!("the host's kernel support could not be established: {error}"),
                explicit: format!(
                    "{name} is on, but the host's kernel support could not be established: \
                     {error}. Unset {name}: the stock route serves this host."
                ),
            })
        }
    }
}

/// Decide the route of one model load. `env` reads one process environment
/// variable; `host_refusal` says why this host cannot run a table switch's
/// kernels, if it cannot ([`host_switch_refusal`]). An error is a startup
/// error: an invalid `HF2Q_DFLASH2`, a required DFlash2 that cannot be
/// satisfied, or a switch set on by hand that this host cannot run.
pub(crate) fn decide(
    facts: &TargetFacts,
    host: HostFacts,
    env: &dyn Fn(&str) -> Option<String>,
    store: &dyn DrafterStore,
    host_refusal: &dyn Fn(&str) -> Option<HostRefusal>,
) -> Result<Selection> {
    let request = parse_policy(env(DFLASH2_ENV).as_deref()).map_err(anyhow::Error::msg)?;
    // A switch the operator set on is the operator's: it is never withheld,
    // so a host that cannot run its kernels refuses the load here, by name,
    // instead of failing at the first dispatch. "On" is what the switch's
    // own reader takes as on: a value the reader leaves off selects no
    // kernel and refuses nothing.
    for row in SWITCHES {
        if turns_on(switch(row.name), env(row.name).as_deref()) {
            if let Some(refusal) = host_refusal(row.name) {
                bail!("{}", refusal.explicit);
            }
        }
    }
    let prior = PriorRoute::of(facts, host, env);
    if request == PolicyRequest::Off {
        return Ok(Selection::prior_route(
            request,
            facts,
            &prior,
            format!("{DFLASH2_ENV}=0 keeps the prior route"),
        ));
    }
    let required = request == PolicyRequest::Required;

    let (descriptor, class) = match qualify(facts) {
        Ok(qualified) => qualified,
        Err(Unqualified::Refused(reason)) if required => {
            bail!("{DFLASH2_ENV}=1 cannot be satisfied: {reason}")
        }
        Err(Unqualified::NotServed(reason)) | Err(Unqualified::Refused(reason)) => {
            let mut selection = Selection::prior_route(request, facts, &prior, reason.clone());
            if required {
                selection
                    .warnings
                    .push(format!("{DFLASH2_ENV}=1 has no effect: {reason}"));
            }
            return Ok(selection);
        }
    };
    let withdrawn = |reason: String, warnings: Vec<String>| {
        let mut selection = Selection::prior_route(request, facts, &prior, reason);
        selection.descriptor = Some(descriptor);
        selection.class = Some(class);
        selection.warnings = warnings;
        selection
    };
    let mut warnings = Vec::new();

    let named = |name: &str| {
        env(name)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    };
    // An explicit checkpoint is used when both variables name a file. Under
    // auto a stale or incomplete pair never prevents startup: it is reported
    // and the pinned drafter is resolved instead.
    let explicit = match (named(DFLASH2_CONFIG_ENV), named(DFLASH2_WEIGHTS_ENV)) {
        (None, None) => None,
        (Some(config_path), Some(weights_path)) => {
            match store.explicit_usable(&config_path, &weights_path) {
                Ok(()) => Some((config_path, weights_path)),
                Err(reason) if required => {
                    bail!("{DFLASH2_ENV}=1 cannot be satisfied: {reason}")
                }
                Err(reason) => {
                    warnings.push(format!(
                        "ignoring {DFLASH2_CONFIG_ENV} and {DFLASH2_WEIGHTS_ENV}: {reason}; \
                         resolving the pinned drafter"
                    ));
                    None
                }
            }
        }
        _ => {
            let message =
                format!("{DFLASH2_CONFIG_ENV} and {DFLASH2_WEIGHTS_ENV} are used only together");
            if required {
                bail!("{DFLASH2_ENV}=1: {message}");
            }
            warnings.push(format!(
                "{message}; ignoring the one that is set and resolving the pinned drafter"
            ));
            None
        }
    };
    let files = match explicit {
        Some((config_path, weights_path)) => DrafterFiles {
            config_path,
            weights_path,
            source: DrafterSource::Explicit,
        },
        None => match store.cached(descriptor) {
            Some((config_path, weights_path)) => DrafterFiles {
                config_path,
                weights_path,
                source: DrafterSource::Cache,
            },
            None => match store.download(descriptor) {
                Ok((config_path, weights_path)) => DrafterFiles {
                    config_path,
                    weights_path,
                    source: DrafterSource::Downloaded,
                },
                Err(reason) if required => bail!(
                    "{DFLASH2_ENV}=1 cannot be satisfied: DFlash2 drafter {}@{} is not in the \
                     Hugging Face cache and could not be downloaded: {reason}",
                    descriptor.repository,
                    descriptor.revision
                ),
                Err(reason) => {
                    warnings.push(format!(
                        "DFlash2 drafter unavailable repository={} revision={} reason={reason}; \
                         serving the prior route",
                        descriptor.repository, descriptor.revision
                    ));
                    return Ok(withdrawn(
                        format!("drafter unavailable: {reason}"),
                        warnings,
                    ));
                }
            },
        },
    };

    // The drafter is bound to this target by the adapter's own shape check,
    // here on the header and again on the loaded model at provisioning.
    if let Some(geometry) = facts.geometry.as_ref() {
        let shape = DFlash2TargetShape {
            architecture: &facts.architecture,
            hidden_size: geometry.hidden_size as usize,
            vocab_size: geometry.vocab_size as usize,
            num_hidden_layers: geometry.layers as usize,
        };
        if let Err(reason) = store.bind(descriptor, &files.config_path, shape) {
            if required {
                bail!(
                    "{DFLASH2_ENV}=1 cannot be satisfied: DFlash2 checkpoint {} does not bind \
                     to this target: {reason}",
                    descriptor.id
                );
            }
            return Ok(withdrawn(
                format!(
                    "DFlash2 checkpoint {} does not bind to this target: {reason}",
                    descriptor.id
                ),
                warnings,
            ));
        }
    }

    // The profile is applied switch by switch: a row this host cannot run,
    // or this artifact's storage does not admit, keeps its own default, and
    // the drafter serves with the rows that are left.
    let profile = RouteProfile::of(descriptor);
    let refusal = |row: &SwitchRow| {
        host_refusal(row.name)
            .map(|refusal| refusal.reason)
            .or_else(|| switch_refusal(facts, row))
    };
    let admitted = ResolvedRoutes::resolve(profile, &refusal, env);
    // A static qualification boundary, decided from the artifact's storage
    // before the load: auto qualifies a MoE target only when its batched MoE
    // verification route is admitted. Without it every row of a 32-row
    // verify step runs the per-row expert path, which is not a qualified
    // fast path: on the Qwen3.6 launcher default (Q5_K shared experts) the
    // exact four-agent Q5 took 48.58 and 49.13 s with the drafter against
    // 39.46 s on the prior route. Nothing here watches acceptance or output
    // at run time, and no route is switched after the load. A required
    // DFlash2 keeps the drafter and is told the cost.
    //
    // Admitted means what the loader will do, not what the switch says: the
    // loader keeps a layer on the per-row route when its storage is not the
    // batched route's, whoever turned the switch on. So the storage is asked
    // also when the operator set the switch by hand.
    const MOE_BATCHED: Switch = switch("HF2Q_QWEN_MOE_BATCHED");
    if descriptor.family_adapter == DFlash2FamilyAdapter::QwenMoe {
        let not_admitted = match moe_batched_refusal(facts) {
            Some(storage) if admitted.is_on(MOE_BATCHED) => Some(format!(
                "HF2Q_QWEN_MOE_BATCHED is set, but the artifact's storage keeps the per-row \
                 route: {storage}"
            )),
            Some(storage) => Some(format!("HF2Q_QWEN_MOE_BATCHED is off: {storage}")),
            None if admitted.is_on(MOE_BATCHED) => None,
            None => Some("HF2Q_QWEN_MOE_BATCHED is off: it is switched off".to_owned()),
        };
        if let Some(why) = not_admitted {
            let unqualified = format!(
                "the batched MoE verification route is not admitted ({why}), and DFlash2 \
                 verification on the per-row expert path is not a qualified fast path"
            );
            if !required {
                return Ok(withdrawn(unqualified, warnings));
            }
            warnings.push(format!(
                "{DFLASH2_ENV}=1 serves DFlash2 on an unqualified route: {unqualified}; it was \
                 measured slower than the prior route on such a target"
            ));
        }
    }
    let dropped = ResolvedRoutes::resolve(
        profile,
        &|row| refusal(row).or_else(|| (!row.planes.is_empty()).then(|| MEMORY_REFUSAL.to_owned())),
        env,
    );
    let plan_memory = |routes: &ResolvedRoutes| -> Result<_> {
        let drafter = DrafterMemoryBound::required(descriptor, facts, routes)?;
        Ok((
            drafter,
            routed_plane_bytes(facts, routes, true),
            route_scratch_bytes(facts, routes),
        ))
    };
    let ((admitted_drafter, admitted_planes, admitted_scratch),
         (dropped_drafter, dropped_planes, dropped_scratch)) =
        match (plan_memory(&admitted), plan_memory(&dropped)) {
            (Ok(admitted), Ok(dropped)) => (admitted, dropped),
            (Err(reason), _) | (_, Err(reason)) => {
                if required {
                    bail!("{DFLASH2_ENV}=1 cannot bind the drafter allocation bound: {reason}");
                }
                return Ok(withdrawn(
                    format!("drafter allocation bound unavailable: {reason}"),
                    warnings,
                ));
            }
        };
    let (verdict, headroom) = memory_verdict(
        host,
        facts.model_file_bytes,
        admitted_drafter.total(admitted_planes, admitted_scratch),
        dropped_drafter.total(dropped_planes, dropped_scratch),
    );
    let formula = format!(
        "headroom {} = the smaller of present {} (available {}) and budget {} (physical {}), \
         each after resident engines {}, model file {}, KV budget {} and reserve {}",
        gib(headroom.bytes()),
        gib(headroom.present_bytes),
        gib(host.available_bytes),
        gib(headroom.budget_bytes),
        gib(host.physical_bytes),
        gib(host.resident_bytes),
        gib(facts.model_file_bytes),
        gib(host.kv_budget_bytes),
        gib(MEMORY_RESERVE_BYTES)
    );
    let qualified = match class {
        QualifiedClass::PinnedPair => "pinned pair".to_owned(),
        QualifiedClass::SameShapes => {
            format!("same architecture and shapes as {}", descriptor.id)
        }
    };
    let estimate = |drafter: DrafterMemoryBound, plane_bytes, scratch_bytes| MemoryEstimate {
        headroom_bytes: headroom.bytes(),
        present_headroom_bytes: headroom.present_bytes,
        available_bytes: host.available_bytes,
        resident_bytes: host.resident_bytes,
        drafter_bytes: drafter.charge(scratch_bytes),
        drafter_load_bytes: drafter.load_bytes,
        drafter_run_bytes: drafter.run_bytes,
        plane_bytes,
        scratch_bytes,
    };
    let (routes, reason, memory) = match verdict {
        MemoryVerdict::Full => (
            admitted,
            qualified,
            estimate(admitted_drafter, admitted_planes, admitted_scratch),
        ),
        MemoryVerdict::NoDrafter if !required => {
            let mut selection = withdrawn(
                format!(
                    "{qualified}; the drafter ({}) does not fit in memory: {formula}",
                    gib(dropped_drafter.charge(dropped_scratch))
                ),
                warnings,
            );
            selection.memory = Some(estimate(dropped_drafter, dropped_planes, dropped_scratch));
            return Ok(selection);
        }
        MemoryVerdict::DropPlanes | MemoryVerdict::NoDrafter => {
            if verdict == MemoryVerdict::NoDrafter {
                warnings.push(format!(
                    "{DFLASH2_ENV}=1 keeps the drafter ({}) although it does not fit: {formula}",
                    gib(dropped_drafter.charge(dropped_scratch))
                ));
            }
            (
                dropped,
                format!(
                    "{qualified}; prepared planes ({}) dropped from the profile, drafter {} \
                     kept: {formula}",
                    gib(admitted_planes),
                    gib(dropped_drafter.charge(dropped_scratch))
                ),
                estimate(dropped_drafter, dropped_planes, dropped_scratch),
            )
        }
    };
    let commitment = route_commitment(facts, host, &routes, Some(memory.drafter_bytes));
    Ok(Selection {
        request,
        plan: Some(DFlash2Plan {
            descriptor,
            class,
            config_path: files.config_path,
            weights_path: files.weights_path,
            source: files.source,
        }),
        routes: Arc::new(routes),
        reason,
        descriptor: Some(descriptor),
        target_source: facts.source_identity(),
        warnings,
        memory: Some(memory),
        commitment,
        class: Some(class),
        prior,
        kv_rows: facts.kv_rows,
    })
}

/// Upper bound of what an engine loaded from `path` holds resident beside
/// its KV budget: the model file, the planes and scratch of the routes its
/// environment selects and, when the target qualifies for a drafter, the
/// drafter and every prepared plane and scratch byte its storage and this
/// host admit. The hot-swap pool admits a load against
/// this before the loader allocates anything. `config` is the engine
/// configuration when the caller has one; without it the target is taken
/// to be served SlotAware with four slots.
pub(crate) fn admission_bytes(
    path: &Path,
    config: Option<&crate::serve::multi_model::EngineConfig>,
) -> Option<u64> {
    let gguf = GgufFile::open(path).ok()?;
    let facts = TargetFacts::read(
        &gguf,
        path,
        config.map_or(EngineMode::SlotAware { max_slots: 4 }, |config| {
            config.engine_mode
        }),
        config.is_some_and(|config| config.dwq_overlay_path.is_some()),
        config.and_then(|config| config.requested_context),
    );
    let env = route_profile::process_value;
    let host = host_routes();
    Some(
        facts
            .model_file_bytes
            .saturating_add(route_upper_bound(&facts, &env, &|name| {
                host_switch_refusal(&host, name)
            })),
    )
}

/// Drafter, plane and scratch bytes of the fullest route the policy could
/// select for `facts`: no drafter lookup and no memory rule. Without a
/// drafter (the kill switch, a target that does not qualify) that is the
/// route the environment alone selects, whose planes are resident whatever
/// the policy.
pub(crate) fn route_upper_bound(
    facts: &TargetFacts,
    env: &dyn Fn(&str) -> Option<String>,
    host_refusal: &dyn Fn(&str) -> Option<HostRefusal>,
) -> u64 {
    let route_bytes = |routes: &ResolvedRoutes, drafter: bool| {
        routed_plane_bytes(facts, routes, drafter)
            .saturating_add(route_scratch_bytes(facts, routes))
    };
    let prior = route_bytes(&ResolvedRoutes::resolve(None, &|_| None, env), false);
    let request = match parse_policy(env(DFLASH2_ENV).as_deref()) {
        Ok(request @ (PolicyRequest::Auto | PolicyRequest::Required)) => request,
        _ => return prior,
    };
    let Ok((descriptor, _class)) = qualify(facts) else {
        return prior;
    };
    let profile = RouteProfile::of(descriptor);
    let refusal = |row: &SwitchRow| {
        host_refusal(row.name)
            .map(|refusal| refusal.reason)
            .or_else(|| switch_refusal(facts, row))
    };
    let routes = ResolvedRoutes::resolve(profile, &refusal, env);
    let dropped = ResolvedRoutes::resolve(
        profile,
        &|row| refusal(row).or_else(|| (!row.planes.is_empty()).then(|| MEMORY_REFUSAL.to_owned())),
        env,
    );
    let planned = |routes: &ResolvedRoutes| -> Result<_> {
        let drafter = DrafterMemoryBound::required(descriptor, facts, routes)?;
        let bytes = drafter.total(
            routed_plane_bytes(facts, routes, true),
            route_scratch_bytes(facts, routes),
        );
        // A refused/unrepresentable route is not a byte-count candidate.
        // decide cannot select its full profile; it uses a finite dropped
        // profile or withdraws to the prior route instead.
        Ok((bytes != u64::MAX).then_some(bytes))
    };
    match (planned(&routes), planned(&dropped)) {
        (Ok(full), Ok(partial)) => {
            if request == PolicyRequest::Required && partial.is_none() {
                return u64::MAX;
            }
            prior.max(full.unwrap_or(prior)).max(partial.unwrap_or(prior))
        }
        // `decide` withdraws the same invalid plan in auto mode before load.
        _ if request == PolicyRequest::Auto => prior,
        _ => u64::MAX,
    }
}

/// [`decide`] for the model at `path` on this host and process environment.
/// `resident_bytes` is what the engines already resident have committed.
pub(crate) fn resolve_for_load(
    path: &Path,
    config: &crate::serve::multi_model::EngineConfig,
    resident_bytes: u64,
) -> Result<Selection> {
    let facts = {
        let gguf = GgufFile::open(path)
            .map_err(|error| anyhow::anyhow!("GGUF header parse failed: {error}"))?;
        TargetFacts::read(
            &gguf,
            path,
            config.engine_mode,
            config.dwq_overlay_path.is_some(),
            config.requested_context,
        )
    };
    // Asking the host here also seeds the process's host answers before any
    // prefill scope or drafter load reads them.
    let host = host_routes();
    decide(
        &facts,
        HostFacts::read(config.kv_cache_budget_bytes, resident_bytes),
        &route_profile::process_value,
        &HubDrafterStore,
        &|name| host_switch_refusal(&host, name),
    )
    .and_then(|selection| unprobed_host(selection, host.as_ref().err().map(String::as_str)))
    .with_context(|| format!("DFlash2 policy for {}", path.display()))
}

/// What a host that could not be asked about its kernels does to a
/// selection. Auto does not load a drafter on such a host: it logs the
/// failure and keeps the prior route, whose load asks nothing of the probe.
/// A required DFlash2 surfaces the failure.
pub(crate) fn unprobed_host(selection: Selection, probe_error: Option<&str>) -> Result<Selection> {
    let Some(error) = probe_error.filter(|_| selection.selected()) else {
        return Ok(selection);
    };
    if selection.request == PolicyRequest::Required {
        bail!(
            "{DFLASH2_ENV}=1 cannot be satisfied: the host's kernel support could not be \
             established: {error}"
        );
    }
    let mut prior = selection.without_dflash2(format!(
        "the host's kernel support could not be established: {error}"
    ));
    prior.warnings.push(format!(
        "DFlash2 not selected: the host's kernel support could not be established: {error}; \
         serving the prior route"
    ));
    Ok(prior)
}

#[cfg(test)]
#[path = "dflash2_policy_tests.rs"]
pub(crate) mod tests;
