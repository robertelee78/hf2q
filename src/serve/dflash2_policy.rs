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
use crate::inference::spec_decode::dflash2::config::DFlash2TargetShape;
use crate::inference::spec_decode::dflash2::route_profile::{
    PlaneSet, ResolvedRoutes, RouteProfile, SwitchRow,
};
use crate::inference::spec_decode::dflash2::schema::{
    DFlash2CheckpointDescriptor, DFlash2FamilyAdapter, QwenTargetGeometry,
    KNOWN_DFLASH2_CHECKPOINTS,
};
use crate::inference::spec_decode::dflash2::weights::DFlash2WeightsFile;
use crate::serve::api::engine::EngineMode;

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
    pub(crate) model_file_bytes: u64,
    pub(crate) tensors: Vec<TensorFact>,
    /// Architecture, dimensions, bound tensor shapes and tokenizer identity;
    /// `None` when the header does not describe a Qwen target.
    pub(crate) geometry: Option<QwenTargetGeometry>,
    /// Whether a DWQ overlay is applied on load.
    pub(crate) dwq_overlay: bool,
    /// Whether the chunk-scan prefill investigation route is on; the
    /// prepared prefill refuses to run beside it.
    pub(crate) chunk_scan_prefill: bool,
}

/// SHA-256 over the tokenizer pre-type, vocabulary and merges of a GGUF:
/// for each section its name, its entry count and every entry
/// length-prefixed (all counts little-endian `u64`).
pub(crate) fn tokenizer_identity(gguf: &GgufFile) -> Option<String> {
    use mlx_native::gguf::MetadataValue;
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let mut section = |name: &str, entries: &[&str]| {
        hasher.update(name.as_bytes());
        hasher.update((entries.len() as u64).to_le_bytes());
        for entry in entries {
            hasher.update((entry.len() as u64).to_le_bytes());
            hasher.update(entry.as_bytes());
        }
    };
    let strings = |key: &str| -> Option<Vec<&str>> {
        match gguf.metadata(key)? {
            MetadataValue::Array(values) => values
                .iter()
                .map(|value| match value {
                    MetadataValue::String(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    };
    section("pre", &[gguf.metadata_string("tokenizer.ggml.pre")?]);
    section("tokens", &strings("tokenizer.ggml.tokens")?);
    section("merges", &strings("tokenizer.ggml.merges").unwrap_or_default());
    Some(hex::encode(hasher.finalize()))
}

/// The geometry of a Qwen target, from its GGUF header alone.
pub(crate) fn read_geometry(gguf: &GgufFile) -> Option<QwenTargetGeometry> {
    let architecture = gguf.metadata_string("general.architecture")?;
    let config = crate::inference::models::qwen35::Qwen35Config::from_gguf(gguf).ok()?;
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
        tokenizer_sha256: tokenizer_identity(gguf)?.into(),
    })
}

impl TargetFacts {
    /// Read the facts from the GGUF header and the engine configuration.
    pub(crate) fn read(
        gguf: &GgufFile,
        path: &Path,
        engine_mode: EngineMode,
        dwq_overlay: bool,
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
        Self {
            provenance: crate::core::provenance::detect(gguf),
            slots: match engine_mode {
                EngineMode::SlotAware { max_slots } => Some(max_slots),
                _ => None,
            },
            model_file_bytes: std::fs::metadata(path).map_or(0, |metadata| metadata.len()),
            tensors,
            geometry: qwen.then(|| read_geometry(gguf)).flatten(),
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
    /// The configured aggregate KV budget; zero when none is configured.
    pub(crate) kv_budget_bytes: u64,
}

impl HostFacts {
    pub(crate) fn read(kv_budget_bytes: Option<u64>) -> Self {
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        Self {
            physical_bytes: system.total_memory(),
            kv_budget_bytes: kv_budget_bytes.unwrap_or(0),
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
}

impl Selection {
    fn prior_route(
        request: PolicyRequest,
        facts: &TargetFacts,
        reason: impl Into<String>,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Self {
        Self {
            request,
            plan: None,
            routes: Arc::new(ResolvedRoutes::resolve(None, &|_| None, env)),
            reason: reason.into(),
            descriptor: None,
            target_source: facts.source_identity(),
            warnings: Vec::new(),
            memory: None,
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

    /// This selection with DFlash2 and the profile withdrawn.
    pub(crate) fn without_dflash2(&self, reason: impl Into<String>) -> Self {
        Self {
            request: self.request,
            plan: None,
            routes: Arc::new(ResolvedRoutes::from_process_env()),
            reason: reason.into(),
            descriptor: self.descriptor,
            target_source: self.target_source.clone(),
            warnings: Vec::new(),
            memory: None,
        }
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
            drafter_bytes = memory.drafter_bytes,
            prepared_plane_bytes = memory.plane_bytes,
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
    let layers = facts.geometry.as_ref().map_or(0, |geometry| geometry.layers);
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
        "HF2Q_QWEN_MOE_STAGED_M8" => return moe_staged_refusal(facts),
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
    match ggml_type {
        GgmlType::Q4_K => elements / 32 * 16 + elements / 256 * 16,
        GgmlType::Q6_K => elements / 32 * 24 + elements / 256 * 20,
        _ => 0,
    }
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

/// Bytes of every prepared plane in `sets` over the target's tensors.
pub(crate) fn plane_bytes(tensors: &[TensorFact], sets: &[PlaneSet]) -> u64 {
    tensors
        .iter()
        .filter(|tensor| plane_set_of(&tensor.name).is_some_and(|set| sets.contains(&set)))
        .map(|tensor| {
            let elements = tensor.shape.iter().fold(1_u64, |product, extent| {
                product.saturating_mul(*extent as u64)
            });
            prepared_plane_bytes(tensor.ggml_type, elements)
        })
        .fold(0_u64, u64::saturating_add)
}

/// The memory rule's numbers.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MemoryEstimate {
    /// Physical memory minus the model file, the configured KV budget and
    /// [`MEMORY_RESERVE_BYTES`].
    pub(crate) headroom_bytes: u64,
    pub(crate) drafter_bytes: u64,
    /// Prepared planes of the routes as resolved.
    pub(crate) plane_bytes: u64,
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

/// `headroom = physical - model file - KV budget - reserve`; the full profile
/// needs `drafter + planes <= headroom`, the drafter alone `drafter <=
/// headroom`.
pub(crate) fn memory_verdict(
    host: HostFacts,
    model_file_bytes: u64,
    drafter_bytes: u64,
    profile_plane_bytes: u64,
) -> (MemoryVerdict, u64) {
    let headroom = host
        .physical_bytes
        .saturating_sub(model_file_bytes)
        .saturating_sub(host.kv_budget_bytes)
        .saturating_sub(MEMORY_RESERVE_BYTES);
    let verdict = if drafter_bytes.saturating_add(profile_plane_bytes) <= headroom {
        MemoryVerdict::Full
    } else if drafter_bytes <= headroom {
        MemoryVerdict::DropPlanes
    } else {
        MemoryVerdict::NoDrafter
    };
    (verdict, headroom)
}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1_u64 << 30) as f64)
}

const MEMORY_REFUSAL: &str = "its prepared planes do not fit in memory";

/// Decide the route of one model load. `env` reads one process environment
/// variable. An error is a startup error: an invalid `HF2Q_DFLASH2`, or a
/// required DFlash2 that cannot be satisfied.
pub(crate) fn decide(
    facts: &TargetFacts,
    host: HostFacts,
    env: &dyn Fn(&str) -> Option<String>,
    store: &dyn DrafterStore,
) -> Result<Selection> {
    let request = parse_policy(env(DFLASH2_ENV).as_deref()).map_err(anyhow::Error::msg)?;
    if request == PolicyRequest::Off {
        return Ok(Selection::prior_route(
            request,
            facts,
            format!("{DFLASH2_ENV}=0 keeps the prior route"),
            env,
        ));
    }
    let required = request == PolicyRequest::Required;

    let (descriptor, class) = match qualify(facts) {
        Ok(qualified) => qualified,
        Err(Unqualified::Refused(reason)) if required => {
            bail!("{DFLASH2_ENV}=1 cannot be satisfied: {reason}")
        }
        Err(Unqualified::NotServed(reason)) | Err(Unqualified::Refused(reason)) => {
            let mut prior = Selection::prior_route(request, facts, reason.clone(), env);
            if required {
                prior
                    .warnings
                    .push(format!("{DFLASH2_ENV}=1 has no effect: {reason}"));
            }
            return Ok(prior);
        }
    };
    let withdrawn = |reason: String, warnings: Vec<String>| {
        let mut prior = Selection::prior_route(request, facts, reason, env);
        prior.descriptor = Some(descriptor);
        prior.warnings = warnings;
        prior
    };
    let mut warnings = Vec::new();

    let explicit = |name: &str| {
        env(name)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    };
    let files = match (explicit(DFLASH2_CONFIG_ENV), explicit(DFLASH2_WEIGHTS_ENV)) {
        (Some(config_path), Some(weights_path)) => DrafterFiles {
            config_path,
            weights_path,
            source: DrafterSource::Explicit,
        },
        (config, weights) => {
            if config.is_some() || weights.is_some() {
                let message = format!(
                    "{DFLASH2_CONFIG_ENV} and {DFLASH2_WEIGHTS_ENV} are used only together"
                );
                if required {
                    bail!("{DFLASH2_ENV}=1: {message}");
                }
                warnings.push(format!(
                    "{message}; ignoring the one that is set and resolving the pinned drafter"
                ));
            }
            match store.cached(descriptor) {
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
                        "{DFLASH2_ENV}=1 cannot be satisfied: DFlash2 drafter {}@{} is not in \
                         the Hugging Face cache and could not be downloaded: {reason}",
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
            }
        }
    };

    // The drafter is bound to this target by the adapter's own shape check,
    // here on the header and again on the loaded model at provisioning. An
    // explicit checkpoint that cannot be read is left to provisioning, which
    // reports it as it always has.
    if let (Some(geometry), true) = (
        facts.geometry.as_ref(),
        files.source != DrafterSource::Explicit || files.config_path.is_file(),
    ) {
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

    // The profile is applied switch by switch: a row this artifact's storage
    // does not admit keeps its own default, and the drafter serves with the
    // rows that are left.
    let profile = RouteProfile::of(descriptor);
    let admitted = ResolvedRoutes::resolve(profile, &|row| switch_refusal(facts, row), env);
    let admitted_planes = plane_bytes(&facts.tensors, &admitted.plane_sets());
    let (verdict, headroom) = memory_verdict(
        host,
        facts.model_file_bytes,
        descriptor.weights_bytes,
        admitted_planes,
    );
    let formula = format!(
        "headroom {} = physical {} - model file {} - KV budget {} - reserve {}",
        gib(headroom),
        gib(host.physical_bytes),
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
    let (routes, reason) = match verdict {
        MemoryVerdict::Full => (admitted, qualified),
        MemoryVerdict::NoDrafter if !required => {
            let mut prior = withdrawn(
                format!(
                    "{qualified}; the drafter ({}) does not fit in memory: {formula}",
                    gib(descriptor.weights_bytes)
                ),
                warnings,
            );
            prior.memory = Some(MemoryEstimate {
                headroom_bytes: headroom,
                drafter_bytes: descriptor.weights_bytes,
                plane_bytes: 0,
            });
            return Ok(prior);
        }
        MemoryVerdict::DropPlanes | MemoryVerdict::NoDrafter => {
            let dropped = ResolvedRoutes::resolve(
                profile,
                &|row| {
                    switch_refusal(facts, row)
                        .or_else(|| (!row.planes.is_empty()).then(|| MEMORY_REFUSAL.to_owned()))
                },
                env,
            );
            if verdict == MemoryVerdict::NoDrafter {
                warnings.push(format!(
                    "{DFLASH2_ENV}=1 keeps the drafter ({}) although it does not fit: {formula}",
                    gib(descriptor.weights_bytes)
                ));
            }
            (
                dropped,
                format!(
                    "{qualified}; prepared planes ({}) dropped from the profile, drafter {} \
                     kept: {formula}",
                    gib(admitted_planes),
                    gib(descriptor.weights_bytes)
                ),
            )
        }
    };
    let memory = MemoryEstimate {
        headroom_bytes: headroom,
        drafter_bytes: descriptor.weights_bytes,
        plane_bytes: plane_bytes(&facts.tensors, &routes.plane_sets()),
    };
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
    })
}

/// [`decide`] for the model at `path` on this host and process environment.
pub(crate) fn resolve_for_load(
    path: &Path,
    engine_mode: EngineMode,
    kv_budget_bytes: Option<u64>,
    dwq_overlay: bool,
) -> Result<Selection> {
    let facts = {
        let gguf = GgufFile::open(path)
            .map_err(|error| anyhow::anyhow!("GGUF header parse failed: {error}"))?;
        TargetFacts::read(&gguf, path, engine_mode, dwq_overlay)
    };
    decide(
        &facts,
        HostFacts::read(kv_budget_bytes),
        &|name| std::env::var(name).ok(),
        &HubDrafterStore,
    )
    .with_context(|| format!("DFlash2 policy for {}", path.display()))
}

#[cfg(test)]
#[path = "dflash2_policy_tests.rs"]
mod tests;
