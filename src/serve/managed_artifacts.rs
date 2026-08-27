//! ADR-051 shared local-first repository resolver and managed binding index.

mod inventory;
mod local;
mod materialize;
mod projector;
mod resolution;
mod storage;

pub(crate) use inventory::print_inventory;
use inventory::{find_matching_loose, scan_roots, visit_files};
#[cfg(test)]
use local::local_candidate_eligible;
#[cfg(test)]
use local::{
    find_best_matching_cached_hub_with, find_best_matching_loose, find_best_matching_loose_with,
    find_best_matching_loose_with_hash, select_local,
};
use local::{
    find_best_matching_cached_hub_with_progress, find_best_matching_loose_with_progress,
    hash_hosted_local_candidate, select_local_with_progress,
};
use materialize::verify_or_refuse_existing_destination;
#[cfg(test)]
use materialize::{clone_requires_copy, materialize_preverified_exact};
use materialize::{
    materialize_hub_cache_symlink, materialize_retained_exact, PreparedLocalArtifact,
};
#[cfg(test)]
use projector::best_effort_projector_with_catalog;
use projector::{
    best_effort_projector_with_catalog_expected_with_progress,
    best_effort_projector_with_catalog_with_progress, materialize_prepared_projector,
    prepare_projector_action, retain_cached_projector_at, select_projector_companion,
    PreparedProjectorSource,
};
#[cfg(test)]
use projector::{resolve_projector, resolve_projector_with_catalog};
pub(crate) use resolution::planned_native_product_bytes;
#[cfg(test)]
pub(crate) use resolution::resolve_repository;
pub(crate) use resolution::resolve_repository_with_progress;
#[cfg(test)]
use resolution::{
    admit_automatic_projector_preflight, automatic_artifact_admissible,
    best_effort_manual_projector_with_catalog, bind_existing_local_projector,
    bound_candidate_is_at_least_as_recent, exact_local_projector_catalog_reference,
    hosted_pair_requires_projector, native_convert, post_lock_local_candidate_wins,
    prepare_cached_projector_in_place_with_sources, prepare_local_candidate_with_catalog_resolver,
    prepare_selected_local, prepare_selected_local_decision_with_preflight,
    repository_recommended_quant, resolve_repository_with_progress_and_catalog,
    reverify_candidate_after_catalog, select_compatible_hosted, select_hosted,
    select_native_fallback_quant, select_native_quant_from_exact_plans,
};
use storage::{
    candidate_from_binding, conversion_authority, projector_authority_from_receipt, read_binding,
    safe_basename, scan_bindings, sidecar_path, write_binding,
};
#[cfg(test)]
use storage::{candidate_from_binding_in, validate_binding};

use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::api::local_artifacts::{LocalArtifactInventory, LocalArtifactProvenance};
use super::cache::{CacheLock, ModelCache};
use super::multi_model::LoadedPool;
use super::quant_select::{quant_type_from_gguf_file, select_quant, GpuInfo, QuantType};
use super::startup_progress::{StartupEvent, StartupOrigin};
use crate::core::hardware::HardwareProfile;
#[cfg(test)]
use crate::input::hf_download::managed_hub_cache_link_is_expected_dangling_in;
use crate::input::hf_download::{
    cached_hub_gguf_path, check_hosted_text_local_projector_plan_with_device,
    check_hub_artifact_destination, check_hub_artifact_pair_plan_from_state,
    check_hub_artifact_plan, check_local_artifact_pair_plan_with_authorities,
    check_local_text_hosted_projector_plan_with_authorities, download_hub_companion_with_progress,
    download_hub_gguf_with_progress, hf_hub_cache_dir, managed_hub_cache_link_is_expected_dangling,
    resolve_hub_gguf_catalog, retain_managed_hub_cache_link, retain_managed_hub_cache_link_in,
    validate_hub_gguf_header_compatibility, validate_retained_local_hub_gguf_compatibility,
    DownloadError, DownloadedHubArtifact, HubGgufArtifact, HubGgufCatalog,
};
use crate::input::hf_reference::HfModelReference;
use crate::model_spec::{
    default_convert_output, managed_model_root, managed_revision_dir, resolve_output_path,
    RepositoryModelSpec,
};

const SCHEMA_VERSION: u32 = 1;
const SIDECAR_SUFFIX: &str = ".hf2q.json";
const MAX_SIDECAR_BYTES: u64 = 64 * 1024;
const MAX_CONVERSION_RECEIPT_BYTES: u64 = 1024 * 1024;
const MAX_SCAN_DEPTH: usize = 6;
const MAX_SCAN_ENTRIES: usize = 4096;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactBinding {
    local_filename: String,
    hub_filename: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManagedBinding {
    schema_version: u32,
    repository: String,
    revision: String,
    quant: String,
    origin: String,
    materialized_at_secs: u64,
    last_used_at_secs: u64,
    artifact: ArtifactBinding,
    projector: Option<ArtifactBinding>,
}

#[derive(Debug, Clone)]
struct Candidate {
    repository: String,
    revision: String,
    path: PathBuf,
    root: PathBuf,
    bytes: u64,
    sha256: String,
    quant: QuantType,
    origin: String,
    materialized_at_secs: u64,
    last_used_at_secs: u64,
    projector: Option<(PathBuf, u64, String)>,
    sidecar: Option<PathBuf>,
    receipt_target_identity: Option<crate::core::bounded_file::StableFileIdentity>,
}

pub(crate) struct ResolvedManagedModel {
    pub(crate) gguf_path: PathBuf,
    pub(crate) mmproj_path: Option<PathBuf>,
    pub(crate) repository: String,
    pub(crate) revision: String,
    pub(crate) quant: QuantType,
    pub(crate) origin: String,
    pub(crate) warnings: Vec<String>,
    pub(crate) track_success_history: bool,
    pub(crate) activation_authority: Option<crate::core::bounded_file::StableRegularFile>,
    pub(crate) mmproj_sha256: Option<String>,
    pub(crate) mmproj_activation_authority: Option<crate::core::bounded_file::StableRegularFile>,
}

type StartupProgress<'a> = dyn FnMut(StartupEvent) + 'a;

fn verify_candidate(candidate: &Candidate) -> Result<crate::core::bounded_file::StableRegularFile> {
    let mut silent = |_| {};
    verify_candidate_with_progress(candidate, &mut silent)
}

fn verify_candidate_with_progress(
    candidate: &Candidate,
    progress: &mut StartupProgress<'_>,
) -> Result<crate::core::bounded_file::StableRegularFile> {
    #[cfg(test)]
    VERIFY_CANDIDATE_CALLS.with(|calls| calls.set(calls.get() + 1));
    progress(StartupEvent::LocalCandidate {
        quant: candidate.quant.as_str().to_owned(),
        origin: StartupOrigin::from_internal(&candidate.origin),
        bytes: candidate.bytes,
        filename: display_filename(&candidate.path),
    });
    let retained = crate::core::bounded_file::StableRegularFile::open_operator_path_exact(
        &candidate.path,
        candidate.bytes,
    )?
    .context("local candidate changed before verification")?;
    if candidate
        .receipt_target_identity
        .is_some_and(|expected| retained.identity() != expected)
    {
        if candidate.origin == LocalArtifactProvenance::ConversionReceipt.as_str() {
            bail!("linked conversion target changed after receipt authentication");
        }
        bail!("linked managed cache target changed after binding authentication");
    }
    let actual_quant = quant_type_from_gguf_file(retained.try_clone()?, &candidate.path)
        .context("read local GGUF quant metadata")?;
    if actual_quant != candidate.quant {
        bail!(
            "local GGUF metadata quant {} does not match bound quant {}",
            actual_quant,
            candidate.quant
        );
    }
    validate_retained_local_runtime_tensor_layout(&retained)?;
    Ok(retained)
}

#[cfg(test)]
thread_local! {
    static VERIFY_CANDIDATE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static VERIFY_PROJECTOR_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static AFTER_AUTOMATIC_PROJECTOR_PREPARED: std::cell::RefCell<Option<Box<dyn FnOnce(&Path)>>> =
        std::cell::RefCell::new(None);
    static AFTER_RETAINED_TEXT_PROJECTOR_METADATA: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        std::cell::RefCell::new(None);
    static BEFORE_MANUAL_HOSTED_PROJECTOR_FALLBACK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        std::cell::RefCell::new(None);
}

#[cfg(test)]
fn reset_verify_candidate_calls() {
    VERIFY_CANDIDATE_CALLS.with(|calls| calls.set(0));
}

#[cfg(test)]
fn verify_candidate_calls() -> usize {
    VERIFY_CANDIDATE_CALLS.with(std::cell::Cell::get)
}

#[cfg(test)]
fn reset_verify_projector_calls() {
    VERIFY_PROJECTOR_CALLS.with(|calls| calls.set(0));
}

#[cfg(test)]
fn verify_projector_calls() -> usize {
    VERIFY_PROJECTOR_CALLS.with(std::cell::Cell::get)
}

#[cfg(test)]
struct AutomaticProjectorHookGuard;

#[cfg(test)]
impl Drop for AutomaticProjectorHookGuard {
    fn drop(&mut self) {
        AFTER_AUTOMATIC_PROJECTOR_PREPARED.with(|slot| *slot.borrow_mut() = None);
    }
}

#[cfg(test)]
fn set_after_automatic_projector_prepared(
    hook: impl FnOnce(&Path) + 'static,
) -> AutomaticProjectorHookGuard {
    AFTER_AUTOMATIC_PROJECTOR_PREPARED.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    AutomaticProjectorHookGuard
}

#[cfg(test)]
fn run_after_automatic_projector_prepared(path: &Path) {
    AFTER_AUTOMATIC_PROJECTOR_PREPARED.with(|slot| {
        if let Some(hook) = slot.borrow_mut().take() {
            hook(path);
        }
    });
}

#[cfg(not(test))]
fn run_after_automatic_projector_prepared(_path: &Path) {}

#[cfg(test)]
struct RetainedTextProjectorMetadataHookGuard;

#[cfg(test)]
impl Drop for RetainedTextProjectorMetadataHookGuard {
    fn drop(&mut self) {
        AFTER_RETAINED_TEXT_PROJECTOR_METADATA.with(|slot| *slot.borrow_mut() = None);
    }
}

#[cfg(test)]
fn set_after_retained_text_projector_metadata(
    hook: impl FnOnce() + 'static,
) -> RetainedTextProjectorMetadataHookGuard {
    AFTER_RETAINED_TEXT_PROJECTOR_METADATA.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    RetainedTextProjectorMetadataHookGuard
}

#[cfg(test)]
fn run_after_retained_text_projector_metadata() {
    AFTER_RETAINED_TEXT_PROJECTOR_METADATA.with(|slot| {
        if let Some(hook) = slot.borrow_mut().take() {
            hook();
        }
    });
}

#[cfg(not(test))]
fn run_after_retained_text_projector_metadata() {}

#[cfg(test)]
struct ManualHostedProjectorFallbackHookGuard;

#[cfg(test)]
impl Drop for ManualHostedProjectorFallbackHookGuard {
    fn drop(&mut self) {
        BEFORE_MANUAL_HOSTED_PROJECTOR_FALLBACK.with(|slot| *slot.borrow_mut() = None);
    }
}

#[cfg(test)]
fn set_before_manual_hosted_projector_fallback(
    hook: impl FnOnce() + 'static,
) -> ManualHostedProjectorFallbackHookGuard {
    BEFORE_MANUAL_HOSTED_PROJECTOR_FALLBACK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    ManualHostedProjectorFallbackHookGuard
}

#[cfg(test)]
fn run_before_manual_hosted_projector_fallback() {
    BEFORE_MANUAL_HOSTED_PROJECTOR_FALLBACK.with(|slot| {
        if let Some(hook) = slot.borrow_mut().take() {
            hook();
        }
    });
}

#[cfg(not(test))]
fn run_before_manual_hosted_projector_fallback() {}

fn validate_retained_local_runtime_tensor_layout(
    retained: &crate::core::bounded_file::StableRegularFile,
) -> Result<()> {
    let gguf = mlx_native::gguf::GgufFile::from_file(retained.try_clone()?)?;
    let arch = gguf.metadata_string("general.architecture").unwrap_or("");
    for name in gguf.tensor_names() {
        let info = gguf
            .tensor_info(name)
            .expect("tensor name returned by GGUF must have tensor info");
        if let Some(reason) = local_runtime_tensor_incompatibility(arch, name, info.ggml_type) {
            bail!("GGUF tensor layout is not executable by this build: {reason}");
        }
    }
    crate::input::hf_download::validate_qwen_runtime_admission(&gguf)
        .map_err(|reason| anyhow!("GGUF runtime admission failed: {reason}"))?;
    if !retained.is_stable()? {
        bail!("local GGUF changed during runtime layout admission");
    }
    Ok(())
}

fn local_runtime_tensor_incompatibility(
    arch: &str,
    name: &str,
    ggml_type: mlx_native::ops::quantized_matmul_ggml::GgmlType,
) -> Option<String> {
    use crate::inference::models::qwen35::forward_gpu::qwen35_native_embedding_type_supported;
    use crate::inference::models::qwen35::weight_loader::{
        qwen35_dense_ffn_type_supported, qwen35_moe_expert_type_supported,
        qwen35_native_projection_type_supported,
    };
    use crate::inference::models::qwen3vl_text::weights::qwen3vl_projection_type_supported;

    if matches!(arch, "qwen35" | "qwen35moe") {
        let supported = if name == "token_embd.weight" {
            qwen35_native_embedding_type_supported(ggml_type)
        } else if qwen35_dense_ffn_name(name) {
            qwen35_dense_ffn_type_supported(ggml_type)
        } else if qwen35_moe_expert_name(name) {
            qwen35_moe_expert_type_supported(ggml_type)
        } else if qwen35_native_projection_name(name) {
            qwen35_native_projection_type_supported(ggml_type)
        } else {
            true
        };
        if !supported {
            return Some(format!(
                "{name} uses unsupported {ggml_type:?} storage for {arch}"
            ));
        }
    } else if (arch == "qwen3_vl" || arch == "qwen3vl")
        && qwen3vl_projection_name(name)
        && !qwen3vl_projection_type_supported(ggml_type)
    {
        return Some(format!(
            "{name} uses unsupported {ggml_type:?} storage for {arch}"
        ));
    }
    None
}

fn qwen35_native_projection_name(name: &str) -> bool {
    name == "output.weight"
        || [
            ".attn_q.weight",
            ".attn_k.weight",
            ".attn_v.weight",
            ".attn_output.weight",
            ".attn_qkv.weight",
            ".attn_gate.weight",
            ".ssm_alpha.weight",
            ".ssm_beta.weight",
            ".ssm_out.weight",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

fn qwen35_dense_ffn_name(name: &str) -> bool {
    [".ffn_gate.weight", ".ffn_up.weight", ".ffn_down.weight"]
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

fn qwen35_moe_expert_name(name: &str) -> bool {
    [
        ".ffn_gate_exps.weight",
        ".ffn_up_exps.weight",
        ".ffn_down_exps.weight",
    ]
    .iter()
    .any(|suffix| name.ends_with(suffix))
}

fn qwen3vl_projection_name(name: &str) -> bool {
    [
        ".attn_q.weight",
        ".attn_k.weight",
        ".attn_v.weight",
        ".attn_output.weight",
        ".ffn_gate.weight",
        ".ffn_up.weight",
        ".ffn_down.weight",
    ]
    .iter()
    .any(|suffix| name.ends_with(suffix))
}

fn verify_candidate_projector(candidate: &Candidate) -> Result<Option<PathBuf>> {
    let mut silent = |_| {};
    verify_candidate_projector_with_progress(candidate, &mut silent)
}

fn verify_candidate_projector_with_progress(
    candidate: &Candidate,
    _progress: &mut StartupProgress<'_>,
) -> Result<Option<PathBuf>> {
    let Some((path, bytes, expected_sha256)) = candidate.projector.as_ref() else {
        return Ok(None);
    };
    #[cfg(test)]
    VERIFY_PROJECTOR_CALLS.with(|calls| calls.set(calls.get() + 1));
    let metadata = match fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if !metadata.is_file() || metadata.len() != *bytes {
        return Ok(None);
    }
    let Some(mut retained) =
        crate::core::bounded_file::StableRegularFile::open_operator_path_exact(path, *bytes)?
    else {
        return Ok(None);
    };
    let Some(actual_sha256) = retained.sha256()? else {
        return Ok(None);
    };
    if !actual_sha256.eq_ignore_ascii_case(expected_sha256) || !retained.is_stable()? {
        return Ok(None);
    }
    let gguf = match mlx_native::gguf::GgufFile::from_file(retained.try_clone()?) {
        Ok(gguf) => gguf,
        Err(_) => return Ok(None),
    };
    if gguf.tensor_names().is_empty() || !retained.is_stable()? {
        return Ok(None);
    }
    Ok(Some(path.clone()))
}

fn retain_verified_projector_authority(
    path: &Path,
    bytes: u64,
    expected_sha256: &str,
) -> Result<Option<crate::core::bounded_file::StableRegularFile>> {
    let Some(mut authority) =
        crate::core::bounded_file::StableRegularFile::open_operator_path_exact(path, bytes)?
    else {
        return Ok(None);
    };
    let Some(actual_sha256) = authority.sha256()? else {
        return Ok(None);
    };
    Ok(
        (actual_sha256.eq_ignore_ascii_case(expected_sha256) && authority.is_stable()?)
            .then_some(authority),
    )
}

fn display_filename(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "<unnamed GGUF>".into())
}

impl Candidate {
    fn into_resolved(
        self,
        mut mmproj_path: Option<PathBuf>,
        mut warnings: Vec<String>,
        verified_activation: Option<crate::core::bounded_file::StableRegularFile>,
    ) -> Result<ResolvedManagedModel> {
        // Final activation admission retains the exact inode that passed the
        // bounded quant/runtime check. Managed bindings do not pay a second
        // full-payload hash merely to close the resolve-to-load pathname gap.
        let activation_authority = match verified_activation {
            Some(authority) => {
                if !authority.is_stable()? {
                    bail!("managed text GGUF changed after bounded verification");
                }
                authority
            }
            None => verify_candidate(&self)
                .context("managed text GGUF changed before retained activation")?,
        };
        let mmproj_sha256 = mmproj_path.as_ref().and_then(|path| {
            self.projector
                .as_ref()
                .filter(|(bound, _, _)| bound == path)
                .map(|(_, _, sha256)| sha256.clone())
        });
        let mut mmproj_sha256 = mmproj_sha256;
        let mmproj_activation_authority = match mmproj_path.as_ref().and_then(|path| {
            self.projector
                .as_ref()
                .filter(|(bound, _, _)| bound == path)
                .map(|(path, bytes, _)| (path, *bytes))
        }) {
            Some((path, bytes)) => {
                let expected_sha256 = mmproj_sha256
                    .as_deref()
                    .context("automatic managed mmproj lost its digest binding")?;
                match retain_verified_projector_authority(path, bytes, expected_sha256) {
                    Ok(Some(authority)) => Some(authority),
                    Ok(None) => {
                        warnings.push(
                            "automatic managed mmproj changed or no longer matches its digest before retained activation; serving text-only"
                                .into(),
                        );
                        mmproj_path = None;
                        mmproj_sha256 = None;
                        None
                    }
                    Err(error) => {
                        warnings.push(format!(
                            "automatic managed mmproj retention failed; serving text-only: {error}"
                        ));
                        mmproj_path = None;
                        mmproj_sha256 = None;
                        None
                    }
                }
            }
            None if mmproj_path.is_some() => {
                warnings.push(
                    "automatic managed mmproj has no retained digest binding; serving text-only"
                        .into(),
                );
                mmproj_path = None;
                mmproj_sha256 = None;
                None
            }
            None => None,
        };
        Ok(ResolvedManagedModel {
            gguf_path: self.path,
            mmproj_path,
            repository: self.repository,
            revision: self.revision,
            quant: self.quant,
            origin: self.origin,
            warnings,
            track_success_history: true,
            activation_authority: Some(activation_authority),
            mmproj_sha256,
            mmproj_activation_authority,
        })
    }
}

pub(crate) fn mark_successful_use(
    repository: &str,
    revision: &str,
    quant: QuantType,
    path: &Path,
    expected_authority: &crate::core::bounded_file::StableRegularFile,
    cache: &mut ModelCache,
) -> Result<()> {
    let now = now_secs();
    let _lock = cache
        .lock_quant(repository, quant)
        .with_context(|| format!("lock successful-use publication for {repository}:{quant}"))?;
    if !expected_authority.is_stable()?
        || !crate::core::bounded_file::regular_path_matches_identity(
            path,
            expected_authority.identity(),
        )?
    {
        bail!("managed model changed before successful-use publication");
    }
    let sidecar = sidecar_path(path);
    let receipt_binding =
        receipt_usage_binding(path, repository, revision, quant, expected_authority, now)?;
    let mut binding = match (read_binding(&sidecar)?, receipt_binding) {
        (Some(existing), Some(mut receipt)) => {
            if !usage_binding_matches_receipt(&existing, &receipt) {
                bail!("managed binding artifact changed before successful-use publication");
            }
            // Receipt-backed sidecars are recency-only. Preserve the existing
            // text authority spelling so atomic replacement remains allowed,
            // but clear any projector/origin fields the sidecar tried to add.
            receipt.artifact.hub_filename = existing.artifact.hub_filename;
            Some(receipt)
        }
        (None, Some(receipt)) => Some(receipt),
        (Some(existing), None) => Some(existing),
        (None, None) => None,
    };
    if let Some(binding) = binding.as_mut() {
        if binding.repository != repository
            || !binding.revision.eq_ignore_ascii_case(revision)
            || binding.quant != quant.as_str()
        {
            bail!("managed binding repository changed before successful-use publication");
        }
        binding.last_used_at_secs = now;
        write_binding(&sidecar, binding)?;
    }
    // Reload under the quant lock so an old in-memory snapshot cannot flush
    // over another process's completed manifest update.
    ModelCache::open_at(cache.root())?.touch_quant_if_matches(
        repository,
        revision,
        quant,
        expected_authority.identity(),
    )?;
    Ok(())
}

fn usage_binding_matches_receipt(existing: &ManagedBinding, receipt: &ManagedBinding) -> bool {
    existing.schema_version == receipt.schema_version
        && existing.repository == receipt.repository
        && existing.revision.eq_ignore_ascii_case(&receipt.revision)
        && existing.quant.eq_ignore_ascii_case(&receipt.quant)
        && existing.artifact.local_filename == receipt.artifact.local_filename
        && existing.artifact.bytes == receipt.artifact.bytes
        && existing
            .artifact
            .sha256
            .eq_ignore_ascii_case(&receipt.artifact.sha256)
}

/// Build a recency-only binding for a regular conversion output or final-leaf
/// symlink whose retained target is authenticated by an hf2q conversion
/// receipt. The receipt remains repository/revision/artifact authority; this
/// sidecar is imported only when every immutable field still matches.
fn receipt_usage_binding(
    logical_path: &Path,
    repository: &str,
    revision: &str,
    quant: QuantType,
    expected_authority: &crate::core::bounded_file::StableRegularFile,
    last_used_at_secs: u64,
) -> Result<Option<ManagedBinding>> {
    match fs::symlink_metadata(logical_path) {
        Ok(metadata) if metadata.file_type().is_symlink() || metadata.is_file() => {}
        Ok(_) => return Ok(None),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    }
    let target = expected_authority
        .canonical_path_for_identity()?
        .context("successful-use activation authority no longer has one exact path")?;
    let Some(candidate) = conversion_authority(&target)? else {
        return Ok(None);
    };
    if candidate.repository != repository
        || !candidate.revision.eq_ignore_ascii_case(revision)
        || candidate.quant != quant
    {
        bail!("receipt-bound model identity changed before successful-use publication");
    }
    let Some(logical) = crate::core::bounded_file::StableRegularFile::open_operator_path_exact(
        logical_path,
        candidate.bytes,
    )?
    else {
        bail!("receipt-bound model changed before successful-use publication");
    };
    let Some(target_authority) =
        crate::core::bounded_file::StableRegularFile::open_exact(&target, candidate.bytes)?
    else {
        bail!("receipt-bound conversion target changed before successful-use publication");
    };
    if logical.identity() != expected_authority.identity()
        || target_authority.identity() != expected_authority.identity()
        || !expected_authority.is_stable()?
        || !logical.is_stable()?
        || !target_authority.is_stable()?
    {
        bail!("receipt-bound model target changed before successful-use publication");
    }
    let local_filename = logical_path
        .file_name()
        .and_then(|name| name.to_str())
        .context("receipt-bound logical filename is not UTF-8")?
        .to_owned();
    let hub_filename = target
        .file_name()
        .and_then(|name| name.to_str())
        .context("receipt-bound target filename is not UTF-8")?
        .to_owned();
    Ok(Some(ManagedBinding {
        schema_version: SCHEMA_VERSION,
        repository: candidate.repository,
        revision: candidate.revision.to_ascii_lowercase(),
        quant: candidate.quant.as_str().to_owned(),
        origin: candidate.origin,
        materialized_at_secs: candidate.materialized_at_secs,
        last_used_at_secs,
        artifact: ArtifactBinding {
            local_filename,
            hub_filename,
            bytes: candidate.bytes,
            sha256: candidate.sha256.to_ascii_lowercase(),
        },
        // A recency sidecar never grants projector authority. The conversion
        // receipts beside the exact text/projector targets do that.
        projector: None,
    }))
}

pub(crate) fn text_requires_projector(path: &Path) -> Result<bool> {
    let gguf = mlx_native::gguf::GgufFile::open(path)?;
    Ok(text_gguf_requires_projector(&gguf))
}

pub(crate) fn text_gguf_requires_projector(gguf: &mlx_native::gguf::GgufFile) -> bool {
    let arch = gguf.metadata_string("general.architecture").unwrap_or("");
    if arch == "gemma4"
        || arch.contains("qwen3vl")
        || gguf
            .metadata_string("hf2q.vision.projector_profile")
            .is_some()
    {
        return true;
    }
    let markers = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"];
    let present = gguf.metadata("tokenizer.ggml.tokens").is_some_and(|value| {
        if let mlx_native::gguf::MetadataValue::Array(values) = value {
            markers.iter().all(|marker| values.iter().any(|value| matches!(value, mlx_native::gguf::MetadataValue::String(token) if token == marker)))
        } else {
            false
        }
    });
    present
}

/// Resolve a projector already present beside an explicitly supplied local
/// GGUF. Managed/conversion authority wins; otherwise the text GGUF's bound
/// projector digest or an unambiguous sibling name is used. The serve pair
/// guard remains the final activation authority.
pub(crate) fn resolve_local_path_projector(path: &Path) -> Result<Option<PathBuf>> {
    if !text_requires_projector(path)? {
        return Ok(None);
    }
    resolve_local_path_projector_required(path)
}

/// Resolve a local projector when repository or authenticated GGUF metadata
/// has already established that the text artifact is multimodal.
pub(crate) fn resolve_local_path_projector_required(path: &Path) -> Result<Option<PathBuf>> {
    let expected = expected_projector_sha256(path)?;
    resolve_local_path_projector_required_with_expected(path, expected.as_deref())
}

fn resolve_local_path_projector_required_with_expected(
    path: &Path,
    expected_projector_sha256: Option<&str>,
) -> Result<Option<PathBuf>> {
    let receipt_candidate = conversion_authority(path)?;
    if let Some(candidate) = receipt_candidate.as_ref() {
        if let Some(projector) = verify_candidate_projector(candidate)? {
            let matches = candidate
                .projector
                .as_ref()
                .is_some_and(|(bound, _, sha256)| {
                    bound == &projector
                        && expected_projector_sha256
                            .is_none_or(|expected| sha256.eq_ignore_ascii_case(expected))
                });
            if matches {
                return Ok(Some(projector));
            }
        }
    }
    let sidecar = sidecar_path(path);
    if receipt_candidate.is_none() {
        if let Some(binding) = read_binding(&sidecar)? {
            let candidate = candidate_from_binding(binding, path.to_path_buf(), sidecar)?;
            if let Some(projector) = verify_candidate_projector(&candidate)? {
                let matches = candidate
                    .projector
                    .as_ref()
                    .is_some_and(|(bound, _, sha256)| {
                        bound == &projector
                            && expected_projector_sha256
                                .is_none_or(|expected| sha256.eq_ignore_ascii_case(expected))
                    });
                if matches {
                    return Ok(Some(projector));
                }
            }
        }
    }
    let parent = path.parent().context("local text GGUF has no parent")?;
    let mut candidates = Vec::new();
    for (index, entry) in fs::read_dir(parent)?.enumerate() {
        if index >= 512 {
            break;
        }
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        let candidate = entry.path();
        if candidate == path
            || candidate
                .extension()
                .and_then(|value| value.to_str())
                .is_none_or(|extension| !extension.eq_ignore_ascii_case("gguf"))
        {
            continue;
        }
        let name = match candidate.file_name().and_then(|name| name.to_str()) {
            Some(name) => name.to_ascii_lowercase(),
            None => continue,
        };
        if !name.starts_with("mmproj") && !name.contains("-mmproj") {
            continue;
        }
        let metadata = match fs::metadata(&candidate) {
            Ok(metadata) if metadata.is_file() => metadata,
            _ => continue,
        };
        candidates.push((candidate, metadata.len()));
    }
    candidates.sort_by(|left, right| left.0.cmp(&right.0));

    if let Some(expected) = expected_projector_sha256 {
        for (candidate, bytes) in candidates {
            if crate::core::bounded_file::sha256_operator_path_exact(&candidate, bytes)?
                .is_some_and(|digest| digest.eq_ignore_ascii_case(expected))
            {
                return Ok(Some(candidate));
            }
        }
        bail!("multimodal local GGUF has no sibling matching its bound projector digest");
    }

    if let Some(paired) = paired_projector_path(path)
        .filter(|paired| candidates.iter().any(|(candidate, _)| candidate == paired))
    {
        return Ok(Some(paired));
    }
    match candidates.as_slice() {
        [(candidate, _)] => Ok(Some(candidate.clone())),
        [] => bail!("multimodal local GGUF has no projector beside it; serving text-only"),
        _ => bail!(
            "multimodal local GGUF has multiple unbound projector siblings; serving text-only"
        ),
    }
}

fn expected_projector_sha256(path: &Path) -> Result<Option<String>> {
    crate::core::provenance::projector_sha256(&mlx_native::gguf::GgufFile::open(path)?)
        .map_err(|error| anyhow!(error))
}

fn retained_text_requires_projector(
    authority: &crate::core::bounded_file::StableRegularFile,
) -> Result<bool> {
    retained_text_projector_contract(authority).map(|(required, _)| required)
}

fn retained_expected_projector_sha256(
    authority: &crate::core::bounded_file::StableRegularFile,
) -> Result<Option<String>> {
    retained_text_projector_contract(authority).map(|(_, expected)| expected)
}

fn retained_text_projector_contract(
    authority: &crate::core::bounded_file::StableRegularFile,
) -> Result<(bool, Option<String>)> {
    let gguf = mlx_native::gguf::GgufFile::from_file(authority.try_clone()?)?;
    let required = text_gguf_requires_projector(&gguf);
    let expected =
        crate::core::provenance::projector_sha256(&gguf).map_err(|error| anyhow!(error))?;
    run_after_retained_text_projector_metadata();
    if !authority.is_stable()? {
        bail!("local text GGUF changed during retained projector planning");
    }
    Ok((required, expected))
}

fn paired_projector_path(text: &Path) -> Option<PathBuf> {
    let name = text.file_name()?.to_str()?;
    let stem = name
        .strip_suffix(".gguf")
        .or_else(|| name.strip_suffix(".GGUF"))?;
    Some(text.with_file_name(format!("{stem}-mmproj.gguf")))
}

fn quality_descending() -> [QuantType; 6] {
    [
        QuantType::Q8_0,
        QuantType::Q6_K,
        QuantType::Q5_K_M,
        QuantType::Q4_K_M,
        QuantType::Q3_K_M,
        QuantType::Q2_K,
    ]
}

fn quant_quality(quant: QuantType) -> u8 {
    match quant {
        QuantType::Q2_K => 1,
        QuantType::Q3_K_M => 2,
        QuantType::Q4_K_M => 3,
        QuantType::Q5_K_M => 4,
        QuantType::Q6_K => 5,
        QuantType::Q8_0 => 6,
    }
}

fn candidate_recency(candidate: &Candidate) -> (bool, u64, u64) {
    (
        candidate.last_used_at_secs != 0,
        candidate.last_used_at_secs,
        candidate.materialized_at_secs,
    )
}

fn short_revision(revision: &str) -> &str {
    revision.get(..12).unwrap_or(revision)
}

fn display_timestamp(timestamp: u64) -> String {
    if timestamp == 0 {
        "-".to_owned()
    } else {
        timestamp.to_string()
    }
}

fn now_secs() -> u64 {
    system_time_secs(SystemTime::now())
}

fn system_time_secs(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn is_hex(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests;
