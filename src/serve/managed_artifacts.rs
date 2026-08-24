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
use local::{find_best_matching_cached_hub, find_best_matching_loose, select_local};
#[cfg(test)]
use local::{
    find_best_matching_cached_hub_with, find_best_matching_loose_with,
    find_best_matching_loose_with_hash,
};
#[cfg(test)]
use materialize::clone_requires_copy;
use materialize::verify_or_refuse_existing_destination;
use materialize::{
    materialize_preverified_exact, materialize_retained_exact, PreparedLocalArtifact,
};
use projector::{
    best_effort_projector_with_catalog, materialize_prepared_projector, prepare_projector_action,
    select_projector_companion, PreparedProjectorSource,
};
#[cfg(test)]
use projector::{resolve_projector, resolve_projector_with_catalog, retain_cached_projector_at};
pub(crate) use resolution::planned_native_product_bytes;
pub(crate) use resolution::resolve_repository;
#[cfg(test)]
use resolution::{
    admit_automatic_projector_preflight, automatic_artifact_admissible,
    bound_candidate_is_at_least_as_recent, exact_local_projector_catalog_reference,
    hosted_pair_requires_projector, native_convert, post_lock_local_candidate_wins,
    prepare_local_candidate_with_catalog_resolver, prepare_selected_local,
    prepare_selected_local_decision_with_preflight, repository_recommended_quant,
    reverify_candidate_after_catalog, select_compatible_hosted, select_hosted,
    select_native_fallback_quant, select_native_quant_from_exact_plans,
};
#[cfg(test)]
use storage::validate_binding;
use storage::{
    candidate_from_binding, conversion_authority, projector_authority_from_receipt, read_binding,
    safe_basename, scan_bindings, sidecar_path, write_binding,
};

use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::api::local_artifacts::{
    verify_retained_local_artifact, LocalArtifactInventory, LocalArtifactProvenance,
    LocalVerificationRequest,
};
use super::cache::{CacheLock, ModelCache};
use super::multi_model::LoadedPool;
use super::quant_select::{quant_type_from_gguf_file, select_quant, GpuInfo, QuantType};
use crate::core::hardware::HardwareProfile;
use crate::input::hf_download::{
    cached_hub_gguf_path, check_hosted_text_local_projector_plan_with_device,
    check_hub_artifact_destination, check_hub_artifact_pair_plan_from_state,
    check_hub_artifact_plan, check_local_artifact_pair_plan_with_authorities,
    check_local_text_hosted_projector_plan_with_authorities, download_hub_companion,
    download_hub_gguf, resolve_hub_gguf_catalog, validate_hub_gguf_header_compatibility,
    validate_retained_local_hub_gguf_compatibility, DownloadError, HubGgufArtifact, HubGgufCatalog,
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
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedManagedModel {
    pub(crate) gguf_path: PathBuf,
    pub(crate) mmproj_path: Option<PathBuf>,
    pub(crate) repository: String,
    pub(crate) revision: String,
    pub(crate) quant: QuantType,
    pub(crate) origin: String,
    pub(crate) warnings: Vec<String>,
}

fn verify_candidate(
    candidate: &Candidate,
) -> Result<crate::core::bounded_file::StableFileIdentity> {
    #[cfg(test)]
    VERIFY_CANDIDATE_CALLS.with(|calls| calls.set(calls.get() + 1));
    let retained =
        crate::core::bounded_file::StableRegularFile::open_exact(&candidate.path, candidate.bytes)?
            .context("local candidate changed before verification")?;
    let verified = verify_retained_local_artifact(
        LocalVerificationRequest {
            root: &candidate.root,
            artifact: &candidate.path,
            bytes: candidate.bytes,
            sha256: &candidate.sha256,
            quant: candidate.quant,
        },
        retained,
    )?;
    validate_retained_local_runtime_tensor_layout(&verified.retained)?;
    Ok(verified.retained.identity())
}

#[cfg(test)]
thread_local! {
    static VERIFY_CANDIDATE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn reset_verify_candidate_calls() {
    VERIFY_CANDIDATE_CALLS.with(|calls| calls.set(0));
}

#[cfg(test)]
fn verify_candidate_calls() -> usize {
    VERIFY_CANDIDATE_CALLS.with(std::cell::Cell::get)
}

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
    let Some((path, bytes, sha256)) = candidate.projector.as_ref() else {
        return Ok(None);
    };
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() != *bytes {
        return Ok(None);
    }
    let Some(actual) = crate::core::bounded_file::sha256_regular_nofollow_exact(path, *bytes)?
    else {
        return Ok(None);
    };
    if !actual.eq_ignore_ascii_case(sha256) {
        return Ok(None);
    }
    if expected_projector_sha256(&candidate.path)?
        .as_deref()
        .is_some_and(|expected| !actual.eq_ignore_ascii_case(expected))
    {
        return Ok(None);
    }
    Ok(Some(path.clone()))
}

impl Candidate {
    fn into_resolved(
        self,
        mmproj_path: Option<PathBuf>,
        warnings: Vec<String>,
    ) -> ResolvedManagedModel {
        ResolvedManagedModel {
            gguf_path: self.path,
            mmproj_path,
            repository: self.repository,
            revision: self.revision,
            quant: self.quant,
            origin: self.origin,
            warnings,
        }
    }
}

pub(crate) fn mark_successful_use(
    repository: &str,
    quant: QuantType,
    path: &Path,
    cache: &mut ModelCache,
) -> Result<()> {
    let now = now_secs();
    let _lock = cache
        .lock_quant(repository, quant)
        .with_context(|| format!("lock successful-use publication for {repository}:{quant}"))?;
    let sidecar = sidecar_path(path);
    if let Some(mut binding) = read_binding(&sidecar)? {
        if binding.repository != repository || binding.quant != quant.as_str() {
            bail!("managed binding repository changed before successful-use publication");
        }
        binding.last_used_at_secs = now;
        write_binding(&sidecar, &binding)?;
    }
    // Reload under the quant lock so an old in-memory snapshot cannot flush
    // over another process's completed manifest update.
    ModelCache::open_at(cache.root())?.touch_quant(repository, quant)?;
    Ok(())
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

    let sidecar = sidecar_path(path);
    if let Some(binding) = read_binding(&sidecar)? {
        let candidate = candidate_from_binding(binding, path.to_path_buf(), sidecar)?;
        if let Some(projector) = verify_candidate_projector(&candidate)? {
            return Ok(Some(projector));
        }
    }
    if let Some(candidate) = conversion_authority(path)? {
        if let Some(projector) = verify_candidate_projector(&candidate)? {
            return Ok(Some(projector));
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
        let metadata = match fs::symlink_metadata(&candidate) {
            Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => metadata,
            _ => continue,
        };
        candidates.push((candidate, metadata.len()));
    }
    candidates.sort_by(|left, right| left.0.cmp(&right.0));

    if let Some(expected) = expected_projector_sha256(path)? {
        for (candidate, bytes) in candidates {
            if crate::core::bounded_file::sha256_regular_nofollow_exact(&candidate, bytes)?
                .is_some_and(|digest| digest.eq_ignore_ascii_case(&expected))
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
