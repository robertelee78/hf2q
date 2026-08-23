//! ADR-051 shared local-first repository resolver and managed binding index.

mod inventory;
mod local;
mod materialize;
mod projector;
mod resolution;
mod storage;

pub(crate) use inventory::print_inventory;
use inventory::{find_matching_loose, scan_roots, visit_files};
use local::{find_best_matching_loose, select_local};
use materialize::materialize_preverified_exact;
use materialize::verify_or_refuse_existing_destination;
#[cfg(test)]
use materialize::{materialize_exact, materialize_exact_with_link};
#[cfg(test)]
use projector::select_projector_companion;
use projector::{best_effort_projector_with_catalog, resolve_projector};
pub(crate) use resolution::resolve_repository;
#[cfg(test)]
use resolution::{prepare_selected_local, select_hosted};
#[cfg(test)]
use storage::validate_binding;
use storage::{
    candidate_from_binding, conversion_authority, projector_authority_from_receipt, read_binding,
    safe_basename, scan_bindings, sidecar_path, write_binding,
};

use std::collections::BTreeSet;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::api::local_artifacts::{
    verify_local_artifact, LocalArtifactInventory, LocalArtifactProvenance,
    LocalVerificationRequest,
};
use super::cache::{CacheLock, ModelCache};
use super::quant_select::{quant_type_from_gguf_path, select_quant, GpuInfo, QuantType};
use crate::core::hardware::HardwareProfile;
use crate::input::hf_download::{
    check_hub_artifact_destination, check_hub_artifact_plan, download_hub_companion,
    download_hub_gguf, resolve_hub_gguf_catalog, HubGgufArtifact, HubGgufCatalog,
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

fn verify_candidate(candidate: &Candidate) -> Result<()> {
    verify_local_artifact(LocalVerificationRequest {
        root: &candidate.root,
        artifact: &candidate.path,
        bytes: candidate.bytes,
        sha256: &candidate.sha256,
        quant: candidate.quant,
    })?;
    Ok(())
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
    let actual = crate::core::sha256::compute_file_sha256(path)?;
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
    ModelCache::open_at(cache.root())?.touch(repository)?;
    Ok(())
}

pub(crate) fn text_requires_projector(path: &Path) -> Result<bool> {
    let gguf = mlx_native::gguf::GgufFile::open(path)?;
    let arch = gguf.metadata_string("general.architecture").unwrap_or("");
    if arch == "gemma4"
        || arch.contains("qwen3vl")
        || gguf
            .metadata_string("hf2q.vision.projector_profile")
            .is_some()
    {
        return Ok(true);
    }
    let markers = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"];
    let present = gguf.metadata("tokenizer.ggml.tokens").is_some_and(|value| {
        if let mlx_native::gguf::MetadataValue::Array(values) = value {
            markers.iter().all(|marker| values.iter().any(|value| matches!(value, mlx_native::gguf::MetadataValue::String(token) if token == marker)))
        } else {
            false
        }
    });
    Ok(present)
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
        let _ = metadata;
        candidates.push(candidate);
    }
    candidates.sort();

    if let Some(expected) = expected_projector_sha256(path)? {
        for candidate in candidates {
            if crate::core::sha256::compute_file_sha256(&candidate)?.eq_ignore_ascii_case(&expected)
            {
                return Ok(Some(candidate));
            }
        }
        bail!("multimodal local GGUF has no sibling matching its bound projector digest");
    }

    if let Some(paired) = paired_projector_path(path).filter(|paired| candidates.contains(paired)) {
        return Ok(Some(paired));
    }
    match candidates.as_slice() {
        [candidate] => Ok(Some(candidate.clone())),
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
