//! Bounded local hf2q artifact discovery for diagnostic chat.
//!
//! A filename is not provenance. This inventory accepts only hf2q-owned
//! schema-v3 conversion receipts and canonical `ModelCache` entries. Paths
//! and digests remain server-private; activation revalidates the selected
//! artifact in a cancellable direct child before the loader is invoked.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::{Component, Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::convert::receipt::{ConversionReceipt, CONVERSION_RECEIPT_SCHEMA_VERSION};
use crate::serve::cache::cache_mmproj_path;
use crate::serve::cache::{cache_model_path, CacheManifest, ModelEntry, QuantEntry};
use crate::serve::quant_select::QuantType;

mod verification;

pub use verification::{verify_local_artifact, LocalVerificationReceipt, LocalVerificationRequest};

const RECEIPT_SUFFIX: &str = ".gguf.receipt.json";
const MAX_ROOTS: usize = 9;
const MAX_RECEIPT_BYTES: u64 = 1024 * 1024;
const MAX_SCAN_DEPTH: usize = 6;
const MAX_DIRECTORY_ENTRIES: usize = 4096;
const MAX_RECEIPTS: usize = 512;
const MAX_CANDIDATES: usize = 512;
const MAX_WARNINGS: usize = 32;
const MAX_PUBLIC_NAME_CHARS: usize = 255;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LocalArtifactProvenance {
    ConversionReceipt,
    ManagedCache,
}

impl LocalArtifactProvenance {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ConversionReceipt => "local_receipt",
            Self::ManagedCache => "local_cache",
        }
    }

    const fn precedence(self) -> u8 {
        match self {
            Self::ConversionReceipt => 0,
            Self::ManagedCache => 1,
        }
    }
}

/// Exact server-side authority. `path` and `sha256` must never be serialized.
#[derive(Clone, Debug)]
pub struct LocalGgufArtifact {
    pub repository: String,
    pub revision: String,
    pub filename: String,
    pub root: PathBuf,
    pub path: PathBuf,
    /// Exact same-generation projector discovered from a paired conversion
    /// receipt or managed cache entry. Kept server-private.
    pub projector_path: Option<PathBuf>,
    pub bytes: u64,
    pub sha256: String,
    pub quant_hint: String,
    pub quant: Option<QuantType>,
    pub selectable: bool,
    pub unavailable_reason: Option<String>,
    pub provenance: LocalArtifactProvenance,
}

#[derive(Clone, Debug, Default)]
pub struct LocalArtifactCatalog {
    pub artifacts: Vec<LocalGgufArtifact>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
struct LocalArtifactRoot {
    /// Absolute lexical path. It may not exist at server startup; every scan
    /// resolves and validates it without following a symlink root.
    path: PathBuf,
}

#[derive(Clone, Debug, Default)]
pub struct LocalArtifactInventory {
    roots: Vec<LocalArtifactRoot>,
}

impl LocalArtifactInventory {
    /// Track the server startup directory's `models/` plus explicitly
    /// configured roots. A conventional root may be created after startup;
    /// explicit roots must already be safe directories so typos fail loudly.
    pub fn for_serve(explicit_roots: &[PathBuf]) -> Result<Self> {
        if explicit_roots.len() + 2 > MAX_ROOTS {
            bail!("at most {} local model roots may be configured", MAX_ROOTS);
        }
        let cwd = std::env::current_dir().context("resolve server working directory")?;
        let mut roots = vec![
            LocalArtifactRoot {
                path: lexical_absolute(&cwd.join("models"), &cwd)?,
            },
            LocalArtifactRoot {
                path: lexical_absolute(&crate::model_spec::managed_model_root()?, &cwd)?,
            },
        ];
        for path in explicit_roots {
            let absolute = lexical_absolute(path, &cwd)?;
            validate_existing_root(&absolute)?;
            roots.push(LocalArtifactRoot { path: absolute });
        }
        roots.sort_by(|left, right| left.path.cmp(&right.path));
        roots.dedup_by(|left, right| left.path == right.path);
        Ok(Self { roots })
    }

    #[cfg(test)]
    fn for_test(roots: Vec<PathBuf>) -> Self {
        let cwd = std::env::current_dir().unwrap();
        Self {
            roots: roots
                .into_iter()
                .map(|path| LocalArtifactRoot {
                    path: lexical_absolute(&path, &cwd).unwrap(),
                })
                .collect(),
        }
    }

    /// Discover one repository, or every repository when `repository` is
    /// `None`. The cache snapshot must come from the server's canonical cache
    /// root; cache entries outside that layout are rejected.
    pub fn discover(
        &self,
        repository: Option<&str>,
        cache: Option<(&Path, &CacheManifest)>,
    ) -> LocalArtifactCatalog {
        let mut artifacts = BTreeMap::<String, LocalGgufArtifact>::new();
        let mut conflicts = BTreeSet::<String>::new();
        let mut warnings = Vec::new();
        if let Some((root, manifest)) = cache {
            for model in manifest.models.values() {
                if repository.is_none_or(|expected| expected == model.repo_id) {
                    discover_cache_model(
                        root,
                        model,
                        &mut artifacts,
                        &mut conflicts,
                        &mut warnings,
                    );
                }
            }
        }
        for root in &self.roots {
            discover_receipts(
                root,
                repository,
                &mut artifacts,
                &mut conflicts,
                &mut warnings,
            );
            if artifacts.len() >= MAX_CANDIDATES {
                push_warning(&mut warnings, "local artifact candidate limit reached");
                break;
            }
        }
        bind_local_projectors(&mut artifacts, &mut warnings);
        let mut artifacts = artifacts.into_values().collect::<Vec<_>>();
        artifacts.sort_by(|left, right| {
            (!left.selectable)
                .cmp(&(!right.selectable))
                .then_with(|| {
                    left.provenance
                        .precedence()
                        .cmp(&right.provenance.precedence())
                })
                .then_with(|| left.repository.cmp(&right.repository))
                .then_with(|| left.quant_hint.cmp(&right.quant_hint))
                .then_with(|| left.bytes.cmp(&right.bytes))
                .then_with(|| left.filename.cmp(&right.filename))
        });
        LocalArtifactCatalog {
            artifacts,
            warnings,
        }
    }
}

fn lexical_absolute(path: &Path, cwd: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    bail!("local model root escapes the filesystem root");
                }
            }
            Component::Normal(value) => normalized.push(value),
        }
    }
    if !normalized.is_absolute() {
        bail!("local model root did not resolve to an absolute path");
    }
    Ok(normalized)
}

fn validate_existing_root(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspect local model root {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!(
            "local model root must be a non-symlink directory: {}",
            path.display()
        );
    }
    Ok(())
}

fn canonical_scan_root(root: &LocalArtifactRoot) -> Option<PathBuf> {
    let metadata = fs::symlink_metadata(&root.path).ok()?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return None;
    }
    root.path.canonicalize().ok()
}

fn discover_receipts(
    root: &LocalArtifactRoot,
    repository: Option<&str>,
    artifacts: &mut BTreeMap<String, LocalGgufArtifact>,
    conflicts: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) {
    let Some(canonical_root) = canonical_scan_root(root) else {
        return;
    };
    let mut queue = VecDeque::from([(canonical_root.clone(), 0usize)]);
    let mut visited = 0usize;
    let mut receipts = 0usize;
    while let Some((directory, depth)) = queue.pop_front() {
        if visited >= MAX_DIRECTORY_ENTRIES || receipts >= MAX_RECEIPTS {
            push_warning(warnings, "local receipt scan reached its bounded limit");
            break;
        }
        let Ok(read_dir) = fs::read_dir(&directory) else {
            continue;
        };
        let mut entries = read_dir
            .filter_map(std::result::Result::ok)
            .collect::<Vec<_>>();
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            visited += 1;
            if visited > MAX_DIRECTORY_ENTRIES {
                break;
            }
            let Ok(metadata) = fs::symlink_metadata(entry.path()) else {
                continue;
            };
            if metadata.file_type().is_symlink() {
                continue;
            }
            if metadata.is_dir() {
                if depth < MAX_SCAN_DEPTH {
                    queue.push_back((entry.path(), depth + 1));
                }
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            if !metadata.is_file() || !name.ends_with(RECEIPT_SUFFIX) {
                continue;
            }
            receipts += 1;
            if metadata.len() > MAX_RECEIPT_BYTES {
                push_warning(warnings, "ignored an oversized local conversion receipt");
                continue;
            }
            match inspect_receipt(&canonical_root, &entry.path(), repository) {
                Ok(Some(artifact)) => insert_artifact(artifacts, conflicts, artifact, warnings),
                Ok(None) => {}
                Err(error) => {
                    tracing::debug!(receipt = %entry.path().display(), %error, "ignored invalid local artifact receipt");
                    push_warning(warnings, "ignored an invalid local conversion receipt");
                }
            }
            if artifacts.len() >= MAX_CANDIDATES {
                return;
            }
        }
    }
}

fn inspect_receipt(
    root: &Path,
    receipt_path: &Path,
    expected_repository: Option<&str>,
) -> Result<Option<LocalGgufArtifact>> {
    let bytes =
        crate::core::bounded_file::read_bounded_regular_nofollow(receipt_path, MAX_RECEIPT_BYTES)
            .context("read receipt")?
            .context("receipt is not one stable bounded regular file")?;
    let receipt: ConversionReceipt =
        serde_json::from_slice(&bytes).context("invalid JSON/schema")?;
    if receipt.schema_version != CONVERSION_RECEIPT_SCHEMA_VERSION
        || expected_repository.is_some_and(|expected| expected != receipt.source.repository_id)
    {
        return Ok(None);
    }
    validate_receipt_identity(&receipt)?;
    let receipt_name = receipt_path
        .file_name()
        .and_then(|name| name.to_str())
        .context("receipt filename is not UTF-8")?;
    let artifact_name = receipt_name
        .strip_suffix(".receipt.json")
        .context("receipt does not name a sibling GGUF")?;
    validate_public_name(artifact_name)?;
    let artifact_path = receipt_path.with_file_name(artifact_name);
    let metadata = fs::symlink_metadata(&artifact_path).context("sibling GGUF is missing")?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!("sibling GGUF must be a non-symlink regular file");
    }
    if metadata.len() != receipt.output.size {
        bail!("sibling GGUF size no longer matches receipt");
    }
    let canonical = artifact_path
        .canonicalize()
        .context("canonicalize sibling GGUF")?;
    if !canonical.starts_with(root) {
        bail!("sibling GGUF escapes configured model root");
    }

    // `output.path` is deliberately not dereferenced: receipts are movable
    // beside their exact digest-bound GGUF. The sibling filename, root
    // containment, size, and deferred SHA authority define activation.
    let receipt_quant = QuantType::from_canonical_str(&receipt.quant_selector).ok();
    let quant_hint = bounded_quant_hint(&receipt.quant_selector)?;
    let (quant, selectable, unavailable_reason) = match receipt_quant {
        Some(expected) => {
            let header = mlx_native::gguf::GgufFile::open(&canonical)
                .context("cannot parse supported GGUF header")?;
            let actual = header
                .metadata_u32("general.file_type")
                .and_then(quant_from_file_type);
            if actual != Some(expected) {
                bail!("receipt quant does not match GGUF header");
            }
            (Some(expected), true, None)
        }
        None => (
            None,
            false,
            Some("GGUF quant is not supported by the current mlx-native diagnostic loader".into()),
        ),
    };
    Ok(Some(LocalGgufArtifact {
        repository: receipt.source.repository_id,
        revision: receipt.source.revision.to_ascii_lowercase(),
        filename: artifact_name.to_owned(),
        root: root.to_path_buf(),
        path: canonical,
        projector_path: None,
        bytes: receipt.output.size,
        sha256: receipt.output.sha256.to_ascii_lowercase(),
        quant_hint,
        quant,
        selectable,
        unavailable_reason,
        provenance: LocalArtifactProvenance::ConversionReceipt,
    }))
}

pub(crate) fn validate_receipt_identity(receipt: &ConversionReceipt) -> Result<()> {
    if receipt.source.repository_type != "model"
        || receipt.converter.package != "hf2q"
        || !crate::serve::auto_pipeline::looks_like_hf_repo_id(&receipt.source.repository_id)
    {
        bail!("receipt is not an hf2q Hugging Face model-conversion receipt");
    }
    if !is_hex(&receipt.source.revision, 40)
        || !is_hex(&receipt.source.bundle_sha256, 64)
        || !is_hex(&receipt.converter.git_commit, 40)
        || !is_hex(&receipt.output.sha256, 64)
    {
        bail!("receipt contains malformed immutable identity");
    }
    Ok(())
}

fn discover_cache_model(
    cache_root: &Path,
    model: &ModelEntry,
    artifacts: &mut BTreeMap<String, LocalGgufArtifact>,
    conflicts: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) {
    if model.repo_id.is_empty()
        || !crate::serve::auto_pipeline::looks_like_hf_repo_id(&model.repo_id)
    {
        return;
    }
    let Ok(canonical_root) = cache_root.canonicalize() else {
        return;
    };
    for (quant_name, entry) in &model.quantizations {
        let Ok(quant) = QuantType::from_canonical_str(quant_name) else {
            continue;
        };
        if !entry.quant_type.eq_ignore_ascii_case(quant.as_str()) || !is_hex(&entry.sha256, 64) {
            continue;
        }
        let Ok(expected) = cache_model_path(&canonical_root, &model.repo_id, quant) else {
            continue;
        };
        let Ok(metadata) = fs::symlink_metadata(&entry.gguf_path) else {
            continue;
        };
        if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() != entry.bytes
        {
            push_warning(warnings, "ignored a stale managed-cache artifact");
            continue;
        }
        let Ok(canonical) = entry.gguf_path.canonicalize() else {
            continue;
        };
        let legacy =
            crate::serve::cache::legacy_cache_model_path(&canonical_root, &model.repo_id, quant);
        let is_legacy = canonical == legacy;
        if canonical != expected && canonical != legacy {
            push_warning(
                warnings,
                "ignored a managed-cache artifact outside canonical layout",
            );
            continue;
        }
        if is_legacy && !legacy_cache_authority_matches(&legacy, model, quant_name, entry) {
            push_warning(
                warnings,
                "ignored a legacy cache artifact without matching repository and quant companions",
            );
            continue;
        }
        let Ok(header) = mlx_native::gguf::GgufFile::open(&canonical) else {
            continue;
        };
        if header
            .metadata_u32("general.file_type")
            .and_then(quant_from_file_type)
            != Some(quant)
        {
            continue;
        }
        insert_artifact(
            artifacts,
            conflicts,
            LocalGgufArtifact {
                repository: model.repo_id.clone(),
                revision: bounded_cache_revision(&model.revision),
                filename: "model.gguf".into(),
                root: canonical_root.clone(),
                path: canonical,
                projector_path: None,
                bytes: entry.bytes,
                sha256: entry.sha256.to_ascii_lowercase(),
                quant_hint: quant.as_str().to_owned(),
                quant: Some(quant),
                selectable: true,
                unavailable_reason: None,
                provenance: LocalArtifactProvenance::ManagedCache,
            },
            warnings,
        );

        if let Some(projector_path) = entry.mmproj_path.as_ref() {
            let expected_projector_path = if is_legacy {
                legacy
                    .parent()
                    .expect("validated legacy cache artifact has a quant directory")
                    .join("mmproj.gguf")
            } else {
                let Ok(path) = cache_mmproj_path(&canonical_root, &model.repo_id, quant) else {
                    continue;
                };
                path
            };
            let Ok(projector_metadata) = fs::symlink_metadata(projector_path) else {
                push_warning(warnings, "ignored a stale managed-cache projector");
                continue;
            };
            if projector_metadata.file_type().is_symlink() || !projector_metadata.is_file() {
                push_warning(warnings, "ignored a non-regular managed-cache projector");
                continue;
            }
            let Ok(canonical_projector) = projector_path.canonicalize() else {
                continue;
            };
            if canonical_projector != expected_projector_path {
                push_warning(
                    warnings,
                    "ignored a managed-cache projector outside canonical layout",
                );
                continue;
            }
            let expected_sha = crate::core::provenance::projector_sha256(&header)
                .ok()
                .flatten();
            let Some(expected_sha) = expected_sha else {
                push_warning(warnings, "ignored an unbound managed-cache projector");
                continue;
            };
            insert_artifact(
                artifacts,
                conflicts,
                LocalGgufArtifact {
                    repository: model.repo_id.clone(),
                    revision: bounded_cache_revision(&model.revision),
                    filename: "mmproj.gguf".into(),
                    root: canonical_root.clone(),
                    path: canonical_projector,
                    projector_path: None,
                    bytes: projector_metadata.len(),
                    sha256: expected_sha,
                    quant_hint: "F16-MMPROJ".into(),
                    quant: None,
                    selectable: false,
                    unavailable_reason: Some(
                        "vision projector companion; selected through its bound text model".into(),
                    ),
                    provenance: LocalArtifactProvenance::ManagedCache,
                },
                warnings,
            );
        }
    }
}

fn bind_local_projectors(
    artifacts: &mut BTreeMap<String, LocalGgufArtifact>,
    warnings: &mut Vec<String>,
) {
    let companions = artifacts
        .values()
        .filter(|artifact| !artifact.selectable && artifact.quant_hint == "F16-MMPROJ")
        .map(|artifact| {
            (
                artifact.repository.clone(),
                artifact.revision.clone(),
                artifact.sha256.clone(),
                artifact.path.clone(),
            )
        })
        .collect::<Vec<_>>();
    for artifact in artifacts
        .values_mut()
        .filter(|artifact| artifact.selectable)
    {
        let expected = mlx_native::gguf::GgufFile::open(&artifact.path)
            .ok()
            .and_then(|gguf| {
                crate::core::provenance::projector_sha256(&gguf)
                    .ok()
                    .flatten()
            });
        let Some(expected) = expected else {
            continue;
        };
        let matches = companions
            .iter()
            .filter(|(repository, revision, sha256, path)| {
                repository == &artifact.repository
                    && revision == &artifact.revision
                    && sha256.eq_ignore_ascii_case(&expected)
                    && path.parent() == artifact.path.parent()
            })
            .collect::<Vec<_>>();
        if matches.len() == 1 {
            artifact.projector_path = Some(matches[0].3.clone());
        } else {
            artifact.selectable = false;
            artifact.unavailable_reason = Some(format!(
                "bound projector {expected} has {} exact local companions",
                matches.len()
            ));
            push_warning(
                warnings,
                "ignored an incomplete or ambiguous local multimodal pair",
            );
        }
    }
}

fn legacy_cache_authority_matches(
    artifact: &Path,
    model: &ModelEntry,
    quant_name: &str,
    entry: &QuantEntry,
) -> bool {
    const MAX_CACHE_COMPANION_BYTES: u64 = 1024 * 1024;
    let Some(quant_dir) = artifact.parent() else {
        return false;
    };
    let Some(model_dir) = quant_dir.parent().and_then(Path::parent) else {
        return false;
    };
    let repo_meta = crate::core::bounded_file::read_bounded_regular_nofollow(
        &model_dir.join("repo_meta.json"),
        MAX_CACHE_COMPANION_BYTES,
    )
    .ok()
    .flatten()
    .and_then(|bytes| serde_json::from_slice::<ModelEntry>(&bytes).ok());
    let quant_meta = crate::core::bounded_file::read_bounded_regular_nofollow(
        &quant_dir.join("manifest.json"),
        MAX_CACHE_COMPANION_BYTES,
    )
    .ok()
    .flatten()
    .and_then(|bytes| serde_json::from_slice::<QuantEntry>(&bytes).ok());
    repo_meta.is_some_and(|authority| {
        authority.repo_id == model.repo_id
            && authority.revision == model.revision
            && authority.quantizations.get(quant_name) == Some(entry)
    }) && quant_meta.as_ref() == Some(entry)
}

fn insert_artifact(
    artifacts: &mut BTreeMap<String, LocalGgufArtifact>,
    conflicts: &mut BTreeSet<String>,
    artifact: LocalGgufArtifact,
    warnings: &mut Vec<String>,
) {
    let key = format!("{}\0{}", artifact.repository, artifact.sha256);
    if conflicts.contains(&key) {
        return;
    }
    if let Some(existing) = artifacts.get(&key) {
        if existing.bytes != artifact.bytes || existing.quant != artifact.quant {
            artifacts.remove(&key);
            conflicts.insert(key);
            push_warning(warnings, "ignored conflicting local artifact authorities");
            return;
        }
        if existing.provenance.precedence() <= artifact.provenance.precedence() {
            return;
        }
    }
    artifacts.insert(key, artifact);
}

fn validate_public_name(value: &str) -> Result<()> {
    if value.is_empty()
        || value.chars().count() > MAX_PUBLIC_NAME_CHARS
        || value.chars().any(char::is_control)
        || value.contains('/')
        || value.contains('\\')
    {
        bail!("artifact filename is not safe for display");
    }
    Ok(())
}

fn bounded_quant_hint(value: &str) -> Result<String> {
    if value.is_empty()
        || value.len() > 32
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        bail!("receipt quant selector is malformed");
    }
    Ok(value.to_ascii_uppercase())
}

fn bounded_cache_revision(value: &str) -> String {
    if is_hex(value, 40) {
        value.to_ascii_lowercase()
    } else {
        "local".into()
    }
}

fn push_warning(warnings: &mut Vec<String>, warning: &str) {
    if warnings.len() < MAX_WARNINGS {
        warnings.push(warning.to_owned());
    }
}

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub(super) fn quant_from_file_type(file_type: u32) -> Option<QuantType> {
    QuantType::from_gguf_file_type(file_type)
}

#[cfg(test)]
mod tests;
