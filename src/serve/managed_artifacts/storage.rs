use super::*;

pub(super) fn scan_bindings(
    model_dirs: &[PathBuf],
    repository: Option<&str>,
) -> Result<Vec<Candidate>> {
    let mut candidates = Vec::new();
    for root in scan_roots(model_dirs)? {
        visit_files(&root, |path, _, mut file| {
            if path.to_string_lossy().ends_with(SIDECAR_SUFFIX) {
                if file.has_operator_symlink_leaf() {
                    return Ok(());
                }
                if let Ok(Some(bytes)) = file.read_bounded(MAX_SIDECAR_BYTES) {
                    let Ok(binding) = serde_json::from_slice::<ManagedBinding>(&bytes) else {
                        return Ok(());
                    };
                    if validate_binding(&binding).is_err() {
                        return Ok(());
                    }
                    if repository.is_none_or(|expected| expected == binding.repository) {
                        let artifact = sibling_from_sidecar(path)?;
                        // A regular conversion output keeps its schema-v3
                        // receipt as repository/artifact authority. Its
                        // sidecar may contribute recency only in the receipt
                        // branch below; never admit it as a second managed
                        // candidate with projector authority.
                        if conversion_authority(&artifact).ok().flatten().is_some() {
                            return Ok(());
                        }
                        if let Ok(candidate) =
                            candidate_from_binding(binding, artifact, path.to_owned())
                        {
                            candidates.push(candidate);
                        }
                    }
                }
            } else if path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
            {
                // A final-leaf symlink may point at an hf2q conversion in an
                // operator library. Authenticate the small adjacent receipt
                // at the exact retained target, but preserve the logical link
                // as the displayed/managed model path. The text payload is
                // not hashed on this serve-in-place path.
                let Some(authority_path) = file.canonical_path_for_identity()? else {
                    return Ok(());
                };
                let Ok(Some(mut candidate)) = conversion_authority(&authority_path) else {
                    return Ok(());
                };
                {
                    if file.canonical_path_for_identity()?.as_deref()
                        != Some(authority_path.as_path())
                    {
                        return Ok(());
                    }
                    if repository.is_some_and(|expected| expected != candidate.repository.as_str())
                    {
                        return Ok(());
                    }
                    if let Some((projector_target, bytes, sha256)) = candidate.projector.clone() {
                        if let Some(logical_projector) =
                            logical_alias_for_target(path, &projector_target, bytes)?
                        {
                            candidate.projector = Some((logical_projector, bytes, sha256));
                        }
                    }
                    candidate.path = path.to_path_buf();
                    candidate.receipt_target_identity = Some(file.identity());
                    candidate.root = path
                        .parent()
                        .context("linked conversion artifact has no parent")?
                        .to_path_buf();
                    let usage_sidecar = sidecar_path(path);
                    if let Ok(Some(binding)) = read_binding(&usage_sidecar) {
                        if binding.repository == candidate.repository
                            && binding.revision.eq_ignore_ascii_case(&candidate.revision)
                            && binding.quant.eq_ignore_ascii_case(candidate.quant.as_str())
                            && path.file_name().and_then(|name| name.to_str())
                                == Some(binding.artifact.local_filename.as_str())
                            && binding.artifact.bytes == candidate.bytes
                            && binding
                                .artifact
                                .sha256
                                .eq_ignore_ascii_case(&candidate.sha256)
                        {
                            candidate.last_used_at_secs = binding.last_used_at_secs;
                        }
                    }
                    candidates.push(candidate);
                }
            }
            Ok(())
        })?;
    }
    Ok(candidates)
}

fn logical_alias_for_target(
    logical_text: &Path,
    target: &Path,
    expected_bytes: u64,
) -> Result<Option<PathBuf>> {
    let Some(expected) =
        crate::core::bounded_file::StableRegularFile::open_exact(target, expected_bytes)?
    else {
        return Ok(None);
    };
    let parent = logical_text
        .parent()
        .context("linked text artifact has no parent")?;
    let paired = paired_projector_path(logical_text);
    let mut matches = Vec::new();
    for (index, entry) in fs::read_dir(parent)?.enumerate() {
        if index >= 512 {
            break;
        }
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        if path == logical_text
            || path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_none_or(|extension| !extension.eq_ignore_ascii_case("gguf"))
        {
            continue;
        }
        let Some(alias) = crate::core::bounded_file::StableRegularFile::open_operator_path_exact(
            &path,
            expected_bytes,
        )?
        else {
            continue;
        };
        if alias.identity() == expected.identity() && alias.is_stable()? {
            matches.push(path);
        }
    }
    matches.sort();
    if let Some(paired) = paired.filter(|path| matches.iter().any(|candidate| candidate == path)) {
        return Ok(Some(paired));
    }
    Ok(match matches.as_slice() {
        [path] => Some(path.clone()),
        _ => None,
    })
}

pub(super) fn candidate_from_binding(
    binding: ManagedBinding,
    path: PathBuf,
    sidecar: PathBuf,
) -> Result<Candidate> {
    validate_binding(&binding)?;
    if path.file_name().and_then(|name| name.to_str())
        != Some(binding.artifact.local_filename.as_str())
    {
        bail!("managed sidecar filename does not match its sibling artifact");
    }
    let metadata = fs::symlink_metadata(&path)
        .with_context(|| format!("managed artifact is missing: {}", path.display()))?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() != binding.artifact.bytes
    {
        bail!("managed artifact is not an exact-size non-symlink regular file");
    }
    let root = path
        .parent()
        .context("managed artifact has no parent")?
        .to_path_buf();
    let projector = binding.projector.as_ref().and_then(|projector| {
        let projector_path = path.with_file_name(&projector.local_filename);
        let metadata = fs::symlink_metadata(&projector_path).ok()?;
        (metadata.is_file()
            && !metadata.file_type().is_symlink()
            && metadata.len() == projector.bytes)
            .then(|| (projector_path, projector.bytes, projector.sha256.clone()))
    });
    Ok(Candidate {
        repository: binding.repository,
        revision: binding.revision,
        path,
        root,
        bytes: binding.artifact.bytes,
        sha256: binding.artifact.sha256,
        quant: QuantType::from_canonical_str(&binding.quant).map_err(|error| anyhow!(error))?,
        origin: binding.origin,
        materialized_at_secs: binding.materialized_at_secs,
        last_used_at_secs: binding.last_used_at_secs,
        projector,
        sidecar: Some(sidecar),
        receipt_target_identity: None,
    })
}

pub(super) fn conversion_authority(path: &Path) -> Result<Option<Candidate>> {
    use crate::convert::receipt::{receipt_path, CONVERSION_RECEIPT_SCHEMA_VERSION};
    let artifact_metadata = match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => metadata,
        _ => return Ok(None),
    };
    let Some(receipt) = read_conversion_receipt(&receipt_path(path))? else {
        return Ok(None);
    };
    if receipt.schema_version != CONVERSION_RECEIPT_SCHEMA_VERSION
        || crate::serve::api::local_artifacts::validate_receipt_identity(&receipt).is_err()
        || receipt.output.size != artifact_metadata.len()
    {
        return Ok(None);
    }
    let quant = match QuantType::from_canonical_str(&receipt.quant_selector) {
        Ok(quant) => quant,
        Err(_) => return Ok(None),
    };
    let root = path
        .parent()
        .context("conversion output has no parent")?
        .to_path_buf();
    let projector = paired_projector_path(path).and_then(|projector| {
        projector_authority_from_receipt(
            &projector,
            &receipt.source.repository_id,
            &receipt.source.revision,
        )
        .ok()
        .flatten()
    });
    Ok(Some(Candidate {
        repository: receipt.source.repository_id,
        revision: receipt.source.revision,
        path: path.to_path_buf(),
        root,
        bytes: receipt.output.size,
        sha256: receipt.output.sha256,
        quant,
        origin: LocalArtifactProvenance::ConversionReceipt
            .as_str()
            .to_owned(),
        materialized_at_secs: artifact_metadata
            .modified()
            .ok()
            .map(system_time_secs)
            .unwrap_or(0),
        last_used_at_secs: 0,
        projector,
        sidecar: None,
        receipt_target_identity: None,
    }))
}

pub(super) fn projector_authority_from_receipt(
    path: &Path,
    expected_repository: &str,
    expected_revision: &str,
) -> Result<Option<(PathBuf, u64, String)>> {
    use crate::convert::receipt::{receipt_path, CONVERSION_RECEIPT_SCHEMA_VERSION};
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => metadata,
        _ => return Ok(None),
    };
    let Some(receipt) = read_conversion_receipt(&receipt_path(path))? else {
        return Ok(None);
    };
    if receipt.schema_version != CONVERSION_RECEIPT_SCHEMA_VERSION
        || crate::serve::api::local_artifacts::validate_receipt_identity(&receipt).is_err()
        || receipt.source.repository_id != expected_repository
        || !receipt
            .source
            .revision
            .eq_ignore_ascii_case(expected_revision)
        || !receipt.quant_selector.eq_ignore_ascii_case("f16-mmproj")
        || receipt.output.size != metadata.len()
    {
        return Ok(None);
    }
    Ok(Some((
        path.to_path_buf(),
        receipt.output.size,
        receipt.output.sha256,
    )))
}

fn read_conversion_receipt(
    path: &Path,
) -> Result<Option<crate::convert::receipt::ConversionReceipt>> {
    let Some(bytes) = crate::core::bounded_file::read_bounded_regular_nofollow(
        path,
        MAX_CONVERSION_RECEIPT_BYTES,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(serde_json::from_slice(&bytes)?))
}

pub(super) fn read_binding(path: &Path) -> Result<Option<ManagedBinding>> {
    let Some(bytes) =
        crate::core::bounded_file::read_bounded_regular_nofollow(path, MAX_SIDECAR_BYTES)?
    else {
        return Ok(None);
    };
    let binding: ManagedBinding = serde_json::from_slice(&bytes)?;
    validate_binding(&binding)?;
    Ok(Some(binding))
}

pub(super) fn write_binding(path: &Path, binding: &ManagedBinding) -> Result<()> {
    validate_binding(binding)?;
    let parent = path.parent().context("managed binding has no parent")?;
    fs::create_dir_all(parent)?;
    let _lock = crate::core::paired_artifact::PairLock::exclusive(path)
        .context("lock managed binding publication")?;
    let existing = match fs::symlink_metadata(path) {
        Ok(_) => {
            let existing = read_binding(path)?
                .context("refusing to replace an invalid or non-regular managed binding")?;
            if !same_binding_authority(&existing, binding) {
                bail!("refusing to replace a managed binding for different artifact authority");
            }
            Some(existing)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut temporary, binding)?;
    temporary.write_all(b"\n")?;
    temporary.as_file_mut().sync_all()?;
    if existing.is_some() {
        temporary.persist(path).map_err(|error| error.error)?;
    } else {
        temporary
            .persist_noclobber(path)
            .map_err(|error| error.error)?;
    }
    Ok(())
}

fn same_binding_authority(left: &ManagedBinding, right: &ManagedBinding) -> bool {
    left.schema_version == right.schema_version
        && left.repository == right.repository
        && left.revision.eq_ignore_ascii_case(&right.revision)
        && left.quant.eq_ignore_ascii_case(&right.quant)
        && left.artifact.local_filename == right.artifact.local_filename
        && left.artifact.hub_filename == right.artifact.hub_filename
        && left.artifact.bytes == right.artifact.bytes
        && left
            .artifact
            .sha256
            .eq_ignore_ascii_case(&right.artifact.sha256)
}

pub(super) fn validate_binding(binding: &ManagedBinding) -> Result<()> {
    if binding.schema_version != SCHEMA_VERSION
        || !crate::serve::auto_pipeline::looks_like_hf_repo_id(&binding.repository)
        || !is_hex(&binding.revision, 40)
        || !is_hex(&binding.artifact.sha256, 64)
        || binding.artifact.bytes == 0
    {
        bail!("managed artifact binding has invalid immutable identity");
    }
    safe_local_filename(&binding.artifact.local_filename)?;
    if let Some(projector) = &binding.projector {
        safe_local_filename(&projector.local_filename)?;
        if projector.bytes == 0 || !is_hex(&projector.sha256, 64) {
            bail!("managed projector binding has invalid immutable identity");
        }
    }
    QuantType::from_canonical_str(&binding.quant).map_err(|error| anyhow!(error))?;
    Ok(())
}

pub(super) fn sidecar_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(SIDECAR_SUFFIX);
    PathBuf::from(name)
}

fn sibling_from_sidecar(path: &Path) -> Result<PathBuf> {
    let rendered = path.as_os_str().to_string_lossy();
    let artifact = rendered
        .strip_suffix(SIDECAR_SUFFIX)
        .context("invalid managed sidecar suffix")?;
    Ok(PathBuf::from(artifact))
}

pub(super) fn safe_basename(filename: &str) -> Result<&str> {
    let basename = filename
        .rsplit('/')
        .next()
        .context("hosted filename is empty")?;
    safe_local_filename(basename)?;
    Ok(basename)
}

fn safe_local_filename(filename: &str) -> Result<()> {
    if filename.is_empty()
        || filename == "."
        || filename == ".."
        || filename
            .chars()
            .any(|character| matches!(character, '/' | '\\' | '\0'))
    {
        bail!("unsafe managed artifact filename");
    }
    Ok(())
}
