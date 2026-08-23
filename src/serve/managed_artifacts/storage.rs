use super::*;

pub(super) fn scan_bindings(
    model_dirs: &[PathBuf],
    repository: Option<&str>,
) -> Result<Vec<Candidate>> {
    let mut candidates = Vec::new();
    for root in scan_roots(model_dirs)? {
        visit_files(&root, |path, _| {
            if path.to_string_lossy().ends_with(SIDECAR_SUFFIX) {
                if let Ok(Some(binding)) = read_binding(path) {
                    if repository.is_none_or(|expected| expected == binding.repository) {
                        if let Ok(candidate) = candidate_from_binding(
                            binding,
                            sibling_from_sidecar(path)?,
                            path.to_owned(),
                        ) {
                            candidates.push(candidate);
                        }
                    }
                }
            }
            Ok(())
        })?;
    }
    Ok(candidates)
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
        || receipt.converter.package != "hf2q"
        || receipt.output.size != artifact_metadata.len()
        || !crate::serve::auto_pipeline::looks_like_hf_repo_id(&receipt.source.repository_id)
        || !is_hex(&receipt.source.revision, 40)
        || !is_hex(&receipt.output.sha256, 64)
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
        || receipt.converter.package != "hf2q"
        || receipt.source.repository_id != expected_repository
        || !receipt
            .source
            .revision
            .eq_ignore_ascii_case(expected_revision)
        || !receipt.quant_selector.eq_ignore_ascii_case("f16-mmproj")
        || receipt.output.size != metadata.len()
        || !is_hex(&receipt.output.sha256, 64)
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
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata)
            if metadata.is_file()
                && !metadata.file_type().is_symlink()
                && metadata.len() <= MAX_CONVERSION_RECEIPT_BYTES =>
        {
            metadata
        }
        _ => return Ok(None),
    };
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    fs::File::open(path)?
        .take(MAX_CONVERSION_RECEIPT_BYTES + 1)
        .read_to_end(&mut bytes)?;
    Ok(Some(serde_json::from_slice(&bytes)?))
}

pub(super) fn read_binding(path: &Path) -> Result<Option<ManagedBinding>> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata)
            if metadata.is_file()
                && !metadata.file_type().is_symlink()
                && metadata.len() <= MAX_SIDECAR_BYTES =>
        {
            metadata
        }
        _ => return Ok(None),
    };
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    fs::File::open(path)?
        .take(MAX_SIDECAR_BYTES + 1)
        .read_to_end(&mut bytes)?;
    let binding: ManagedBinding = serde_json::from_slice(&bytes)?;
    validate_binding(&binding)?;
    Ok(Some(binding))
}

pub(super) fn write_binding(path: &Path, binding: &ManagedBinding) -> Result<()> {
    validate_binding(binding)?;
    let parent = path.parent().context("managed binding has no parent")?;
    fs::create_dir_all(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut temporary, binding)?;
    temporary.write_all(b"\n")?;
    temporary.as_file_mut().sync_all()?;
    temporary.persist(path).map_err(|error| error.error)?;
    Ok(())
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
