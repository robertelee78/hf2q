use super::*;

pub(super) enum PreparedProjectorSource {
    Existing(crate::core::bounded_file::StableRegularFile),
    HubCache(DownloadedHubArtifact),
    Local(inventory::ExactLooseFile),
    Hosted,
}

pub(super) struct PreparedProjector {
    pub(super) artifact: HubGgufArtifact,
    pub(super) destination: PathBuf,
    pub(super) source: PreparedProjectorSource,
    destination_parent: Option<crate::core::bounded_file::StableDirectory>,
    destination_name: std::ffi::OsString,
}

pub(super) fn prepare_projector_action(
    artifact: HubGgufArtifact,
    destination: PathBuf,
    model_dirs: &[PathBuf],
) -> Result<PreparedProjector> {
    let source =
        if resolution::verify_or_refuse_existing_hosted_destination(&destination, &artifact)? {
            let retained = crate::core::bounded_file::StableRegularFile::open_operator_path_exact(
                &destination,
                artifact.bytes,
            )?
            .context("exact projector destination changed after planning")?;
            PreparedProjectorSource::Existing(retained)
        } else if let Some(cached) = retain_cached_projector_download(&artifact)? {
            PreparedProjectorSource::HubCache(cached)
        } else if let Some(local) = find_matching_loose(&artifact, model_dirs)? {
            PreparedProjectorSource::Local(local)
        } else {
            PreparedProjectorSource::Hosted
        };
    let destination_name = destination
        .file_name()
        .context("projector destination has no filename")?
        .to_os_string();
    let destination_parent = if matches!(source, PreparedProjectorSource::Local(_)) {
        Some(crate::core::bounded_file::StableDirectory::create_and_open(
            destination
                .parent()
                .context("projector destination has no parent")?,
        )?)
    } else {
        None
    };
    Ok(PreparedProjector {
        artifact,
        destination,
        source,
        destination_parent,
        destination_name,
    })
}

impl PreparedProjector {
    pub(super) fn source_device_id(&self) -> Option<u64> {
        match &self.source {
            PreparedProjectorSource::HubCache(source) => Some(source.retained.device_id()),
            PreparedProjectorSource::Local(source) => Some(source.retained.device_id()),
            _ => None,
        }
    }

    pub(super) fn destination_device_id(&self) -> Option<u64> {
        match (&self.source, &self.destination_parent) {
            (PreparedProjectorSource::Existing(retained), _) => Some(retained.device_id()),
            (_, Some(destination)) => Some(destination.device_id()),
            _ => None,
        }
    }

    pub(super) fn destination_available_bytes(&self) -> Option<u64> {
        self.destination_parent
            .as_ref()
            .and_then(|destination| destination.available_bytes())
    }

    pub(super) fn destination_is_exact(&self) -> bool {
        matches!(self.source, PreparedProjectorSource::Existing(_))
    }

    pub(super) fn is_current(&self) -> Result<bool> {
        let source_current = match &self.source {
            PreparedProjectorSource::Existing(retained) => retained.is_stable()?,
            PreparedProjectorSource::HubCache(source) => source.retained.is_stable()?,
            PreparedProjectorSource::Local(source) => source.retained.is_stable()?,
            PreparedProjectorSource::Hosted => true,
        };
        Ok(source_current
            && self
                .destination_parent
                .as_ref()
                .map_or(Ok(true), |destination| destination.is_current())?)
    }
}

fn retain_cached_projector_download(
    artifact: &HubGgufArtifact,
) -> Result<Option<DownloadedHubArtifact>> {
    let Some(snapshot_path) = cached_hub_gguf_path(artifact) else {
        return Ok(None);
    };
    let source = retain_cached_projector_at(artifact, &snapshot_path)?;
    Ok(Some(DownloadedHubArtifact {
        snapshot_path,
        blob_path: source.path,
        retained: source.retained,
    }))
}

pub(super) fn retain_cached_projector_at(
    artifact: &HubGgufArtifact,
    snapshot_path: &Path,
) -> Result<inventory::ExactLooseFile> {
    let revision_dir = snapshot_path
        .ancestors()
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.eq_ignore_ascii_case(&artifact.revision))
                && path
                    .parent()
                    .and_then(Path::file_name)
                    .is_some_and(|name| name == "snapshots")
        })
        .context("cached projector is outside an exact-revision snapshot")?;
    let repository_cache = revision_dir
        .parent()
        .and_then(Path::parent)
        .context("cached projector snapshot has no repository cache root")?;
    let blob_root = repository_cache
        .join("blobs")
        .canonicalize()
        .context("cached projector repository has no canonical blob root")?;
    let canonical = snapshot_path
        .canonicalize()
        .context("resolve exact cached projector blob")?;
    if !canonical.starts_with(&blob_root) || canonical == blob_root {
        bail!(
            "cached projector snapshot escapes its repository blob store: {}",
            snapshot_path.display()
        );
    }
    let mut retained =
        crate::core::bounded_file::StableRegularFile::open_exact(&canonical, artifact.bytes)?
            .context("exact Hub-cache projector changed before planning")?;
    let digest = retained
        .sha256()?
        .context("exact Hub-cache projector changed while hashing")?;
    if !digest.eq_ignore_ascii_case(&artifact.sha256) {
        bail!(
            "exact Hub-cache projector failed SHA-256 verification: {}",
            snapshot_path.display()
        );
    }
    Ok(inventory::ExactLooseFile {
        path: canonical,
        retained,
    })
}

pub(super) fn materialize_prepared_projector(
    plan: PreparedProjector,
    candidate: &mut Candidate,
    warnings: &mut Vec<String>,
    progress: &mut StartupProgress<'_>,
) -> Result<PathBuf> {
    if !plan.is_current()? {
        bail!("prepared projector authority changed before activation");
    }
    match plan.source {
        PreparedProjectorSource::Existing(retained) => {
            if !retained.is_stable()? {
                bail!("exact projector destination changed before activation");
            }
        }
        PreparedProjectorSource::HubCache(source) => materialize_hub_cache_symlink(
            source,
            &plan.destination,
            plan.artifact.bytes,
            &plan.artifact.sha256,
        )?,
        PreparedProjectorSource::Local(source) => materialize::materialize_retained_exact_at(
            source.retained,
            plan.destination_parent
                .context("prepared local projector has no destination authority")?,
            &plan.destination_name,
            &plan.artifact.repository,
            plan.artifact.bytes,
            &plan.artifact.sha256,
        )?,
        PreparedProjectorSource::Hosted => {
            let source = download_projector_with_progress(&plan.artifact, progress)?;
            materialize_hub_cache_symlink(
                source,
                &plan.destination,
                plan.artifact.bytes,
                &plan.artifact.sha256,
            )?;
        }
    }
    candidate.projector = Some((
        plan.destination.clone(),
        plan.artifact.bytes,
        plan.artifact.sha256.clone(),
    ));
    persist_candidate_projector(candidate, &plan.destination, &plan.artifact, warnings);
    Ok(plan.destination)
}

pub(super) fn download_projector_with_progress(
    artifact: &HubGgufArtifact,
    progress: &mut StartupProgress<'_>,
) -> Result<DownloadedHubArtifact> {
    let filename = display_filename(Path::new(&artifact.filename));
    Ok(download_hub_companion_with_progress(
        artifact,
        &mut |update| {
            progress(StartupEvent::HostedDownloadProgress {
                filename: filename.clone(),
                completed_bytes: update.completed_bytes,
                total_bytes: update.total_bytes,
                bytes_per_second: update.bytes_per_second,
                elapsed_ms: update.elapsed_ms,
            });
        },
    )?)
}

#[cfg(test)]
pub(super) fn resolve_projector(
    candidate: &mut Candidate,
    model_dirs: &[PathBuf],
    warnings: &mut Vec<String>,
) -> Result<Option<PathBuf>> {
    if !text_requires_projector(&candidate.path)? {
        return Ok(None);
    }
    match verify_candidate_projector(candidate) {
        Ok(Some(path)) => return Ok(Some(path)),
        Ok(None) => {}
        Err(error) => warnings.push(format!(
            "local mmproj verification failed; serving text-only: {error}"
        )),
    }
    let reference = HfModelReference::parse(&candidate.repository, Some(&candidate.revision))?;
    match resolve_hub_gguf_catalog(reference) {
        Ok(catalog) => Ok(best_effort_projector_with_catalog(
            candidate,
            model_dirs,
            &catalog,
            catalog.requires_projector,
            warnings,
        )),
        Err(error) => {
            warnings.push(format!(
                "multimodal projector metadata unavailable; serving text-only: {error}"
            ));
            Ok(None)
        }
    }
}

#[cfg(test)]
pub(super) fn best_effort_projector_with_catalog(
    candidate: &mut Candidate,
    model_dirs: &[PathBuf],
    catalog: &HubGgufCatalog,
    repository_requires_projector: bool,
    warnings: &mut Vec<String>,
) -> Option<PathBuf> {
    best_effort_projector_with_catalog_with_progress(
        candidate,
        model_dirs,
        catalog,
        repository_requires_projector,
        warnings,
        &mut |_| {},
    )
}

pub(super) fn best_effort_projector_with_catalog_with_progress(
    candidate: &mut Candidate,
    model_dirs: &[PathBuf],
    catalog: &HubGgufCatalog,
    repository_requires_projector: bool,
    warnings: &mut Vec<String>,
    progress: &mut StartupProgress<'_>,
) -> Option<PathBuf> {
    best_effort_projector_with_catalog_expected_with_progress(
        candidate,
        model_dirs,
        catalog,
        repository_requires_projector,
        None,
        warnings,
        progress,
    )
}

pub(super) fn best_effort_projector_with_catalog_expected_with_progress(
    candidate: &mut Candidate,
    model_dirs: &[PathBuf],
    catalog: &HubGgufCatalog,
    repository_requires_projector: bool,
    retained_expected_projector_sha256: Option<&str>,
    warnings: &mut Vec<String>,
    progress: &mut StartupProgress<'_>,
) -> Option<PathBuf> {
    match resolve_projector_with_catalog_requirement(
        candidate,
        model_dirs,
        catalog,
        repository_requires_projector,
        retained_expected_projector_sha256,
        warnings,
        progress,
    ) {
        Ok(path) => path,
        Err(error) => {
            warnings.push(format!(
                "automatic mmproj preparation failed; serving text-only: {error}"
            ));
            None
        }
    }
}

#[cfg(test)]
pub(super) fn resolve_projector_with_catalog(
    candidate: &mut Candidate,
    model_dirs: &[PathBuf],
    catalog: &HubGgufCatalog,
    warnings: &mut Vec<String>,
) -> Result<Option<PathBuf>> {
    resolve_projector_with_catalog_requirement(
        candidate,
        model_dirs,
        catalog,
        catalog.requires_projector,
        None,
        warnings,
        &mut |_| {},
    )
}

fn resolve_projector_with_catalog_requirement(
    candidate: &mut Candidate,
    model_dirs: &[PathBuf],
    catalog: &HubGgufCatalog,
    repository_requires_projector: bool,
    retained_expected_projector_sha256: Option<&str>,
    warnings: &mut Vec<String>,
    progress: &mut StartupProgress<'_>,
) -> Result<Option<PathBuf>> {
    if !repository_requires_projector && !text_requires_projector(&candidate.path)? {
        return Ok(None);
    }
    if let Some(path) = verify_candidate_projector(candidate)? {
        let matches = candidate
            .projector
            .as_ref()
            .is_some_and(|(bound, _, sha256)| {
                bound == &path
                    && retained_expected_projector_sha256
                        .is_none_or(|expected| sha256.eq_ignore_ascii_case(expected))
            });
        if matches {
            return Ok(Some(path));
        }
        candidate.projector = None;
    }
    let expected = retained_expected_projector_sha256
        .map(str::to_owned)
        .map_or_else(
            || expected_projector_sha256(&candidate.path),
            |value| Ok(Some(value)),
        )?;
    let companions = catalog
        .artifacts
        .iter()
        .filter(|artifact| artifact.role == "companion")
        .filter(|artifact| {
            expected
                .as_deref()
                .is_none_or(|sha| artifact.sha256.eq_ignore_ascii_case(sha))
        })
        .collect::<Vec<_>>();
    let Some(artifact) = select_projector_companion(candidate, companions, expected.as_deref())?
    else {
        warnings.push(
            "multimodal text model has no unambiguous matching hosted mmproj; serving text-only"
                .into(),
        );
        return Ok(None);
    };
    let parent = candidate.path.parent().context("text GGUF has no parent")?;
    let destination = parent.join(safe_basename(&artifact.filename)?);
    if !resolution::verify_or_refuse_existing_hosted_destination(&destination, &artifact)? {
        let source = match find_matching_loose(&artifact, model_dirs)? {
            Some(source) => source,
            None => {
                check_hub_artifact_plan(&artifact, &destination)?;
                let source = download_projector_with_progress(&artifact, progress)?;
                materialize_hub_cache_symlink(
                    source,
                    &destination,
                    artifact.bytes,
                    &artifact.sha256,
                )?;
                candidate.projector =
                    Some((destination.clone(), artifact.bytes, artifact.sha256.clone()));
                persist_candidate_projector(candidate, &destination, &artifact, warnings);
                return Ok(Some(destination));
            }
        };
        materialize_retained_exact(
            source.retained,
            &destination,
            &artifact.repository,
            artifact.bytes,
            &artifact.sha256,
        )?;
    }
    candidate.projector = Some((destination.clone(), artifact.bytes, artifact.sha256.clone()));
    persist_candidate_projector(candidate, &destination, &artifact, warnings);
    Ok(Some(destination))
}

fn persist_candidate_projector(
    candidate: &Candidate,
    destination: &Path,
    artifact: &HubGgufArtifact,
    warnings: &mut Vec<String>,
) {
    let Some(sidecar) = candidate.sidecar.as_ref() else {
        return;
    };
    let persist = || -> Result<()> {
        let mut binding = read_binding(sidecar)?.context("managed text binding disappeared")?;
        binding.projector = Some(ArtifactBinding {
            local_filename: destination
                .file_name()
                .and_then(|name| name.to_str())
                .context("mmproj filename is not UTF-8")?
                .to_owned(),
            hub_filename: artifact.filename.clone(),
            bytes: artifact.bytes,
            sha256: artifact.sha256.clone(),
        });
        write_binding(sidecar, &binding)
    };
    if let Err(error) = persist() {
        warnings.push(format!(
            "verified mmproj will be loaded, but its use history could not be persisted: {error}"
        ));
    }
}

pub(super) fn select_projector_companion(
    candidate: &Candidate,
    companions: Vec<&HubGgufArtifact>,
    expected_sha256: Option<&str>,
) -> Result<Option<HubGgufArtifact>> {
    match companions.as_slice() {
        [] => return Ok(None),
        [artifact] => return Ok(Some((**artifact).clone())),
        _ => {}
    }

    let text_filename = candidate
        .sidecar
        .as_ref()
        .and_then(|sidecar| read_binding(sidecar).ok().flatten())
        .map(|binding| binding.artifact.hub_filename)
        .or_else(|| {
            candidate
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_owned)
        })
        .context("selected multimodal text artifact has no UTF-8 filename")?;
    let text_basename = safe_basename(&text_filename)?;
    let text_stem = text_basename
        .strip_suffix(".gguf")
        .or_else(|| text_basename.strip_suffix(".GGUF"))
        .unwrap_or(text_basename);
    let paired_name = format!("{text_stem}-mmproj.gguf");
    let exact_name = companions
        .iter()
        .copied()
        .filter(|artifact| {
            safe_basename(&artifact.filename)
                .is_ok_and(|name| name.eq_ignore_ascii_case(&paired_name))
        })
        .collect::<Vec<_>>();
    if let [artifact] = exact_name.as_slice() {
        return Ok(Some((**artifact).clone()));
    }

    if expected_sha256.is_none() {
        let generic = companions
            .iter()
            .copied()
            .filter(|artifact| {
                safe_basename(&artifact.filename)
                    .is_ok_and(|name| name.to_ascii_lowercase().starts_with("mmproj-"))
            })
            .collect::<Vec<_>>();
        if let [artifact] = generic.as_slice() {
            return Ok(Some((**artifact).clone()));
        }
    }
    Ok(None)
}
