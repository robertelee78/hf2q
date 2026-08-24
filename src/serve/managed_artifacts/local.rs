use super::resolution::automatic_artifact_admissible;
use super::*;

pub(super) struct ExactHostedLocal {
    pub(super) artifact: HubGgufArtifact,
    #[cfg(test)]
    pub(super) path: PathBuf,
    pub(super) materialized: SystemTime,
    pub(super) requires_projector: bool,
    pub(super) retained: crate::core::bounded_file::StableRegularFile,
}

pub(super) fn find_best_matching_loose(
    artifacts: &[HubGgufArtifact],
    exact: Option<QuantType>,
    model_dirs: &[PathBuf],
    excluded: &[crate::core::bounded_file::StableFileIdentity],
    warnings: &mut Vec<String>,
) -> Result<Option<ExactHostedLocal>> {
    find_best_matching_loose_stable(
        artifacts,
        exact,
        model_dirs,
        excluded,
        warnings,
        |_, artifact, retained| {
            let compatibility = validate_retained_local_hub_gguf_compatibility(retained, artifact)
                .map_err(|error| error.to_string())?;
            validate_retained_local_runtime_tensor_layout(retained)
                .map_err(|error| error.to_string())?;
            Ok(compatibility.requires_projector)
        },
        |_, retained| {
            retained
                .sha256()?
                .context("manual GGUF changed or ceased to be a stable regular file")
        },
    )
}

#[cfg(test)]
pub(super) fn find_best_matching_loose_with(
    artifacts: &[HubGgufArtifact],
    exact: Option<QuantType>,
    model_dirs: &[PathBuf],
    warnings: &mut Vec<String>,
    mut admit: impl FnMut(&Path, &HubGgufArtifact) -> std::result::Result<(), String>,
) -> Result<Option<(HubGgufArtifact, PathBuf)>> {
    find_best_matching_loose_stable(
        artifacts,
        exact,
        model_dirs,
        &[],
        warnings,
        |path, artifact, _| admit(path, artifact).map(|_| false),
        |_, retained| {
            retained
                .sha256()?
                .context("manual GGUF changed or ceased to be a stable regular file")
        },
    )
    .map(|selected| selected.map(|selected| (selected.artifact, selected.path)))
}

#[cfg(test)]
pub(super) fn find_best_matching_loose_with_hash(
    artifacts: &[HubGgufArtifact],
    exact: Option<QuantType>,
    model_dirs: &[PathBuf],
    warnings: &mut Vec<String>,
    mut admit: impl FnMut(&Path, &HubGgufArtifact) -> std::result::Result<(), String>,
    mut hash: impl FnMut(&Path, u64) -> Result<String>,
) -> Result<Option<(HubGgufArtifact, PathBuf)>> {
    find_best_matching_loose_stable(
        artifacts,
        exact,
        model_dirs,
        &[],
        warnings,
        |path, artifact, _| admit(path, artifact).map(|_| false),
        |path, retained| hash(path, retained.try_clone()?.metadata()?.len()),
    )
    .map(|selected| selected.map(|selected| (selected.artifact, selected.path)))
}

fn find_best_matching_loose_stable(
    artifacts: &[HubGgufArtifact],
    exact: Option<QuantType>,
    model_dirs: &[PathBuf],
    excluded: &[crate::core::bounded_file::StableFileIdentity],
    warnings: &mut Vec<String>,
    mut admit: impl FnMut(
        &Path,
        &HubGgufArtifact,
        &crate::core::bounded_file::StableRegularFile,
    ) -> std::result::Result<bool, String>,
    mut hash: impl FnMut(&Path, &mut crate::core::bounded_file::StableRegularFile) -> Result<String>,
) -> Result<Option<ExactHostedLocal>> {
    let eligible = artifacts
        .iter()
        .filter_map(|artifact| {
            let quant = artifact
                .quant_hint
                .as_deref()
                .and_then(|value| QuantType::from_canonical_str(value).ok())?;
            exact
                .map_or(true, |expected| quant == expected)
                .then_some((quant, artifact))
        })
        .collect::<Vec<_>>();
    let mut candidates = Vec::new();
    let mut seen = BTreeSet::new();
    for root in scan_roots(model_dirs)? {
        visit_files(&root, |path, metadata, file| {
            if excluded
                .iter()
                .any(|identity| identity.same_inode(file.identity()))
            {
                return Ok(());
            }
            if !seen.insert(path.to_path_buf())
                || path
                    .extension()
                    .and_then(|value| value.to_str())
                    .is_none_or(|extension| !extension.eq_ignore_ascii_case("gguf"))
            {
                return Ok(());
            }
            let matching = eligible
                .iter()
                .filter(|(_, artifact)| artifact.bytes == metadata.len())
                .collect::<Vec<_>>();
            if matching.is_empty() {
                return Ok(());
            }
            let modified = metadata.modified().unwrap_or(UNIX_EPOCH);
            let matching = matching
                .into_iter()
                .map(|(_, artifact)| (**artifact).clone())
                .collect::<Vec<_>>();
            if !matching.is_empty() {
                candidates.push((modified, path.to_path_buf(), matching, file));
            }
            Ok(())
        })?;
    }
    candidates.sort_by(|left, right| right.0.cmp(&left.0));
    for (_, path, artifacts, mut retained) in candidates {
        let digest = match hash(&path, &mut retained) {
            Ok(digest) => digest,
            Err(error) => {
                warnings.push(format!(
                    "ignored unstable manually downloaded GGUF {}: {error}",
                    path.display()
                ));
                continue;
            }
        };
        let matching = artifacts
            .into_iter()
            .filter(|artifact| digest.eq_ignore_ascii_case(&artifact.sha256))
            .collect::<Vec<_>>();
        if matching.len() > 1 {
            let filenames = matching
                .iter()
                .map(|artifact| artifact.filename.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            bail!(
                "manually downloaded bytes match multiple hosted identities and remain ambiguous: {filenames}"
            );
        }
        if let Some(artifact) = matching.into_iter().next() {
            match admit(&path, &artifact, &retained) {
                Ok(requires_projector) => {
                    return Ok(Some(ExactHostedLocal {
                        artifact,
                        #[cfg(test)]
                        path,
                        materialized: retained
                            .try_clone()?
                            .metadata()?
                            .modified()
                            .unwrap_or(UNIX_EPOCH),
                        requires_projector,
                        retained,
                    }))
                }
                Err(reason) => warnings.push(format!(
                    "ignored incompatible manually downloaded {} before adoption: {reason}",
                    artifact.filename
                )),
            }
        }
    }
    Ok(None)
}

pub(super) fn find_best_matching_cached_hub(
    artifacts: &[HubGgufArtifact],
    exact: Option<QuantType>,
    warnings: &mut Vec<String>,
) -> Result<Option<ExactHostedLocal>> {
    find_best_matching_cached_hub_stable(
        artifacts,
        exact,
        warnings,
        |artifact| {
            let path = cached_hub_gguf_path(artifact)?;
            let materialized = path
                .metadata()
                .ok()
                .and_then(|metadata| metadata.modified().ok())
                .unwrap_or(UNIX_EPOCH);
            Some((path, materialized))
        },
        |_, artifact, retained| {
            let compatibility = validate_retained_local_hub_gguf_compatibility(retained, artifact)
                .map_err(|error| error.to_string())?;
            validate_retained_local_runtime_tensor_layout(retained)
                .map_err(|error| error.to_string())?;
            Ok(compatibility.requires_projector)
        },
    )
}

#[cfg(test)]
pub(super) fn find_best_matching_cached_hub_with(
    artifacts: &[HubGgufArtifact],
    exact: Option<QuantType>,
    warnings: &mut Vec<String>,
    mut lookup: impl FnMut(&HubGgufArtifact) -> Option<(PathBuf, SystemTime)>,
    mut admit: impl FnMut(&Path, &HubGgufArtifact) -> std::result::Result<(), String>,
) -> Result<Option<(HubGgufArtifact, PathBuf)>> {
    find_best_matching_cached_hub_stable(
        artifacts,
        exact,
        warnings,
        &mut lookup,
        |path, artifact, _| admit(path, artifact).map(|_| false),
    )
    .map(|selected| selected.map(|selected| (selected.artifact, selected.path)))
}

fn find_best_matching_cached_hub_stable(
    artifacts: &[HubGgufArtifact],
    exact: Option<QuantType>,
    warnings: &mut Vec<String>,
    mut lookup: impl FnMut(&HubGgufArtifact) -> Option<(PathBuf, SystemTime)>,
    mut admit: impl FnMut(
        &Path,
        &HubGgufArtifact,
        &crate::core::bounded_file::StableRegularFile,
    ) -> std::result::Result<bool, String>,
) -> Result<Option<ExactHostedLocal>> {
    let mut candidates = artifacts
        .iter()
        .filter_map(|artifact| {
            let quant = artifact
                .quant_hint
                .as_deref()
                .and_then(|value| QuantType::from_canonical_str(value).ok())?;
            if exact.is_some_and(|expected| expected != quant) {
                return None;
            }
            let (path, materialized) = lookup(artifact)?;
            Some((materialized, artifact.clone(), path))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.0.cmp(&left.0));
    for (materialized, artifact, path) in candidates {
        let Some(mut retained) =
            crate::core::bounded_file::StableRegularFile::open_exact(&path, artifact.bytes)?
        else {
            warnings.push(format!(
                "ignored unstable Hub-cache artifact {}",
                path.display()
            ));
            continue;
        };
        let Some(digest) = retained.sha256()? else {
            warnings.push(format!(
                "ignored unstable Hub-cache artifact {}",
                path.display()
            ));
            continue;
        };
        if !digest.eq_ignore_ascii_case(&artifact.sha256) {
            warnings.push(format!(
                "ignored corrupted Hub-cache artifact {}: SHA-256 does not match the exact repository catalog",
                path.display()
            ));
            continue;
        }
        match admit(&path, &artifact, &retained) {
            Ok(requires_projector) => {
                return Ok(Some(ExactHostedLocal {
                    artifact,
                    #[cfg(test)]
                    path,
                    materialized,
                    requires_projector,
                    retained,
                }))
            }
            Err(reason) => warnings.push(format!(
                "ignored incompatible Hub-cache artifact {}: {reason}",
                path.display()
            )),
        }
    }
    Ok(None)
}

pub(super) fn select_local(
    spec: &RepositoryModelSpec,
    model_dirs: &[PathBuf],
    cache: &ModelCache,
    held_quant_lock: Option<QuantType>,
    available_memory_bytes: u64,
    pool_budget_bytes: u64,
    warnings: &mut Vec<String>,
) -> Result<
    Option<(
        Candidate,
        crate::core::bounded_file::StableFileIdentity,
        Option<CacheLock>,
    )>,
> {
    let inventory = LocalArtifactInventory::for_serve(model_dirs)?;
    let cache_manifest = cache.manifest_snapshot()?;
    let mut candidates = scan_bindings(model_dirs, Some(&spec.repository))?;
    let catalog = inventory.discover(
        Some(&spec.repository),
        Some((cache.root(), &cache_manifest)),
    );
    warnings.extend(catalog.warnings);
    for artifact in catalog
        .artifacts
        .into_iter()
        .filter(|artifact| artifact.selectable)
    {
        let Some(quant) = artifact.quant else {
            continue;
        };
        let (last_used, projector) = cache_manifest
            .models
            .get(&artifact.repository)
            .and_then(|model| {
                model
                    .quantizations
                    .get(quant.as_str())
                    .map(|entry| (entry.last_used_at_secs, entry.mmproj_path.clone()))
            })
            .unwrap_or((0, None));
        let materialized = artifact
            .path
            .metadata()
            .ok()
            .and_then(|metadata| metadata.modified().ok())
            .map(system_time_secs)
            .unwrap_or(0);
        let receipt_projector = paired_projector_path(&artifact.path).and_then(|path| {
            projector_authority_from_receipt(&path, &artifact.repository, &artifact.revision)
                .ok()
                .flatten()
        });
        let projector = projector
            .and_then(|path| {
                projector_authority_from_receipt(&path, &artifact.repository, &artifact.revision)
                    .ok()
                    .flatten()
            })
            .or(receipt_projector);
        candidates.push(Candidate {
            repository: artifact.repository,
            revision: artifact.revision,
            path: artifact.path,
            root: artifact.root,
            bytes: artifact.bytes,
            sha256: artifact.sha256,
            quant,
            origin: artifact.provenance.as_str().to_owned(),
            materialized_at_secs: materialized,
            last_used_at_secs: last_used,
            projector,
            sidecar: None,
        });
    }
    candidates.retain(|candidate| {
        let eligible = local_candidate_eligible(
            spec,
            &candidate,
            held_quant_lock,
            available_memory_bytes,
            pool_budget_bytes,
        );
        if !eligible
            && spec.quant.is_none()
            && !automatic_artifact_admissible(
                candidate.bytes,
                available_memory_bytes,
                pool_budget_bytes,
            )
        {
            warnings.push(format!(
                "ignored local {} {} ({} bytes): current automatic admission budget is {} bytes",
                candidate.quant,
                candidate.path.display(),
                candidate.bytes,
                available_memory_bytes
            ));
        }
        eligible
    });
    candidates.sort_by(|left, right| candidate_recency(right).cmp(&candidate_recency(left)));
    let mut seen = BTreeSet::new();
    for candidate in candidates {
        if !seen.insert(candidate.path.clone()) {
            continue;
        }
        let local_lock = if held_quant_lock == Some(candidate.quant) {
            None
        } else {
            Some(
                cache
                    .lock_quant(&spec.repository, candidate.quant)
                    .with_context(|| {
                        format!(
                            "lock local resolution for {}:{}",
                            spec.repository, candidate.quant
                        )
                    })?,
            )
        };
        match verify_candidate(&candidate) {
            Ok(identity) => return Ok(Some((candidate, identity, local_lock))),
            Err(error) => {
                warnings.push(format!(
                    "ignored invalid local {} {}: {error}",
                    candidate.quant,
                    candidate.path.display()
                ));
            }
        }
    }
    Ok(None)
}

pub(super) fn local_candidate_eligible(
    spec: &RepositoryModelSpec,
    candidate: &Candidate,
    held_quant_lock: Option<QuantType>,
    available_memory_bytes: u64,
    pool_budget_bytes: u64,
) -> bool {
    held_quant_lock.is_none_or(|quant| candidate.quant == quant)
        && spec.quant.map_or_else(
            || {
                automatic_artifact_admissible(
                    candidate.bytes,
                    available_memory_bytes,
                    pool_budget_bytes,
                )
            },
            |quant| candidate.quant == quant,
        )
}
