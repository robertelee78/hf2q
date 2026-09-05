use super::resolution::automatic_artifact_admissible;
use super::*;

pub(super) struct ExactHostedLocal {
    pub(super) artifact: HubGgufArtifact,
    pub(super) path: PathBuf,
    pub(super) materialized: SystemTime,
    pub(super) requires_projector: bool,
    pub(super) retained: crate::core::bounded_file::StableRegularFile,
}

#[cfg(test)]
pub(super) fn find_best_matching_loose(
    artifacts: &[HubGgufArtifact],
    exact: Option<QuantType>,
    model_dirs: &[PathBuf],
    excluded: &[crate::core::bounded_file::StableFileIdentity],
    warnings: &mut Vec<String>,
) -> Result<Option<ExactHostedLocal>> {
    let mut silent = |_| {};
    find_best_matching_loose_with_progress(
        artifacts,
        exact,
        model_dirs,
        excluded,
        warnings,
        &mut silent,
    )
}

pub(super) fn find_best_matching_loose_with_progress(
    artifacts: &[HubGgufArtifact],
    exact: Option<QuantType>,
    model_dirs: &[PathBuf],
    excluded: &[crate::core::bounded_file::StableFileIdentity],
    warnings: &mut Vec<String>,
    progress: &mut StartupProgress<'_>,
) -> Result<Option<ExactHostedLocal>> {
    find_best_matching_loose_structural(artifacts, model_dirs, exact, excluded, warnings, progress)
}

fn find_best_matching_loose_structural(
    artifacts: &[HubGgufArtifact],
    model_dirs: &[PathBuf],
    exact: Option<QuantType>,
    excluded: &[crate::core::bounded_file::StableFileIdentity],
    warnings: &mut Vec<String>,
    progress: &mut StartupProgress<'_>,
) -> Result<Option<ExactHostedLocal>> {
    let eligible = artifacts
        .iter()
        .filter(|artifact| {
            artifact
                .quant_hint
                .as_deref()
                .and_then(|value| QuantType::from_canonical_str(value).ok())
                .is_some_and(|quant| exact.is_none_or(|expected| expected == quant))
        })
        .collect::<Vec<_>>();
    let mut candidates = Vec::new();
    let mut seen = Vec::new();
    for root in scan_roots(model_dirs)? {
        visit_files(&root, |path, metadata, retained| {
            if excluded
                .iter()
                .any(|identity| identity.same_inode(retained.identity()))
                || seen
                    .iter()
                    .any(|identity: &crate::core::bounded_file::StableFileIdentity| {
                        identity.same_inode(retained.identity())
                    })
                || path
                    .extension()
                    .and_then(|value| value.to_str())
                    .is_none_or(|extension| !extension.eq_ignore_ascii_case("gguf"))
            {
                return Ok(());
            }
            seen.push(retained.identity());
            let matching = eligible
                .iter()
                .filter(|artifact| artifact.bytes == metadata.len())
                .copied()
                .collect::<Vec<_>>();
            let [artifact] = matching.as_slice() else {
                if matching.len() > 1 {
                    warnings.push(format!(
                        "ignored structurally ambiguous local GGUF {}: multiple repository artifacts have the same quant and byte length; filenames are hints, not identity authority",
                        path.display()
                    ));
                }
                return Ok(());
            };
            candidates.push((
                metadata.modified().unwrap_or(UNIX_EPOCH),
                path.to_path_buf(),
                (**artifact).clone(),
                retained,
            ));
            Ok(())
        })?;
    }
    candidates.sort_by(|left, right| right.0.cmp(&left.0));
    for (materialized, path, artifact, retained) in candidates {
        let quant = artifact
            .quant_hint
            .as_deref()
            .and_then(|value| QuantType::from_canonical_str(value).ok())
            .context("structurally matched local GGUF has no supported quant")?;
        progress(StartupEvent::LocalCandidate {
            quant: quant.as_str().to_owned(),
            origin: StartupOrigin::ManualStructuralMatch,
            bytes: artifact.bytes,
            filename: display_filename(&path),
        });
        let compatibility =
            match validate_retained_local_hub_gguf_compatibility(&retained, &artifact) {
                Ok(compatibility) => compatibility,
                Err(error) => {
                    warnings.push(format!(
                        "ignored unsupported local GGUF {}: {error}",
                        path.display()
                    ));
                    continue;
                }
            };
        if let Err(error) = validate_retained_local_runtime_tensor_layout(&retained) {
            warnings.push(format!(
                "ignored non-executable local GGUF {}: {error}",
                path.display()
            ));
            continue;
        }
        return Ok(Some(ExactHostedLocal {
            artifact,
            path,
            materialized,
            requires_projector: compatibility.requires_projector,
            retained,
        }));
    }
    Ok(None)
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
        |_, retained, _| {
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
        |path, retained, _| hash(path, retained.try_clone()?.metadata()?.len()),
    )
    .map(|selected| selected.map(|selected| (selected.artifact, selected.path)))
}

#[cfg(test)]
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
    mut hash: impl FnMut(
        &Path,
        &mut crate::core::bounded_file::StableRegularFile,
        &[HubGgufArtifact],
    ) -> Result<String>,
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
        let digest = match hash(&path, &mut retained, &artifacts) {
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

pub(super) fn find_best_matching_cached_hub_with_progress(
    artifacts: &[HubGgufArtifact],
    exact: Option<QuantType>,
    warnings: &mut Vec<String>,
    progress: &mut StartupProgress<'_>,
) -> Result<Option<ExactHostedLocal>> {
    let mut candidates = Vec::new();
    for artifact in artifacts {
        let quant = artifact
            .quant_hint
            .as_deref()
            .and_then(|value| QuantType::from_canonical_str(value).ok());
        let Some(quant) = quant else {
            continue;
        };
        if exact.is_some_and(|expected| expected != quant) {
            continue;
        }
        let Some(local) = retain_cached_hub_artifact(artifact)? else {
            continue;
        };
        let materialized = local
            .retained
            .try_clone()?
            .metadata()?
            .modified()
            .unwrap_or(UNIX_EPOCH);
        candidates.push((materialized, artifact.clone(), local, quant));
    }
    candidates.sort_by(|left, right| right.0.cmp(&left.0));
    for (materialized, artifact, local, quant) in candidates {
        let path = local.path;
        let retained = local.retained;
        progress(StartupEvent::LocalCandidate {
            quant: quant.as_str().to_owned(),
            origin: StartupOrigin::HuggingFaceCacheStructuralMatch,
            bytes: artifact.bytes,
            filename: display_filename(&path),
        });
        let compatibility =
            match validate_retained_local_hub_gguf_compatibility(&retained, &artifact) {
                Ok(compatibility) => compatibility,
                Err(error) => {
                    warnings.push(format!(
                        "ignored unsupported Hugging Face cache GGUF {}: {error}",
                        path.display()
                    ));
                    continue;
                }
            };
        if let Err(error) = validate_retained_local_runtime_tensor_layout(&retained) {
            warnings.push(format!(
                "ignored non-executable Hugging Face cache GGUF {}: {error}",
                path.display()
            ));
            continue;
        }
        return Ok(Some(ExactHostedLocal {
            artifact,
            path,
            materialized,
            requires_projector: compatibility.requires_projector,
            retained,
        }));
    }
    Ok(None)
}

pub(super) fn retain_cached_hub_artifact(
    artifact: &HubGgufArtifact,
) -> Result<Option<inventory::ExactLooseFile>> {
    let Some(snapshot_path) = cached_hub_gguf_path(artifact) else {
        return Ok(None);
    };
    let revision_dir = snapshot_path.ancestors().find(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.eq_ignore_ascii_case(&artifact.revision))
            && path
                .parent()
                .and_then(Path::file_name)
                .is_some_and(|name| name == "snapshots")
    });
    let Some(repository_cache) = revision_dir.and_then(Path::parent).and_then(Path::parent) else {
        return Ok(None);
    };
    let blob_root = match repository_cache.join("blobs").canonicalize() {
        Ok(path) => path,
        Err(_) => return Ok(None),
    };
    let canonical = match snapshot_path.canonicalize() {
        Ok(path) if path.starts_with(&blob_root) && path != blob_root => path,
        _ => return Ok(None),
    };
    let Some(retained) =
        crate::core::bounded_file::StableRegularFile::open_exact(&canonical, artifact.bytes)?
    else {
        return Ok(None);
    };
    Ok(Some(inventory::ExactLooseFile {
        path: canonical,
        retained,
    }))
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
        |_, _, retained| {
            retained
                .sha256()?
                .context("Hub-cache GGUF changed while it was being verified")
        },
    )
    .map(|selected| selected.map(|selected| (selected.artifact, selected.path)))
}

#[cfg(test)]
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
    mut hash: impl FnMut(
        &Path,
        &HubGgufArtifact,
        &mut crate::core::bounded_file::StableRegularFile,
    ) -> Result<String>,
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
        let digest = match hash(&path, &artifact, &mut retained) {
            Ok(digest) => digest,
            Err(error) => {
                warnings.push(format!(
                    "ignored unstable Hub-cache artifact {}: {error}",
                    path.display()
                ));
                continue;
            }
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

pub(super) fn hash_hosted_local_candidate(
    path: &Path,
    bytes: u64,
    quant: Option<QuantType>,
    origin: StartupOrigin,
    retained: &mut crate::core::bounded_file::StableRegularFile,
    progress: &mut StartupProgress<'_>,
) -> Result<String> {
    if let Some(quant) = quant {
        progress(StartupEvent::LocalCandidate {
            quant: quant.as_str().to_owned(),
            origin,
            bytes,
            filename: display_filename(path),
        });
    } else {
        progress(StartupEvent::VerifyStart {
            artifact: "text GGUF".into(),
            bytes,
            filename: display_filename(path),
        });
    }
    let started = std::time::Instant::now();
    let step = (bytes / 20).max(256 * 1024 * 1024);
    let mut next_report = step.min(bytes);
    retained
        .sha256_with_progress(|completed_bytes| {
            if completed_bytes >= next_report || completed_bytes == bytes {
                progress(StartupEvent::VerifyProgress {
                    artifact: "text GGUF".into(),
                    completed_bytes,
                    total_bytes: bytes,
                    elapsed_ms: started.elapsed().as_millis() as u64,
                });
                next_report = completed_bytes.saturating_add(step).min(bytes);
            }
        })?
        .context("local GGUF changed or ceased to be a stable regular file")
}

#[cfg(test)]
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
        crate::core::bounded_file::StableRegularFile,
        Option<CacheLock>,
    )>,
> {
    let mut silent = |_| {};
    select_local_with_progress(
        spec,
        model_dirs,
        cache,
        held_quant_lock,
        available_memory_bytes,
        pool_budget_bytes,
        warnings,
        &mut silent,
    )
}

pub(super) fn select_local_with_progress(
    spec: &RepositoryModelSpec,
    model_dirs: &[PathBuf],
    cache: &ModelCache,
    held_quant_lock: Option<QuantType>,
    available_memory_bytes: u64,
    pool_budget_bytes: u64,
    warnings: &mut Vec<String>,
    progress: &mut StartupProgress<'_>,
) -> Result<
    Option<(
        Candidate,
        crate::core::bounded_file::StableRegularFile,
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
            hub_filename: None,
            quant,
            origin: artifact.provenance.as_str().to_owned(),
            materialized_at_secs: materialized,
            last_used_at_secs: last_used,
            projector,
            sidecar: None,
            receipt_target_identity: None,
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
        match verify_candidate_with_progress(&candidate, progress) {
            Ok(authority) => return Ok(Some((candidate, authority, local_lock))),
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
    let selector_matches = spec.requested_selector().is_none_or(|_| {
        candidate.hub_filename.as_deref().map_or_else(
            || spec.quant == Some(candidate.quant),
            |filename| spec.matches_hosted_filename(filename),
        )
    });
    selector_matches
        && held_quant_lock.is_none_or(|quant| candidate.quant == quant)
        && spec.quant.map_or_else(
            || {
                spec.requested_selector().is_some()
                    || automatic_artifact_admissible(
                        candidate.bytes,
                        available_memory_bytes,
                        pool_budget_bytes,
                    )
            },
            |quant| candidate.quant == quant,
        )
}

#[cfg(test)]
mod selector_tests {
    use super::*;

    fn candidate(hub_filename: Option<&str>) -> Candidate {
        Candidate {
            repository: "owner/model".into(),
            revision: "a".repeat(40),
            path: PathBuf::from("model.gguf"),
            root: PathBuf::from("."),
            bytes: 1,
            sha256: "b".repeat(64),
            hub_filename: hub_filename.map(str::to_owned),
            quant: QuantType::Q8_0,
            origin: "test".into(),
            materialized_at_secs: 0,
            last_used_at_secs: 0,
            projector: None,
            sidecar: None,
            receipt_target_identity: None,
        }
    }

    #[test]
    fn publisher_selector_reuses_only_matching_hosted_binding() {
        let spec = crate::model_spec::parse_repository_spec("owner/model:UD-Q8_K_XL").unwrap();
        assert!(local_candidate_eligible(
            &spec,
            &candidate(Some("model-UD-Q8_K_XL.gguf")),
            None,
            u64::MAX,
            u64::MAX,
        ));
        assert!(!local_candidate_eligible(
            &spec,
            &candidate(Some("model-Q8_0.gguf")),
            None,
            u64::MAX,
            u64::MAX,
        ));
        assert!(!local_candidate_eligible(
            &spec,
            &candidate(None),
            None,
            u64::MAX,
            u64::MAX,
        ));
    }

    #[test]
    fn canonical_selector_can_reuse_matching_conversion_receipt() {
        let spec = crate::model_spec::parse_repository_spec("owner/model:Q8_0").unwrap();
        assert!(local_candidate_eligible(
            &spec,
            &candidate(None),
            None,
            u64::MAX,
            u64::MAX,
        ));
    }
}
