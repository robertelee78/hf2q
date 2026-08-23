use super::*;

pub(super) fn find_best_matching_loose(
    artifacts: &[HubGgufArtifact],
    exact: Option<QuantType>,
    recommended: QuantType,
    model_dirs: &[PathBuf],
) -> Result<Option<(HubGgufArtifact, PathBuf)>> {
    let eligible = artifacts
        .iter()
        .filter_map(|artifact| {
            let quant = artifact
                .quant_hint
                .as_deref()
                .and_then(|value| QuantType::from_canonical_str(value).ok())?;
            exact
                .map_or_else(
                    || quant_quality(quant) <= quant_quality(recommended),
                    |expected| quant == expected,
                )
                .then_some((quant, artifact))
        })
        .collect::<Vec<_>>();
    let mut candidates = Vec::new();
    let mut seen = BTreeSet::new();
    for root in scan_roots(model_dirs)? {
        visit_files(&root, |path, metadata| {
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
            let quant = match quant_type_from_gguf_path(path) {
                Ok(quant) => quant,
                Err(_) => return Ok(()),
            };
            let modified = metadata.modified().unwrap_or(UNIX_EPOCH);
            for (expected_quant, artifact) in matching {
                if *expected_quant == quant {
                    candidates.push((modified, (**artifact).clone(), path.to_path_buf()));
                }
            }
            Ok(())
        })?;
    }
    candidates.sort_by(|left, right| right.0.cmp(&left.0));
    for (_, artifact, path) in candidates {
        if crate::core::sha256::compute_file_sha256(&path)?.eq_ignore_ascii_case(&artifact.sha256) {
            return Ok(Some((artifact, path)));
        }
    }
    Ok(None)
}

pub(super) fn select_local(
    spec: &RepositoryModelSpec,
    model_dirs: &[PathBuf],
    cache: &ModelCache,
    recommended: QuantType,
    held_quant_lock: Option<QuantType>,
    warnings: &mut Vec<String>,
) -> Result<Option<(Candidate, Option<CacheLock>)>> {
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
                    .map(|entry| (model.last_accessed_secs, entry.mmproj_path.clone()))
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
        held_quant_lock.is_none_or(|quant| candidate.quant == quant)
            && spec.quant.map_or_else(
                || quant_quality(candidate.quant) <= quant_quality(recommended),
                |quant| candidate.quant == quant,
            )
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
            Ok(()) => return Ok(Some((candidate, local_lock))),
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
