use super::*;

fn report_local_ready(candidate: &Candidate, progress: &mut StartupProgress<'_>) {
    progress(StartupEvent::LocalReady {
        quant: candidate.quant.as_str().to_owned(),
        origin: StartupOrigin::from_internal(&candidate.origin),
        filename: display_filename(&candidate.path),
    })
}

fn report_model_prepared(candidate: &Candidate, progress: &mut StartupProgress<'_>) {
    progress(StartupEvent::ModelPrepared {
        quant: candidate.quant.as_str().to_owned(),
        origin: StartupOrigin::from_internal(&candidate.origin),
        filename: display_filename(&candidate.path),
    })
}

fn prepare_cached_projector_in_place(
    candidate: &mut Candidate,
    text_authority: &crate::core::bounded_file::StableRegularFile,
    catalog: &HubGgufCatalog,
    progress: &mut StartupProgress<'_>,
) -> Result<Option<PathBuf>> {
    prepare_cached_projector_in_place_with_sources(
        candidate,
        text_authority,
        catalog,
        progress,
        cached_hub_gguf_path,
        |artifact, progress| {
            Ok(projector::download_projector_with_progress(artifact, progress)?.snapshot_path)
        },
    )
}

pub(super) fn prepare_cached_projector_in_place_with_sources(
    candidate: &mut Candidate,
    text_authority: &crate::core::bounded_file::StableRegularFile,
    catalog: &HubGgufCatalog,
    progress: &mut StartupProgress<'_>,
    mut cached: impl FnMut(&HubGgufArtifact) -> Option<PathBuf>,
    mut download: impl FnMut(&HubGgufArtifact, &mut StartupProgress<'_>) -> Result<PathBuf>,
) -> Result<Option<PathBuf>> {
    let expected = retained_expected_projector_sha256(text_authority)?;
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
    let Some(projector) = select_projector_companion(candidate, companions, expected.as_deref())?
    else {
        return Ok(None);
    };
    progress(StartupEvent::ProjectorPrepare {
        filename: display_filename(Path::new(&projector.filename)),
        bytes: projector.bytes,
    });
    // hf-hub returns a snapshot symlink after a fresh download. Authenticate
    // both cached and freshly downloaded pointers into the exact-revision
    // repository blob store before O_NOFOLLOW retained activation.
    let snapshot = match cached(&projector) {
        Some(snapshot) => snapshot,
        None => download(&projector, progress)?,
    };
    let path = retain_cached_projector_at(&projector, &snapshot)?.path;
    candidate.projector = Some((path.clone(), projector.bytes, projector.sha256.clone()));
    Ok(Some(path))
}

pub(super) fn bind_existing_local_projector(
    candidate: &mut Candidate,
    path: PathBuf,
) -> Result<PathBuf> {
    let metadata = fs::metadata(&path)?;
    if !metadata.is_file() {
        bail!("automatic local mmproj does not resolve to a regular file");
    }
    let bytes = metadata.len();
    let mut retained =
        crate::core::bounded_file::StableRegularFile::open_operator_path_exact(&path, bytes)?
            .context("automatic local mmproj changed before retained hashing")?;
    let sha256 = retained
        .sha256()?
        .context("automatic local mmproj changed while hashing")?;
    let gguf = mlx_native::gguf::GgufFile::from_file(retained.try_clone()?)
        .context("automatic local mmproj is not a readable GGUF")?;
    let config = crate::inference::vision::mmproj::MmprojConfig::from_gguf(&gguf)
        .context("automatic local mmproj has unsupported projector metadata")?;
    let tensor_names = gguf.tensor_names();
    crate::inference::vision::mmproj::validate_tensor_set(&config, &tensor_names)
        .context("automatic local mmproj has an incomplete projector tensor set")?;
    let profile = crate::inference::vision::mmproj::detect_arch_profile_with_projector(
        &config.projector,
        &tensor_names,
    );
    if !profile.is_supported() {
        bail!("automatic local mmproj has no supported runtime architecture profile");
    }
    if !retained.is_stable()? {
        bail!("automatic local mmproj changed during structural admission");
    }
    candidate.projector = Some((path.clone(), bytes, sha256));
    Ok(path)
}

pub(super) fn best_effort_manual_projector_with_catalog(
    candidate: &mut Candidate,
    text_authority: &crate::core::bounded_file::StableRegularFile,
    model_dirs: &[PathBuf],
    catalog: &HubGgufCatalog,
    warnings: &mut Vec<String>,
    progress: &mut StartupProgress<'_>,
) -> Option<PathBuf> {
    let expected = match retained_expected_projector_sha256(text_authority) {
        Ok(expected) => expected,
        Err(error) => {
            warnings.push(format!(
                "local text authority changed during mmproj planning; serving text-only: {error}"
            ));
            return None;
        }
    };
    let exact_companion = select_projector_companion(
        candidate,
        catalog
            .artifacts
            .iter()
            .filter(|artifact| artifact.role == "companion")
            .filter(|artifact| {
                expected
                    .as_deref()
                    .is_none_or(|sha| artifact.sha256.eq_ignore_ascii_case(sha))
            })
            .collect(),
        expected.as_deref(),
    )
    .ok()
    .flatten()
    .map(|artifact| {
        (
            artifact.filename.clone(),
            artifact.bytes,
            artifact.sha256.clone(),
        )
    });
    if let Some((filename, bytes, _)) = exact_companion.as_ref() {
        progress(StartupEvent::ProjectorPrepare {
            filename: display_filename(Path::new(filename)),
            bytes: *bytes,
        });
    }
    let local_error = match resolve_local_path_projector_required_with_expected(
        &candidate.path,
        expected.as_deref(),
    ) {
        Ok(Some(path)) => match bind_existing_local_projector(candidate, path) {
            Ok(path) => {
                let local_sha = candidate
                    .projector
                    .as_ref()
                    .map(|(_, _, sha256)| sha256.as_str());
                if let Some((filename, _, expected_sha)) = exact_companion.as_ref() {
                    if local_sha.is_none_or(|sha| !sha.eq_ignore_ascii_case(expected_sha)) {
                        warnings.push(format!(
                            "ignored structurally compatible local sibling mmproj because it does not match exact hosted companion {filename}"
                        ));
                        candidate.projector = None;
                        None
                    } else {
                        return Some(path);
                    }
                } else {
                    if let Some((bound, bytes, _)) = candidate.projector.as_ref() {
                        progress(StartupEvent::ProjectorPrepare {
                            filename: display_filename(bound),
                            bytes: *bytes,
                        });
                    }
                    return Some(path);
                }
            }
            Err(error) => {
                warnings.push(format!(
                    "ignored incompatible local sibling mmproj before hosted fallback: {error}"
                ));
                None
            }
        },
        Ok(None) => None,
        Err(error) => Some(error),
    };
    run_before_manual_hosted_projector_fallback();
    let hosted = best_effort_projector_with_catalog_expected_with_progress(
        candidate,
        model_dirs,
        catalog,
        true,
        expected.as_deref(),
        warnings,
        progress,
    );
    if exact_companion.is_none() && hosted.is_some() {
        if let Some((bound, bytes, _)) = candidate.projector.as_ref() {
            progress(StartupEvent::ProjectorPrepare {
                filename: display_filename(bound),
                bytes: *bytes,
            });
        }
    }
    if hosted.is_none() {
        if let Some(error) = local_error {
            warnings.push(format!(
                "automatic local sibling mmproj preparation failed; serving text-only: {error}"
            ));
        }
    }
    hosted
}

#[cfg(test)]
pub(crate) fn resolve_repository(
    spec: &RepositoryModelSpec,
    explicit_output: Option<&Path>,
    model_dirs: &[PathBuf],
    cache: &mut ModelCache,
    hardware: &HardwareProfile,
    prepare_projector: bool,
    configured_quant: Option<&str>,
) -> Result<ResolvedManagedModel> {
    let mut silent = |_| {};
    resolve_repository_with_progress(
        spec,
        explicit_output,
        model_dirs,
        cache,
        hardware,
        prepare_projector,
        configured_quant,
        &mut silent,
    )
}

pub(crate) fn resolve_repository_with_progress(
    spec: &RepositoryModelSpec,
    explicit_output: Option<&Path>,
    model_dirs: &[PathBuf],
    cache: &mut ModelCache,
    hardware: &HardwareProfile,
    prepare_projector: bool,
    configured_quant: Option<&str>,
    progress: &mut StartupProgress<'_>,
) -> Result<ResolvedManagedModel> {
    resolve_repository_with_progress_and_catalog(
        spec,
        explicit_output,
        model_dirs,
        cache,
        hardware,
        prepare_projector,
        configured_quant,
        progress,
        resolve_hub_gguf_catalog,
    )
}

pub(super) fn resolve_repository_with_progress_and_catalog(
    spec: &RepositoryModelSpec,
    explicit_output: Option<&Path>,
    model_dirs: &[PathBuf],
    cache: &mut ModelCache,
    hardware: &HardwareProfile,
    prepare_projector: bool,
    configured_quant: Option<&str>,
    progress: &mut StartupProgress<'_>,
    mut resolve_catalog: impl FnMut(HfModelReference) -> Result<HubGgufCatalog, DownloadError>,
) -> Result<ResolvedManagedModel> {
    progress(StartupEvent::LocalSearch {
        repository: spec.repository.clone(),
        requested_quant: spec.requested_selector().map(str::to_owned),
    });
    let pool_budget_bytes = LoadedPool::from_hardware(hardware).memory_budget_bytes();
    let mut warnings = Vec::new();
    let initial_local = select_local_with_progress(
        spec,
        model_dirs,
        cache,
        None,
        hardware.available_memory_bytes,
        pool_budget_bytes,
        &mut warnings,
        progress,
    )?;
    // A successfully used managed quant is the strongest automatic-choice
    // signal. It cannot be displaced by a merely newer loose/Hub-cache file,
    // so return it after the one verification already performed by
    // select_local. This avoids catalog latency and tens of GiB of duplicate
    // hashing on the normal repeat-serve path.
    let initial_local = match initial_local {
        Some((mut candidate, authority, local_lock)) if candidate.last_used_at_secs > 0 => {
            let verified_projector = if prepare_projector {
                match verify_candidate_projector_with_progress(&candidate, progress) {
                    Ok(projector) => projector,
                    Err(error) => {
                        warnings.push(format!(
                            "local mmproj verification failed before catalog planning: {error}"
                        ));
                        None
                    }
                }
            } else {
                None
            };
            let needs_hosted_projector_plan = prepare_projector && verified_projector.is_none();
            if needs_hosted_projector_plan {
                drop(local_lock);
                Some((candidate, authority))
            } else {
                let (prepared, suppress_projector) =
                    prepare_selected_local_decision(candidate, explicit_output, &mut warnings)?;
                candidate = prepared;
                let mmproj = if prepare_projector && !suppress_projector {
                    if verified_projector.as_ref().is_some_and(|verified| {
                        candidate
                            .projector
                            .as_ref()
                            .is_some_and(|(path, _, _)| path == verified)
                    }) {
                        verified_projector
                    } else {
                        verify_candidate_projector_with_progress(&candidate, progress)?
                    }
                } else {
                    None
                };
                report_local_ready(&candidate, progress);
                drop(local_lock);
                let authority = explicit_output.is_none().then_some(authority);
                return candidate.into_resolved(mmproj, warnings, authority);
            }
        }
        Some((candidate, authority, lock)) => {
            drop(lock);
            Some((candidate, authority))
        }
        None => None,
    };

    progress(StartupEvent::HubMetadata {
        repository: spec.repository.clone(),
    });
    let reference = HfModelReference::parse(&spec.repository, None)?;
    let mut catalog = match resolve_catalog(reference) {
        Ok(catalog) => catalog,
        Err(error) => {
            let Some((mut candidate, authority)) = initial_local else {
                return Err(error).with_context(|| {
                    format!("resolve hosted GGUF metadata for {}", spec.repository)
                });
            };
            let _lock = cache.lock_quant(&spec.repository, candidate.quant)?;
            if !reverify_candidate_after_catalog(&candidate, authority.identity(), &mut warnings) {
                return Err(error).context("local fallback changed during repository resolution");
            }
            let (prepared, suppress_projector) =
                prepare_selected_local_decision(candidate, explicit_output, &mut warnings)?;
            candidate = prepared;
            let mmproj = if prepare_projector
                && !suppress_projector
                && retained_text_requires_projector(&authority)?
            {
                match verify_candidate_projector(&candidate) {
                    Ok(Some(path)) => Some(path),
                    Ok(None) => {
                        warnings.push(format!(
                            "multimodal projector metadata unavailable; serving text-only: {error}"
                        ));
                        None
                    }
                    Err(projector_error) => {
                        warnings.push(format!(
                            "local mmproj verification failed; serving text-only: {projector_error}"
                        ));
                        None
                    }
                }
            } else {
                None
            };
            report_local_ready(&candidate, progress);
            let authority = explicit_output.is_none().then_some(authority);
            return candidate.into_resolved(mmproj, warnings, authority);
        }
    };
    let selectable = catalog
        .artifacts
        .iter()
        .filter(|artifact| {
            artifact.selectable
                && artifact.role == "text_model"
                && spec.matches_hosted_filename(&artifact.filename)
        })
        .cloned()
        .collect::<Vec<_>>();
    let selectable = selectable
        .into_iter()
        .filter(|artifact| {
            if spec.requested_selector().is_some()
                || automatic_artifact_admissible(
                    artifact.bytes,
                    hardware.available_memory_bytes,
                    pool_budget_bytes,
                )
            {
                true
            } else {
                warnings.push(format!(
                    "ignored hosted {} ({} bytes): current automatic admission budget is {} bytes",
                    artifact.filename, artifact.bytes, hardware.available_memory_bytes
                ));
                false
            }
        })
        .collect::<Vec<_>>();

    // Discover runtime quant from bounded metadata when the publisher label
    // supplies no canonical hint. Preserve that exact artifact, not an alias.
    let mut probed = std::collections::BTreeMap::new();
    let mut admitted_hints = Vec::new();
    for mut artifact in selectable {
        if artifact.quant_hint.is_none() {
            match validate_hub_gguf_header_compatibility(&artifact) {
                Ok(compatibility) => {
                    artifact.quant_hint = Some(compatibility.quant.as_str().to_owned());
                    probed.insert(artifact.filename.clone(), compatibility);
                }
                Err(DownloadError::IncompatibleHostedGguf { reason }) => {
                    warnings.push(format!(
                        "ignored incompatible hosted {}: {reason}",
                        artifact.filename
                    ));
                    continue;
                }
                Err(error) => return Err(error.into()),
            }
        }
        admitted_hints.push(artifact);
    }
    let selectable = admitted_hints;

    let excluded_identity = initial_local
        .as_ref()
        .map(|(_, authority)| authority.identity());
    let excluded = excluded_identity
        .as_ref()
        .map(std::slice::from_ref)
        .unwrap_or(&[]);
    let manual = find_best_matching_loose_with_progress(
        &selectable,
        spec.quant,
        model_dirs,
        excluded,
        &mut warnings,
        progress,
    )?;
    let cached = find_best_matching_cached_hub_with_progress(
        &selectable,
        spec.quant,
        &mut warnings,
        progress,
    )?;
    let materialized =
        |candidate: &local::ExactHostedLocal| system_time_secs(candidate.materialized);
    let (mut loose, loose_origin) = match (manual, cached) {
        (Some(manual), Some(cached)) if materialized(&cached) > materialized(&manual) => {
            (Some(cached), "hf_hub_cache_structural")
        }
        (Some(manual), Some(_)) => (Some(manual), "manual_structural"),
        (Some(manual), None) => (Some(manual), "manual_structural"),
        (None, Some(cached)) => (Some(cached), "hf_hub_cache_structural"),
        (None, None) => (None, "manual_adoption"),
    };
    if let Some((candidate, authority)) = initial_local {
        let loose_recency = loose
            .as_ref()
            .map(|candidate| (false, 0, materialized(candidate)))
            .unwrap_or((false, 0, 0));
        if loose.is_none() || bound_candidate_is_at_least_as_recent(&candidate, loose_recency.2) {
            let _lock = cache.lock_quant(&spec.repository, candidate.quant)?;
            if reverify_candidate_after_catalog(&candidate, authority.identity(), &mut warnings) {
                let (candidate, mmproj) = prepare_local_candidate_with_catalog(
                    candidate,
                    &authority,
                    explicit_output,
                    model_dirs,
                    &catalog,
                    prepare_projector,
                    &mut warnings,
                    progress,
                )?;
                report_local_ready(&candidate, progress);
                let authority = explicit_output.is_none().then_some(authority);
                return candidate.into_resolved(mmproj, warnings, authority);
            }
        }
    }
    // Setup/live recommendation is a fallback, not a reason to ignore a
    // newer compatible quant already present in the canonical Hub cache.
    let recommended = loose
        .as_ref()
        .and_then(|candidate| candidate.artifact.quant_hint.as_deref())
        .and_then(|value| QuantType::from_canonical_str(value).ok())
        .map_or_else(
            || repository_recommended_quant(spec.quant, configured_quant, hardware),
            Ok,
        )?;
    // Structurally admitted loose bytes already match one unique catalog row
    // by quant, byte length, and (when needed) basename. Do not make an
    // automatic recommendation displace compatible bytes the operator owns.
    let mut selected_requires_projector = false;
    let selected = if loose.is_none() {
        select_compatible_hosted(
            &selectable,
            spec.quant,
            recommended,
            spec.is_hosted_only()
                .then(|| spec.requested_selector().unwrap()),
            |artifact| {
                let compatibility = match probed.get(&artifact.filename) {
                    Some(known) => known.clone(),
                    None => validate_hub_gguf_header_compatibility(artifact)?,
                };
                selected_requires_projector = compatibility.requires_projector;
                Ok(())
            },
            &mut warnings,
        )?
    } else {
        None
    };
    if spec.is_hosted_only() && loose.is_none() && selected.is_none() {
        let available = catalog
            .artifacts
            .iter()
            .map(|artifact| artifact.filename.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        bail!("no compatible hosted artifact matches selector {:?}; available exact filenames: {available}. {}",
            spec.requested_selector().unwrap_or(""), warnings.join("; "));
    }
    let (native_fallback_quant, native_product_bytes) = if loose.is_none() && selected.is_none() {
        let source_reference =
            HfModelReference::parse(&catalog.repository, Some(catalog.revision.as_str()))?;
        let prepared = crate::input::hf_download::prepare_native_planning_source(source_reference)
            .with_context(|| {
                format!(
                    "prepare exact native source plan for {}@{}",
                    catalog.repository, catalog.revision
                )
            })?;
        let source_plan = prepared.source_plan();
        if source_plan.repository != catalog.repository
            || !source_plan.revision.eq_ignore_ascii_case(&catalog.revision)
        {
            bail!("native source plan changed the hosted catalog repository/revision identity");
        }
        catalog.source_weight_bytes = Some(source_plan.total_weight_bytes);
        catalog.source_uncached_weight_bytes = Some(source_plan.uncached_weight_bytes);
        let (quant, bytes) = select_native_quant_from_exact_plans(
            spec.quant,
            recommended,
            hardware.available_memory_bytes,
            pool_budget_bytes,
            |quant| plan_native_quant_products(&prepared, quant),
            &mut warnings,
        )?;
        (quant, Some(bytes))
    } else {
        (spec.quant.unwrap_or(recommended), None)
    };
    let target_quant = loose
        .as_ref()
        .and_then(|candidate| candidate.artifact.quant_hint.as_deref())
        .or_else(|| {
            selected
                .as_ref()
                .and_then(|artifact| artifact.quant_hint.as_deref())
        })
        .and_then(|value| QuantType::from_canonical_str(value).ok())
        .unwrap_or(native_fallback_quant);
    let _resolution_lock = cache
        .lock_quant(&spec.repository, target_quant)
        .with_context(|| {
            format!(
                "lock managed resolution for {}:{}",
                spec.repository, target_quant
            )
        })?;
    if let Some((candidate, authority, _local_lock)) = select_local_with_progress(
        spec,
        model_dirs,
        cache,
        Some(target_quant),
        hardware.available_memory_bytes,
        pool_budget_bytes,
        &mut warnings,
        progress,
    )? {
        let selected_loose_materialized_at =
            loose.as_ref().map(|candidate| materialized(candidate));
        if post_lock_local_candidate_wins(&candidate, selected_loose_materialized_at) {
            let (candidate, mmproj) = prepare_local_candidate_with_catalog(
                candidate,
                &authority,
                explicit_output,
                model_dirs,
                &catalog,
                prepare_projector,
                &mut warnings,
                progress,
            )?;
            report_local_ready(&candidate, progress);
            let authority = explicit_output.is_none().then_some(authority);
            return candidate.into_resolved(mmproj, warnings, authority);
        }
    }
    if loose.is_some() && explicit_output.is_none() {
        let loose = loose
            .take()
            .context("local GGUF selection disappeared before activation")?;
        if !loose.retained.is_stable()? {
            bail!("local GGUF changed after bounded metadata admission");
        }
        let quant = loose
            .artifact
            .quant_hint
            .as_deref()
            .and_then(|value| QuantType::from_canonical_str(value).ok())
            .context("local GGUF has no supported quant identity")?;
        let projector_required = prepare_projector
            && hosted_pair_requires_projector(catalog.requires_projector, loose.requires_projector);
        let mut candidate = Candidate {
            hub_filename: Some(loose.artifact.filename.clone()),
            repository: loose.artifact.repository,
            revision: loose.artifact.revision,
            root: loose
                .path
                .parent()
                .context("manual local GGUF has no parent directory")?
                .to_path_buf(),
            path: loose.path,
            bytes: loose.artifact.bytes,
            sha256: loose.artifact.sha256,
            quant,
            origin: loose_origin.into(),
            materialized_at_secs: system_time_secs(loose.materialized),
            last_used_at_secs: 0,
            projector: None,
            sidecar: None,
            receipt_target_identity: None,
        };
        let mut mmproj = if projector_required && loose_origin == "hf_hub_cache_structural" {
            match prepare_cached_projector_in_place(
                &mut candidate,
                &loose.retained,
                &catalog,
                progress,
            ) {
                Ok(Some(path)) => Some(path),
                Ok(None) => {
                    warnings.push(
                        "multimodal text model has no unambiguous exact-revision mmproj; serving text-only"
                            .into(),
                    );
                    None
                }
                Err(error) => {
                    warnings.push(format!(
                        "automatic exact-revision mmproj preparation failed; serving text-only: {error}"
                    ));
                    None
                }
            }
        } else if projector_required {
            best_effort_manual_projector_with_catalog(
                &mut candidate,
                &loose.retained,
                model_dirs,
                &catalog,
                &mut warnings,
                progress,
            )
        } else {
            None
        };
        let projector_binding = mmproj.as_ref().and_then(|path| {
            candidate
                .projector
                .as_ref()
                .filter(|(bound, _, _)| bound == path)
                .map(|(path, bytes, sha256)| (path.clone(), *bytes, sha256.clone()))
        });
        let (mmproj_sha256, mmproj_activation_authority) = match projector_binding {
            Some((path, bytes, sha256)) => {
                run_after_automatic_projector_prepared(&path);
                match retain_verified_projector_authority(&path, bytes, &sha256) {
                    Ok(Some(authority)) => (Some(sha256), Some(authority)),
                    Ok(None) => {
                        warnings.push(
                            "automatic mmproj changed or no longer matches its digest before retained activation; serving text-only"
                                .into(),
                        );
                        mmproj = None;
                        (None, None)
                    }
                    Err(error) => {
                        warnings.push(format!(
                            "automatic mmproj retention failed; serving text-only: {error}"
                        ));
                        mmproj = None;
                        (None, None)
                    }
                }
            }
            None if mmproj.is_some() => {
                warnings.push(
                    "automatic mmproj has no retained digest binding; serving text-only".into(),
                );
                mmproj = None;
                (None, None)
            }
            None => (None, None),
        };
        report_local_ready(&candidate, progress);
        return Ok(ResolvedManagedModel {
            pool_identity: candidate.pool_identity(),
            gguf_path: candidate.path,
            mmproj_path: mmproj,
            repository: candidate.repository,
            revision: candidate.revision,
            quant: candidate.quant,
            origin: candidate.origin,
            warnings,
            track_success_history: false,
            activation_authority: Some(loose.retained),
            mmproj_sha256,
            mmproj_activation_authority,
        });
    }
    // Publication is different from serving in place: before copying bytes
    // to an explicit destination, bind their complete immutable Hub digest.
    // This is intentionally the only local-discovery branch that performs a
    // model-sized hash.
    if explicit_output.is_some() {
        if let Some(loose) = loose.as_mut() {
            let actual = hash_hosted_local_candidate(
                &loose.path,
                loose.artifact.bytes,
                loose
                    .artifact
                    .quant_hint
                    .as_deref()
                    .and_then(|value| QuantType::from_canonical_str(value).ok()),
                StartupOrigin::from_internal(loose_origin),
                &mut loose.retained,
                progress,
            )?;
            if !actual.eq_ignore_ascii_case(&loose.artifact.sha256) {
                bail!(
                    "the structurally compatible local GGUF does not match the immutable hosted payload required for explicit --output publication"
                );
            }
        }
    }
    let mut suppress_automatic_projector = false;
    let mut prepared_projector = None;
    let (mut candidate, prepared_here) = if let Some(loose) = loose {
        let artifact = &loose.artifact;
        let destination = hosted_destination(artifact, explicit_output)?;
        let text_plan = PreparedLocalArtifact::prepare_retained(
            loose.retained,
            &destination,
            artifact.bytes,
            &artifact.sha256,
        )?;
        let text_destination_exact = !text_plan.needs_copy();
        let projector_required = prepare_projector
            && hosted_pair_requires_projector(catalog.requires_projector, loose.requires_projector);
        let projector_plan =
            planned_hosted_projector(artifact, &destination, &catalog, projector_required)
                .and_then(|plan| {
                    plan.map(|(projector, projector_destination)| {
                        prepare_projector_action(projector, projector_destination, model_dirs)
                    })
                    .transpose()
                });
        let projector_plan = match projector_plan {
            Ok(Some(plan)) => {
                let pair_preflight = match &plan.source {
                    PreparedProjectorSource::Existing(_) | PreparedProjectorSource::HubCache(_) => {
                        check_local_artifact_pair_plan_with_authorities(
                            &artifact.repository,
                            text_plan.source_device_id(),
                            text_plan.destination(),
                            text_plan.destination_device_id(),
                            text_plan.destination_available_bytes(),
                            artifact.bytes,
                            text_destination_exact,
                            None,
                        )
                    }
                    PreparedProjectorSource::Local(_) => {
                        check_local_artifact_pair_plan_with_authorities(
                            &artifact.repository,
                            text_plan.source_device_id(),
                            text_plan.destination(),
                            text_plan.destination_device_id(),
                            text_plan.destination_available_bytes(),
                            artifact.bytes,
                            text_destination_exact,
                            Some((
                                plan.source_device_id(),
                                &plan.destination,
                                plan.destination_device_id(),
                                plan.destination_available_bytes(),
                                plan.artifact.bytes,
                                plan.destination_is_exact(),
                            )),
                        )
                    }
                    PreparedProjectorSource::Hosted => {
                        check_local_text_hosted_projector_plan_with_authorities(
                            &artifact.repository,
                            text_plan.source_device_id(),
                            text_plan.destination(),
                            text_plan.destination_device_id(),
                            text_plan.destination_available_bytes(),
                            artifact.bytes,
                            text_destination_exact,
                            &plan.artifact,
                            &plan.destination,
                            plan.destination_device_id(),
                            plan.destination_available_bytes(),
                            plan.destination_is_exact(),
                        )
                    }
                };
                if let Err(error) = pair_preflight {
                    warnings.push(format!(
                        "automatic text/mmproj pair preflight failed; serving text-only: {error}"
                    ));
                    check_local_artifact_pair_plan_with_authorities(
                        &artifact.repository,
                        text_plan.source_device_id(),
                        text_plan.destination(),
                        text_plan.destination_device_id(),
                        text_plan.destination_available_bytes(),
                        artifact.bytes,
                        text_destination_exact,
                        None,
                    )?;
                    suppress_automatic_projector = true;
                    None
                } else {
                    Some(plan)
                }
            }
            Ok(None) if projector_required => {
                warnings.push(
                    "multimodal text model has no unambiguous matching hosted mmproj; serving text-only"
                        .into(),
                );
                check_local_artifact_pair_plan_with_authorities(
                    &artifact.repository,
                    text_plan.source_device_id(),
                    text_plan.destination(),
                    text_plan.destination_device_id(),
                    text_plan.destination_available_bytes(),
                    artifact.bytes,
                    text_destination_exact,
                    None,
                )?;
                suppress_automatic_projector = true;
                None
            }
            Ok(None) => {
                check_local_artifact_pair_plan_with_authorities(
                    &artifact.repository,
                    text_plan.source_device_id(),
                    text_plan.destination(),
                    text_plan.destination_device_id(),
                    text_plan.destination_available_bytes(),
                    artifact.bytes,
                    text_destination_exact,
                    None,
                )?;
                None
            }
            Err(error) => {
                warnings.push(format!(
                    "automatic mmproj planning failed; serving text-only: {error}"
                ));
                check_local_artifact_pair_plan_with_authorities(
                    &artifact.repository,
                    text_plan.source_device_id(),
                    text_plan.destination(),
                    text_plan.destination_device_id(),
                    text_plan.destination_available_bytes(),
                    artifact.bytes,
                    text_destination_exact,
                    None,
                )?;
                suppress_automatic_projector = true;
                None
            }
        };
        let projector_current = match projector_plan.as_ref() {
            Some(plan) => plan.is_current()?,
            None => true,
        };
        if !text_plan.is_current()? || !projector_current {
            bail!("local text/projector authority changed after disk preflight");
        }
        text_plan.materialize(&artifact.repository, artifact.bytes, &artifact.sha256)?;
        let candidate = bind_hosted_destination(
            &destination,
            artifact,
            if text_destination_exact {
                "existing_destination"
            } else {
                loose_origin
            },
        )?;
        prepared_projector = projector_plan;
        (candidate, false)
    } else if let Some(artifact) = selected {
        let destination = hosted_destination(&artifact, explicit_output)?;
        let text_destination_exact =
            verify_or_refuse_existing_hosted_destination(&destination, &artifact)?;
        let projector_required = prepare_projector
            && hosted_pair_requires_projector(
                catalog.requires_projector,
                selected_requires_projector,
            );
        let projector_plan =
            planned_hosted_projector(&artifact, &destination, &catalog, projector_required)
                .and_then(|plan| {
                    plan.map(|(projector, projector_destination)| {
                        prepare_projector_action(projector, projector_destination, model_dirs)
                    })
                    .transpose()
                });
        let projector_plan = match projector_plan {
            Ok(Some(plan)) => {
                let pair_preflight = match &plan.source {
                    PreparedProjectorSource::Existing(_) => {
                        check_hub_artifact_pair_plan_from_state(
                            &artifact,
                            &destination,
                            text_destination_exact,
                            None,
                        )
                    }
                    PreparedProjectorSource::HubCache(_) => {
                        check_hub_artifact_pair_plan_from_state(
                            &artifact,
                            &destination,
                            text_destination_exact,
                            Some((&plan.artifact, &plan.destination, true)),
                        )
                    }
                    PreparedProjectorSource::Local(source) => {
                        check_hosted_text_local_projector_plan_with_device(
                            &artifact,
                            &destination,
                            text_destination_exact,
                            &source.path,
                            Some(source.retained.device_id()),
                            &plan.destination,
                            plan.artifact.bytes,
                            false,
                        )
                    }
                    PreparedProjectorSource::Hosted => check_hub_artifact_pair_plan_from_state(
                        &artifact,
                        &destination,
                        text_destination_exact,
                        Some((&plan.artifact, &plan.destination, false)),
                    ),
                };
                if let Err(error) = pair_preflight {
                    warnings.push(format!(
                        "automatic text/mmproj pair preflight failed; serving text-only: {error}"
                    ));
                    if !text_destination_exact {
                        check_hub_artifact_plan(&artifact, &destination)?;
                    }
                    suppress_automatic_projector = true;
                    None
                } else {
                    Some(plan)
                }
            }
            Ok(None) if projector_required => {
                warnings.push(
                    "multimodal text model has no unambiguous matching hosted mmproj; serving text-only"
                        .into(),
                );
                if !text_destination_exact {
                    check_hub_artifact_plan(&artifact, &destination)?;
                }
                suppress_automatic_projector = true;
                None
            }
            Ok(None) => {
                if !text_destination_exact {
                    check_hub_artifact_plan(&artifact, &destination)?;
                }
                None
            }
            Err(error) => {
                warnings.push(format!(
                    "automatic mmproj planning failed; serving text-only: {error}"
                ));
                if !text_destination_exact {
                    check_hub_artifact_plan(&artifact, &destination)?;
                }
                suppress_automatic_projector = true;
                None
            }
        };
        let candidate = if text_destination_exact {
            bind_hosted_destination(&destination, &artifact, "existing_destination")?
        } else {
            progress(StartupEvent::HostedDownload {
                filename: display_filename(Path::new(&artifact.filename)),
                bytes: artifact.bytes,
            });
            let progress_filename = display_filename(Path::new(&artifact.filename));
            let cached = download_hub_gguf_with_progress(&artifact, &mut |update| {
                progress(StartupEvent::HostedDownloadProgress {
                    filename: progress_filename.clone(),
                    completed_bytes: update.completed_bytes,
                    total_bytes: update.total_bytes,
                    bytes_per_second: update.bytes_per_second,
                    elapsed_ms: update.elapsed_ms,
                });
            })?;
            materialize_hosted(cached, &artifact, explicit_output, "hosted_download")?
        };
        prepared_projector = projector_plan;
        (candidate, !text_destination_exact)
    } else {
        native_convert_with_progress(
            &catalog,
            native_fallback_quant,
            explicit_output,
            native_product_bytes,
            progress,
        )?
    };
    let (prepared, local_suppress_projector) =
        prepare_selected_local_decision(candidate, None, &mut warnings)?;
    candidate = prepared;
    suppress_automatic_projector |= local_suppress_projector;
    let mmproj = if let Some(plan) = prepared_projector {
        match materialize_prepared_projector(plan, &mut candidate, &mut warnings, progress) {
            Ok(path) => Some(path),
            Err(error) => {
                warnings.push(format!(
                    "automatic mmproj preparation failed; serving text-only: {error}"
                ));
                None
            }
        }
    } else {
        (prepare_projector && !suppress_automatic_projector)
            .then(|| {
                best_effort_projector_with_catalog_with_progress(
                    &mut candidate,
                    model_dirs,
                    &catalog,
                    hosted_pair_requires_projector(
                        catalog.requires_projector,
                        selected_requires_projector,
                    ),
                    &mut warnings,
                    progress,
                )
            })
            .flatten()
    };
    if prepared_here {
        report_model_prepared(&candidate, progress);
    } else {
        report_local_ready(&candidate, progress);
    }
    candidate.into_resolved(mmproj, warnings, None)
}

fn plan_native_quant_products(
    prepared: &crate::input::hf_download::PreparedNativePlanningSource,
    quant: QuantType,
) -> Result<u64> {
    let ftype = crate::quantize::ggml_quants::GgufFtype::try_from(quant.gguf_file_type())
        .map_err(|_| anyhow!("unsupported native quant plan for {quant}"))?;
    let source_plan = prepared.source_plan();
    let reference =
        HfModelReference::parse(&source_plan.repository, Some(source_plan.revision.as_str()))?
            .resolve(&source_plan.revision)?;
    let text = crate::convert::cli_driver::plan_standard_text_output_bytes(
        prepared.path(),
        ftype,
        reference,
        prepared.source_bundle_sha256().to_owned(),
        source_plan.requires_projector,
    )?;
    let projector = if source_plan.requires_projector {
        crate::models::vit::planned_vision_tower_output_bytes(
            prepared.path(),
            Some(prepared.source_bundle_sha256()),
            Some("00000000-0000-0000-0000-000000000000"),
        )?
    } else {
        0
    };
    text.checked_add(projector)
        .context("native text plus projector product size overflowed u64")
}

pub(super) fn select_native_quant_from_exact_plans(
    exact: Option<QuantType>,
    recommended: QuantType,
    available_memory_bytes: u64,
    pool_budget_bytes: u64,
    mut plan: impl FnMut(QuantType) -> Result<u64>,
    warnings: &mut Vec<String>,
) -> Result<(QuantType, u64)> {
    let tiers = exact.map_or_else(
        || {
            quality_descending()
                .into_iter()
                .filter(|quant| quant_quality(*quant) <= quant_quality(recommended))
                .collect::<Vec<_>>()
        },
        |exact| vec![exact],
    );
    for quant in tiers {
        let bytes = plan(quant)?;
        if exact.is_some()
            || automatic_artifact_admissible(bytes, available_memory_bytes, pool_budget_bytes)
        {
            return Ok((quant, bytes));
        }
        warnings.push(format!(
            "native {quant} plan is {bytes} bytes and does not fit the automatic runtime budget; trying the next smaller quant"
        ));
    }
    bail!(
        "no supported native quant fits the current automatic runtime budget; request an exact repo:QUANT only after confirming it fits"
    )
}

#[cfg(test)]
pub(super) fn admit_automatic_projector_preflight(
    projector_plan: Result<Option<(HubGgufArtifact, PathBuf)>>,
    pair_preflight: impl FnOnce(Option<&(HubGgufArtifact, PathBuf)>) -> Result<()>,
    text_preflight: impl FnOnce() -> Result<()>,
    warnings: &mut Vec<String>,
) -> Result<bool> {
    let projector_plan = match projector_plan {
        Ok(plan) => plan,
        Err(error) => {
            warnings.push(format!(
                "automatic mmproj planning failed; serving text-only: {error}"
            ));
            text_preflight()?;
            return Ok(true);
        }
    };
    if let Err(error) = pair_preflight(projector_plan.as_ref()) {
        warnings.push(format!(
            "automatic text/mmproj pair preflight failed; serving text-only: {error}"
        ));
        text_preflight()?;
        return Ok(true);
    }
    Ok(false)
}

fn planned_hosted_projector(
    text: &HubGgufArtifact,
    text_destination: &Path,
    catalog: &HubGgufCatalog,
    required: bool,
) -> Result<Option<(HubGgufArtifact, PathBuf)>> {
    if !required {
        return Ok(None);
    }
    let text_name = safe_basename(&text.filename)?;
    let quant = text
        .quant_hint
        .as_deref()
        .and_then(|value| QuantType::from_canonical_str(value).ok())
        .context("selected hosted text artifact has no supported quant identity")?;
    let candidate = Candidate {
        hub_filename: Some(text.filename.clone()),
        repository: text.repository.clone(),
        revision: text.revision.clone(),
        path: PathBuf::from(text_name),
        root: PathBuf::from("."),
        bytes: text.bytes,
        sha256: text.sha256.clone(),
        quant,
        origin: "hosted_plan".into(),
        materialized_at_secs: 0,
        last_used_at_secs: 0,
        projector: None,
        sidecar: None,
        receipt_target_identity: None,
    };
    let companions = catalog
        .artifacts
        .iter()
        .filter(|artifact| artifact.role == "companion")
        .collect::<Vec<_>>();
    let Some(projector) = select_projector_companion(&candidate, companions, None)? else {
        return Ok(None);
    };
    let destination = text_destination
        .parent()
        .context("hosted text destination has no parent")?
        .join(safe_basename(&projector.filename)?);
    Ok(Some((projector, destination)))
}

fn planned_local_projector(
    candidate: &Candidate,
    text_authority: &crate::core::bounded_file::StableRegularFile,
    text_destination: &Path,
    catalog: &HubGgufCatalog,
    required: bool,
) -> Result<Option<(HubGgufArtifact, PathBuf)>> {
    if !required {
        return Ok(None);
    }
    let expected = retained_expected_projector_sha256(text_authority)?;
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
    let Some(projector) = select_projector_companion(candidate, companions, expected.as_deref())?
    else {
        return Ok(None);
    };
    let destination = text_destination
        .parent()
        .context("local text destination has no parent")?
        .join(safe_basename(&projector.filename)?);
    Ok(Some((projector, destination)))
}

fn prepare_local_candidate_with_catalog(
    candidate: Candidate,
    text_authority: &crate::core::bounded_file::StableRegularFile,
    explicit_output: Option<&Path>,
    model_dirs: &[PathBuf],
    catalog: &HubGgufCatalog,
    prepare_projector: bool,
    warnings: &mut Vec<String>,
    progress: &mut StartupProgress<'_>,
) -> Result<(Candidate, Option<PathBuf>)> {
    prepare_local_candidate_with_catalog_resolver(
        candidate,
        text_authority,
        explicit_output,
        model_dirs,
        catalog,
        prepare_projector,
        warnings,
        progress,
        resolve_hub_gguf_catalog,
    )
}

pub(super) fn prepare_local_candidate_with_catalog_resolver(
    mut candidate: Candidate,
    text_authority: &crate::core::bounded_file::StableRegularFile,
    explicit_output: Option<&Path>,
    model_dirs: &[PathBuf],
    catalog: &HubGgufCatalog,
    prepare_projector: bool,
    warnings: &mut Vec<String>,
    progress: &mut StartupProgress<'_>,
    mut resolve_catalog: impl FnMut(HfModelReference) -> Result<HubGgufCatalog, DownloadError>,
) -> Result<(Candidate, Option<PathBuf>)> {
    if !prepare_projector {
        let (candidate, _) = prepare_selected_local_decision(candidate, explicit_output, warnings)?;
        return Ok((candidate, None));
    }
    match verify_candidate_projector(&candidate) {
        Ok(Some(_)) => {
            let (candidate, suppress) =
                prepare_selected_local_decision(candidate, explicit_output, warnings)?;
            let projector = if suppress {
                None
            } else {
                verify_candidate_projector(&candidate)?
            };
            return Ok((candidate, projector));
        }
        Ok(None) => candidate.projector = None,
        Err(error) => {
            warnings.push(format!(
                "local mmproj verification failed; planning an exact hosted replacement: {error}"
            ));
            candidate.projector = None;
        }
    }

    let exact_catalog;
    let catalog = if let Some(reference) =
        exact_local_projector_catalog_reference(&candidate, catalog)?
    {
        exact_catalog = match resolve_catalog(reference) {
            Ok(catalog)
                if catalog.repository == candidate.repository
                    && catalog.revision.eq_ignore_ascii_case(&candidate.revision) =>
            {
                catalog
            }
            Ok(_) => bail!("exact local projector catalog changed repository/revision identity"),
            Err(error) => {
                warnings.push(format!(
                    "exact-revision projector metadata unavailable for {}@{}; serving text-only: {error}",
                    candidate.repository, candidate.revision
                ));
                let (candidate, _) =
                    prepare_selected_local_decision(candidate, explicit_output, warnings)?;
                return Ok((candidate, None));
            }
        };
        &exact_catalog
    } else {
        catalog
    };
    let text_projector_required = retained_text_requires_projector(text_authority)?;
    let projector_required =
        hosted_pair_requires_projector(catalog.requires_projector, text_projector_required);
    if !projector_required {
        let (candidate, _) = prepare_selected_local_decision(candidate, explicit_output, warnings)?;
        return Ok((candidate, None));
    }

    let default = managed_revision_dir(
        &managed_model_root()?,
        &candidate.repository,
        &candidate.revision,
    )?
    .join(
        candidate
            .path
            .file_name()
            .context("selected local artifact has no filename")?,
    );
    let text_destination = resolve_output_path(explicit_output, default)?;
    let text_plan = PreparedLocalArtifact::prepare(
        &candidate.path,
        &text_destination,
        candidate.bytes,
        &candidate.sha256,
    )?;
    let text_destination_exact = !text_plan.needs_copy();
    let projector_plan = planned_local_projector(
        &candidate,
        text_authority,
        &text_destination,
        catalog,
        projector_required,
    )
    .and_then(|plan| {
        plan.map(|(artifact, destination)| {
            prepare_projector_action(artifact, destination, model_dirs)
        })
        .transpose()
    });
    let projector_plan = match projector_plan {
        Ok(Some(plan)) => {
            progress(StartupEvent::ProjectorPrepare {
                filename: display_filename(Path::new(&plan.artifact.filename)),
                bytes: plan.artifact.bytes,
            });
            let pair_preflight = match &plan.source {
                PreparedProjectorSource::Existing(_) | PreparedProjectorSource::HubCache(_) => {
                    check_local_artifact_pair_plan_with_authorities(
                        &candidate.repository,
                        text_plan.source_device_id(),
                        text_plan.destination(),
                        text_plan.destination_device_id(),
                        text_plan.destination_available_bytes(),
                        candidate.bytes,
                        text_destination_exact,
                        None,
                    )
                }
                PreparedProjectorSource::Local(_) => {
                    check_local_artifact_pair_plan_with_authorities(
                        &candidate.repository,
                        text_plan.source_device_id(),
                        text_plan.destination(),
                        text_plan.destination_device_id(),
                        text_plan.destination_available_bytes(),
                        candidate.bytes,
                        text_destination_exact,
                        Some((
                            plan.source_device_id(),
                            &plan.destination,
                            plan.destination_device_id(),
                            plan.destination_available_bytes(),
                            plan.artifact.bytes,
                            plan.destination_is_exact(),
                        )),
                    )
                }
                PreparedProjectorSource::Hosted => {
                    check_local_text_hosted_projector_plan_with_authorities(
                        &candidate.repository,
                        text_plan.source_device_id(),
                        text_plan.destination(),
                        text_plan.destination_device_id(),
                        text_plan.destination_available_bytes(),
                        candidate.bytes,
                        text_destination_exact,
                        &plan.artifact,
                        &plan.destination,
                        plan.destination_device_id(),
                        plan.destination_available_bytes(),
                        plan.destination_is_exact(),
                    )
                }
            };
            if let Err(error) = pair_preflight {
                warnings.push(format!(
                    "automatic text/mmproj pair preflight failed; serving text-only: {error}"
                ));
                check_local_artifact_pair_plan_with_authorities(
                    &candidate.repository,
                    text_plan.source_device_id(),
                    text_plan.destination(),
                    text_plan.destination_device_id(),
                    text_plan.destination_available_bytes(),
                    candidate.bytes,
                    text_destination_exact,
                    None,
                )?;
                None
            } else {
                Some(plan)
            }
        }
        Ok(None) => {
            warnings.push(
                "multimodal text model has no unambiguous matching hosted mmproj; serving text-only"
                    .into(),
            );
            None
        }
        Err(error) => {
            warnings.push(format!(
                "automatic mmproj planning failed; serving text-only: {error}"
            ));
            None
        }
    };
    let projector_current = match projector_plan.as_ref() {
        Some(plan) => plan.is_current()?,
        None => true,
    };
    if !text_plan.is_current()? || !projector_current {
        bail!("local text/projector authority changed after disk preflight");
    }
    text_plan.materialize(&candidate.repository, candidate.bytes, &candidate.sha256)?;
    if text_destination != candidate.path {
        candidate.path = text_destination.clone();
        candidate.root = text_destination
            .parent()
            .context("selected destination has no parent")?
            .to_path_buf();
        candidate.materialized_at_secs = now_secs();
        candidate.origin = "local_adoption".to_owned();
        candidate.sidecar = None;
        candidate.receipt_target_identity = None;
    }
    let projector = match projector_plan {
        Some(plan) => {
            match materialize_prepared_projector(plan, &mut candidate, warnings, progress) {
                Ok(path) => Some(path),
                Err(error) => {
                    warnings.push(format!(
                        "automatic mmproj preparation failed; serving text-only: {error}"
                    ));
                    None
                }
            }
        }
        None => None,
    };
    let (candidate, _) = prepare_selected_local_decision(candidate, None, warnings)?;
    Ok((candidate, projector))
}

pub(super) fn exact_local_projector_catalog_reference(
    candidate: &Candidate,
    catalog: &HubGgufCatalog,
) -> Result<Option<HfModelReference>> {
    if catalog.repository == candidate.repository
        && catalog.revision.eq_ignore_ascii_case(&candidate.revision)
    {
        return Ok(None);
    }
    Ok(Some(HfModelReference::parse(
        &candidate.repository,
        Some(candidate.revision.as_str()),
    )?))
}

pub(super) const fn hosted_pair_requires_projector(
    repository_config_marker: bool,
    authenticated_gguf_marker: bool,
) -> bool {
    repository_config_marker || authenticated_gguf_marker
}

pub(super) fn reverify_candidate_after_catalog(
    candidate: &Candidate,
    identity: crate::core::bounded_file::StableFileIdentity,
    warnings: &mut Vec<String>,
) -> bool {
    match crate::core::bounded_file::regular_path_matches_identity(&candidate.path, identity) {
        Ok(true) => true,
        Ok(false) => {
            warnings.push(
                "local candidate changed during repository resolution; continuing with a hosted/native fallback"
                    .to_owned(),
            );
            false
        }
        Err(error) => {
            warnings.push(format!(
                "local candidate changed during repository resolution; continuing with a hosted/native fallback: {error}"
            ));
            false
        }
    }
}

pub(super) fn bound_candidate_is_at_least_as_recent(
    candidate: &Candidate,
    loose_materialized_at_secs: u64,
) -> bool {
    candidate_recency(candidate) >= (false, 0, loose_materialized_at_secs)
}

pub(super) fn post_lock_local_candidate_wins(
    candidate: &Candidate,
    selected_loose_materialized_at_secs: Option<u64>,
) -> bool {
    selected_loose_materialized_at_secs
        .is_none_or(|loose| bound_candidate_is_at_least_as_recent(candidate, loose))
}

pub(super) fn repository_recommended_quant(
    exact: Option<QuantType>,
    configured: Option<&str>,
    hardware: &HardwareProfile,
) -> Result<QuantType> {
    match exact {
        Some(exact) => Ok(exact),
        None => match configured {
            Some(value) => QuantType::from_canonical_str(value).map_err(|_| {
                anyhow!(
                    "setup convert quant `{value}` cannot drive automatic repository serving; choose one of Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, or Q8_0 with `hf2q setup --default-quant QUANT`, or request an exact repository:QUANT"
                )
            }),
            None => select_quant(&GpuInfo::from_hardware_profile(hardware)),
        },
    }
}

#[cfg(test)]
pub(super) fn select_native_fallback_quant(
    exact: Option<QuantType>,
    recommended: QuantType,
    output_upper_bound_bytes: Option<u64>,
    available_memory_bytes: u64,
    pool_budget_bytes: u64,
) -> Result<QuantType> {
    if let Some(exact) = exact {
        return Ok(exact);
    }
    let Some(output_upper_bound_bytes) = output_upper_bound_bytes else {
        bail!(
            "cannot establish a bounded native output plan for automatic conversion; request an exact repo:QUANT to override automatic admission"
        );
    };
    if automatic_artifact_admissible(
        output_upper_bound_bytes,
        available_memory_bytes,
        pool_budget_bytes,
    ) {
        Ok(recommended)
    } else {
        bail!(
            "the conservative native output bound has insufficient runtime headroom; request an exact repo:QUANT only after confirming it fits"
        );
    }
}

/// Bound every artifact produced by one native conversion before transferring
/// source weights. A paired multimodal conversion writes a quantized text
/// model plus an F16 projector; the exact source-byte total is a conservative
/// upper bound for that projector. Projector-only conversion uses the same
/// bound without also reserving a text-model extent.
pub(crate) fn planned_native_product_bytes(
    source_weight_bytes: u64,
    planned_text_bytes: u64,
    requires_projector: bool,
    text_only: bool,
    projector_only: bool,
) -> u64 {
    if projector_only {
        source_weight_bytes
    } else if requires_projector && !text_only {
        planned_text_bytes.saturating_add(source_weight_bytes)
    } else {
        planned_text_bytes
    }
}

pub(super) fn select_compatible_hosted(
    artifacts: &[HubGgufArtifact],
    exact: Option<QuantType>,
    recommended: QuantType,
    literal_selector: Option<&str>,
    mut probe: impl FnMut(&HubGgufArtifact) -> Result<(), DownloadError>,
    warnings: &mut Vec<String>,
) -> Result<Option<HubGgufArtifact>> {
    let tiers = if literal_selector.is_some() {
        vec![None]
    } else {
        hosted_tiers(exact, recommended)
            .into_iter()
            .map(Some)
            .collect()
    };
    for tier in tiers {
        let mut compatible = Vec::new();
        for artifact in artifacts.iter().filter(|artifact| {
            tier.is_none_or(|tier| {
                artifact
                    .quant_hint
                    .as_deref()
                    .and_then(|hint| QuantType::from_canonical_str(hint).ok())
                    == Some(tier)
            })
        }) {
            match probe(artifact) {
                Ok(()) => compatible.push(artifact.clone()),
                Err(DownloadError::IncompatibleHostedGguf { reason }) => warnings.push(format!(
                    "ignored incompatible hosted {} before payload transfer: {reason}",
                    artifact.filename
                )),
                Err(error) => return Err(error.into()),
            }
        }
        if let Some(selector) = literal_selector {
            match compatible.as_slice() {
                [] => return Ok(None),
                [artifact] => return Ok(Some(artifact.clone())),
                _ => bail!("multiple compatible hosted artifacts match selector {selector:?}: {}; use an exact repository-relative filename",
                    compatible.iter().map(|artifact| artifact.filename.as_str()).collect::<Vec<_>>().join(", ")),
            }
        }
        let tier = tier.expect("quant-ranked selection has a tier");
        if let Some(selected) = select_hosted(&compatible, Some(tier), tier)? {
            return Ok(Some(selected));
        }
    }
    Ok(None)
}

fn hosted_tiers(exact: Option<QuantType>, recommended: QuantType) -> Vec<QuantType> {
    if let Some(exact) = exact {
        return vec![exact];
    }
    let quality = quality_descending();
    quality
        .into_iter()
        .filter(|quant| quant_quality(*quant) <= quant_quality(recommended))
        .chain(
            quality
                .into_iter()
                .rev()
                .filter(|quant| quant_quality(*quant) > quant_quality(recommended)),
        )
        .collect()
}

pub(super) fn select_hosted(
    artifacts: &[HubGgufArtifact],
    exact: Option<QuantType>,
    recommended: QuantType,
) -> Result<Option<HubGgufArtifact>> {
    if artifacts.is_empty() {
        return Ok(None);
    }
    let desired = exact.unwrap_or(recommended);
    let mut tiers = if exact.is_some() {
        vec![desired]
    } else {
        let quality_order = quality_descending();
        let mut tiers = quality_order
            .into_iter()
            .filter(|quant| quant_quality(*quant) <= quant_quality(desired))
            .collect::<Vec<_>>();
        // The caller has already removed artifacts that do not fit automatic
        // runtime admission. Prefer the recommendation and smaller tiers, but
        // if none exists, use the nearest admitted higher tier before an
        // expensive native conversion fallback. This keeps a bare repository
        // operand useful when the publisher offers only (for example) Q5 for
        // a configured Q4 preference.
        tiers.extend(
            quality_order
                .into_iter()
                .rev()
                .filter(|quant| quant_quality(*quant) > quant_quality(desired)),
        );
        tiers
    };
    if tiers.is_empty() {
        tiers.push(desired);
    }
    for tier in tiers {
        let matches = artifacts
            .iter()
            .filter(|artifact| {
                artifact
                    .quant_hint
                    .as_deref()
                    .and_then(|value| QuantType::from_canonical_str(value).ok())
                    == Some(tier)
            })
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [] => {}
            [artifact] => return Ok(Some((**artifact).clone())),
            _ => {
                let filenames = matches
                    .iter()
                    .map(|artifact| artifact.filename.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                bail!(
                    "hosted repository has multiple {tier} artifacts and quant alone is ambiguous: {filenames}"
                )
            }
        }
    }
    // Hosted GGUF is the startup optimization, not the semantic authority for
    // an exact request. When the requested hosted tier is absent (or all of
    // its candidates failed semantic header validation), preserve the native
    // source-conversion fallback for that exact quant.
    Ok(None)
}

pub(super) fn automatic_artifact_admissible(
    bytes: u64,
    available_memory_bytes: u64,
    pool_budget_bytes: u64,
) -> bool {
    const MIN_RUNTIME_HEADROOM: u64 = 2 * 1024 * 1024 * 1024;
    let proportional_headroom = bytes.div_ceil(8);
    let required = bytes.saturating_add(MIN_RUNTIME_HEADROOM.max(proportional_headroom));
    available_memory_bytes >= required && bytes <= pool_budget_bytes
}

fn materialize_hosted(
    source: DownloadedHubArtifact,
    artifact: &HubGgufArtifact,
    explicit_output: Option<&Path>,
    origin: &str,
) -> Result<Candidate> {
    let destination = hosted_destination(artifact, explicit_output)?;
    materialize_hub_cache_symlink(source, &destination, artifact.bytes, &artifact.sha256)?;
    bind_hosted_destination(&destination, artifact, origin)
}

fn bind_hosted_destination(
    destination: &Path,
    artifact: &HubGgufArtifact,
    origin: &str,
) -> Result<Candidate> {
    let now = now_secs();
    let binding = ManagedBinding {
        schema_version: SCHEMA_VERSION,
        repository: artifact.repository.clone(),
        revision: artifact.revision.to_ascii_lowercase(),
        quant: artifact
            .quant_hint
            .clone()
            .context("hosted text quant is missing")?,
        origin: origin.to_owned(),
        materialized_at_secs: now,
        last_used_at_secs: 0,
        artifact: ArtifactBinding {
            local_filename: destination
                .file_name()
                .and_then(|name| name.to_str())
                .context("managed artifact filename is not UTF-8")?
                .to_owned(),
            hub_filename: artifact.filename.clone(),
            bytes: artifact.bytes,
            sha256: artifact.sha256.to_ascii_lowercase(),
        },
        projector: None,
    };
    let sidecar = sidecar_path(&destination);
    write_binding(&sidecar, &binding)?;
    Ok(candidate_from_binding(
        binding,
        destination.to_path_buf(),
        sidecar,
    )?)
}

pub(super) fn hosted_destination(
    artifact: &HubGgufArtifact,
    explicit_output: Option<&Path>,
) -> Result<PathBuf> {
    use sha2::Digest;
    let basename = safe_basename(&artifact.filename)?;
    let revision_dir = managed_revision_dir(
        &managed_model_root()?,
        &artifact.repository,
        &artifact.revision,
    )?;
    // One bounded directory level keeps arbitrary Hub subpaths distinct while
    // preserving the filename and fitting the existing bounded cache scan.
    let default = revision_dir
        .join(hex::encode(sha2::Sha256::digest(
            artifact.filename.as_bytes(),
        )))
        .join(basename);
    resolve_output_path(explicit_output, default)
}

/// Admit either a legacy materialized regular file or the current managed
/// Hugging Face cache link for this exact immutable hosted artifact. A link
/// only wins when its retained target is the active repository's digest-named
/// blob and the exact-revision snapshot still resolves to that same inode.
pub(super) fn verify_or_refuse_existing_hosted_destination(
    destination: &Path,
    artifact: &HubGgufArtifact,
) -> Result<bool> {
    let metadata = match fs::symlink_metadata(destination) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    if metadata.file_type().is_symlink() {
        if retain_managed_hub_cache_link(
            destination,
            &artifact.repository,
            &artifact.revision,
            &artifact.filename,
            artifact.bytes,
            &artifact.sha256,
        )?
        .is_some()
        {
            return Ok(true);
        }
        if managed_hub_cache_link_is_expected_dangling(
            destination,
            &artifact.repository,
            &artifact.sha256,
        )? {
            return Ok(false);
        }
        bail!(
            "destination conflicts with the selected immutable hosted artifact: {}",
            destination.display()
        );
    }
    verify_or_refuse_existing_destination(destination, artifact.bytes, &artifact.sha256)
}

fn prepare_selected_local_decision(
    candidate: Candidate,
    explicit_output: Option<&Path>,
    warnings: &mut Vec<String>,
) -> Result<(Candidate, bool)> {
    prepare_selected_local_decision_with_preflight(
        candidate,
        explicit_output,
        warnings,
        |_, _, _, _, _, _| Ok(()),
    )
}

#[cfg(test)]
pub(super) fn prepare_selected_local(
    candidate: Candidate,
    explicit_output: Option<&Path>,
    warnings: &mut Vec<String>,
) -> Result<Candidate> {
    prepare_selected_local_decision(candidate, explicit_output, warnings)
        .map(|(candidate, _)| candidate)
}

pub(super) fn prepare_selected_local_decision_with_preflight(
    mut candidate: Candidate,
    explicit_output: Option<&Path>,
    warnings: &mut Vec<String>,
    mut pair_preflight: impl FnMut(
        &str,
        &Path,
        &Path,
        u64,
        bool,
        Option<(&Path, &Path, u64, bool)>,
    )
        -> std::result::Result<(), crate::input::hf_download::DownloadError>,
) -> Result<(Candidate, bool)> {
    let mut suppress_automatic_projector = false;
    if let Some(explicit) = explicit_output {
        let default = managed_revision_dir(
            &managed_model_root()?,
            &candidate.repository,
            &candidate.revision,
        )?
        .join(
            candidate
                .path
                .file_name()
                .context("selected local artifact has no filename")?,
        );
        let destination = resolve_output_path(Some(explicit), default)?;
        if destination != candidate.path {
            let text_plan = PreparedLocalArtifact::prepare(
                &candidate.path,
                &destination,
                candidate.bytes,
                &candidate.sha256,
            )?;
            let text_destination_exact = !text_plan.needs_copy();
            let projector_plan = match candidate.projector.clone() {
                Some((path, bytes, sha256)) => match verify_candidate_projector(&candidate) {
                    Ok(Some(_)) => {
                        let projector_destination = destination
                            .parent()
                            .context("selected destination has no parent")?
                            .join(
                                path.file_name()
                                    .context("selected mmproj has no filename")?,
                            );
                        match PreparedLocalArtifact::prepare(
                            &path,
                            &projector_destination,
                            bytes,
                            &sha256,
                        ) {
                            Ok(prepared) => {
                                Some((path, projector_destination, bytes, sha256, prepared))
                            }
                            Err(error) => {
                                suppress_automatic_projector = true;
                                warnings.push(format!(
                                    "automatic local mmproj destination conflicts; serving text-only: {error}"
                                ));
                                None
                            }
                        }
                    }
                    Ok(None) => {
                        suppress_automatic_projector = true;
                        warnings
                            .push("bound local mmproj is unavailable; serving text-only".into());
                        None
                    }
                    Err(error) => {
                        suppress_automatic_projector = true;
                        warnings.push(format!(
                            "bound local mmproj verification failed; serving text-only: {error}"
                        ));
                        None
                    }
                },
                None => None,
            };
            let pair_preflight_result = check_local_artifact_pair_plan_with_authorities(
                &candidate.repository,
                text_plan.source_device_id(),
                text_plan.destination(),
                text_plan.destination_device_id(),
                text_plan.destination_available_bytes(),
                candidate.bytes,
                text_destination_exact,
                projector_plan
                    .as_ref()
                    .map(|(_, destination, bytes, _, prepared)| {
                        (
                            prepared.source_device_id(),
                            destination.as_path(),
                            prepared.destination_device_id(),
                            prepared.destination_available_bytes(),
                            *bytes,
                            !prepared.needs_copy(),
                        )
                    }),
            )
            .and_then(|()| {
                pair_preflight(
                    &candidate.repository,
                    &candidate.path,
                    &destination,
                    candidate.bytes,
                    text_destination_exact,
                    projector_plan
                        .as_ref()
                        .map(|(source, destination, bytes, _, prepared)| {
                            (
                                source.as_path(),
                                destination.as_path(),
                                *bytes,
                                !prepared.needs_copy(),
                            )
                        }),
                )
            });
            let projector_plan = match (projector_plan, pair_preflight_result) {
                (Some(_), Err(error)) => {
                    suppress_automatic_projector = true;
                    warnings.push(format!(
                        "automatic local text/mmproj pair preflight failed; serving text-only: {error}"
                    ));
                    check_local_artifact_pair_plan_with_authorities(
                        &candidate.repository,
                        text_plan.source_device_id(),
                        text_plan.destination(),
                        text_plan.destination_device_id(),
                        text_plan.destination_available_bytes(),
                        candidate.bytes,
                        text_destination_exact,
                        None,
                    )?;
                    pair_preflight(
                        &candidate.repository,
                        &candidate.path,
                        &destination,
                        candidate.bytes,
                        text_destination_exact,
                        None,
                    )?;
                    None
                }
                (None, Err(error)) => return Err(error.into()),
                (plan, Ok(())) => plan,
            };
            let projector_current = match projector_plan.as_ref() {
                Some((_, _, _, _, prepared)) => prepared.is_current()?,
                None => true,
            };
            if !text_plan.is_current()? || !projector_current {
                bail!("local text/projector authority changed after disk preflight");
            }
            text_plan.materialize(&candidate.repository, candidate.bytes, &candidate.sha256)?;
            let projector = match projector_plan {
                Some((_, projector_destination, bytes, sha256, prepared)) => {
                    match prepared.materialize(&candidate.repository, bytes, &sha256) {
                        Ok(()) => Some((projector_destination, bytes, sha256)),
                        Err(error) => {
                            suppress_automatic_projector = true;
                            warnings.push(format!(
                                "automatic local mmproj materialization failed; serving text-only: {error}"
                            ));
                            None
                        }
                    }
                }
                None => None,
            };
            candidate.path = destination.clone();
            candidate.root = destination
                .parent()
                .context("selected destination has no parent")?
                .to_path_buf();
            candidate.materialized_at_secs = now_secs();
            candidate.origin = "local_adoption".to_owned();
            candidate.projector = projector;
            candidate.sidecar = None;
            candidate.receipt_target_identity = None;
        }
    }

    if candidate.sidecar.is_none() {
        let sidecar = sidecar_path(&candidate.path);
        let binding = binding_from_candidate(&candidate)?;
        match write_binding(&sidecar, &binding) {
            Ok(()) => candidate.sidecar = Some(sidecar),
            Err(error) => warnings.push(format!(
                "could not persist local model use history beside {}: {error}",
                candidate.path.display()
            )),
        }
    }
    Ok((candidate, suppress_automatic_projector))
}

pub(super) fn binding_from_candidate(candidate: &Candidate) -> Result<ManagedBinding> {
    let artifact_filename = candidate
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .context("selected local artifact filename is not UTF-8")?
        .to_owned();
    let projector = candidate
        .projector
        .as_ref()
        .map(|(path, bytes, sha256)| {
            Ok::<ArtifactBinding, anyhow::Error>(ArtifactBinding {
                local_filename: path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .context("selected local mmproj filename is not UTF-8")?
                    .to_owned(),
                hub_filename: path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .context("selected local mmproj filename is not UTF-8")?
                    .to_owned(),
                bytes: *bytes,
                sha256: sha256.to_ascii_lowercase(),
            })
        })
        .transpose()?;
    Ok(ManagedBinding {
        schema_version: SCHEMA_VERSION,
        repository: candidate.repository.clone(),
        revision: candidate.revision.to_ascii_lowercase(),
        quant: candidate.quant.as_str().to_owned(),
        origin: candidate.origin.clone(),
        materialized_at_secs: candidate.materialized_at_secs,
        last_used_at_secs: candidate.last_used_at_secs,
        artifact: ArtifactBinding {
            local_filename: artifact_filename.clone(),
            hub_filename: candidate.hub_filename.clone().unwrap_or(artifact_filename),
            bytes: candidate.bytes,
            sha256: candidate.sha256.to_ascii_lowercase(),
        },
        projector,
    })
}

#[cfg(test)]
pub(super) fn native_convert(
    catalog: &HubGgufCatalog,
    quant: QuantType,
    explicit_output: Option<&Path>,
    exact_product_bytes: Option<u64>,
) -> Result<Candidate> {
    let mut silent = |_| {};
    native_convert_with_progress(
        catalog,
        quant,
        explicit_output,
        exact_product_bytes,
        &mut silent,
    )
    .map(|(candidate, _)| candidate)
}

pub(super) fn native_convert_with_progress(
    catalog: &HubGgufCatalog,
    quant: QuantType,
    explicit_output: Option<&Path>,
    exact_product_bytes: Option<u64>,
    progress: &mut StartupProgress<'_>,
) -> Result<(Candidate, bool)> {
    let default = default_convert_output(
        &managed_model_root()?,
        &catalog.repository,
        &catalog.revision,
        quant.as_str(),
    )?;
    let output = resolve_output_path(explicit_output, default)?;
    let destination_exists = match fs::symlink_metadata(&output) {
        Ok(_) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => return Err(error.into()),
    };
    if destination_exists {
        let authority = conversion_authority(&output)?.ok_or_else(|| {
            anyhow!(
                "native conversion destination exists without a valid hf2q conversion receipt: {}",
                output.display()
            )
        })?;
        if authority.quant != quant
            || authority.repository != catalog.repository
            || !authority.revision.eq_ignore_ascii_case(&catalog.revision)
        {
            bail!(
                "native conversion destination conflicts with the requested repository/revision/quant: {}",
                output.display()
            );
        }
        verify_candidate(&authority)?;
        return Ok((authority, false));
    }
    progress(StartupEvent::NativeConversion {
        repository: catalog.repository.clone(),
        quant: quant.as_str().to_owned(),
    });
    let source_plan = crate::input::hf_download::resolve_native_source_plan(
        HfModelReference::parse(&catalog.repository, Some(&catalog.revision))?,
    )?;
    if source_plan.repository != catalog.repository
        || !source_plan.revision.eq_ignore_ascii_case(&catalog.revision)
    {
        bail!("native source plan changed repository/revision during conversion planning");
    }
    let planned_product_bytes = exact_product_bytes.unwrap_or_else(|| {
        planned_native_product_bytes(
            source_plan.total_weight_bytes,
            source_plan.output_upper_bound_bytes,
            source_plan.requires_projector,
            false,
            false,
        )
    });
    crate::input::hf_download::check_native_source_conversion_plan(
        &source_plan,
        &output,
        planned_product_bytes,
    )?;
    let child_output =
        Command::new(std::env::current_exe().context("resolve current hf2q executable")?)
            .arg("--terminal-graphics")
            .arg("off")
            .arg("convert")
            .arg(&catalog.repository)
            .arg("--revision")
            .arg(&catalog.revision)
            .arg("--quant")
            .arg(quant.as_str().to_ascii_lowercase())
            .arg("--output")
            .arg(&output)
            .arg("--no-clobber")
            .output()
            .context("launch native hf2q conversion")?;
    if !child_output.status.success() {
        let detail = bounded_child_stderr(&child_output.stderr);
        bail!(
            "native hf2q conversion failed with {}{}",
            child_output.status,
            if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            }
        );
    }
    let authority =
        conversion_authority(&output)?.context("native conversion emitted no valid receipt")?;
    if authority.quant != quant
        || authority.repository != catalog.repository
        || authority.revision != catalog.revision
    {
        bail!("native conversion receipt does not match the requested repository/revision/quant");
    }
    verify_candidate(&authority)?;
    Ok((authority, true))
}

fn bounded_child_stderr(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .chars()
        .filter(|ch| !ch.is_control() || matches!(ch, '\n' | '\t'))
        .take(4096)
        .collect::<String>()
        .trim()
        .to_owned()
}
