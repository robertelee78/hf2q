use super::*;

pub(crate) fn resolve_repository(
    spec: &RepositoryModelSpec,
    explicit_output: Option<&Path>,
    model_dirs: &[PathBuf],
    cache: &mut ModelCache,
    hardware: &HardwareProfile,
    prepare_projector: bool,
    configured_quant: Option<&str>,
) -> Result<ResolvedManagedModel> {
    let pool_budget_bytes = LoadedPool::from_hardware(hardware).memory_budget_bytes();
    let mut warnings = Vec::new();
    let initial_local = select_local(
        spec,
        model_dirs,
        cache,
        None,
        hardware.available_memory_bytes,
        pool_budget_bytes,
        &mut warnings,
    )?;
    // A successfully used managed quant is the strongest automatic-choice
    // signal. It cannot be displaced by a merely newer loose/Hub-cache file,
    // so return it after the one verification already performed by
    // select_local. This avoids catalog latency and tens of GiB of duplicate
    // hashing on the normal repeat-serve path.
    let initial_local = match initial_local {
        Some((mut candidate, identity, local_lock)) if candidate.last_used_at_secs > 0 => {
            let needs_hosted_projector_plan = if prepare_projector {
                match verify_candidate_projector(&candidate) {
                    Ok(Some(_)) => false,
                    Ok(None) => true,
                    Err(error) => {
                        warnings.push(format!(
                            "local mmproj verification failed before catalog planning: {error}"
                        ));
                        true
                    }
                }
            } else {
                false
            };
            if needs_hosted_projector_plan {
                drop(local_lock);
                Some((candidate, identity))
            } else {
                let (prepared, suppress_projector) =
                    prepare_selected_local_decision(candidate, explicit_output, &mut warnings)?;
                candidate = prepared;
                let mmproj = if prepare_projector && !suppress_projector {
                    verify_candidate_projector(&candidate)?
                } else {
                    None
                };
                drop(local_lock);
                return Ok(candidate.into_resolved(mmproj, warnings));
            }
        }
        Some((candidate, identity, lock)) => {
            drop(lock);
            Some((candidate, identity))
        }
        None => None,
    };

    let reference = HfModelReference::parse(&spec.repository, None)?;
    let mut catalog = match resolve_hub_gguf_catalog(reference) {
        Ok(catalog) => catalog,
        Err(error) => {
            let Some((mut candidate, identity)) = initial_local else {
                return Err(error).with_context(|| {
                    format!("resolve hosted GGUF metadata for {}", spec.repository)
                });
            };
            let _lock = cache.lock_quant(&spec.repository, candidate.quant)?;
            if !reverify_candidate_after_catalog(&candidate, identity, &mut warnings) {
                return Err(error).context("local fallback changed during repository resolution");
            }
            let (prepared, suppress_projector) =
                prepare_selected_local_decision(candidate, explicit_output, &mut warnings)?;
            candidate = prepared;
            let mmproj = if prepare_projector
                && !suppress_projector
                && text_requires_projector(&candidate.path)?
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
            return Ok(candidate.into_resolved(mmproj, warnings));
        }
    };
    let selectable = catalog
        .artifacts
        .iter()
        .filter(|artifact| artifact.selectable && artifact.role == "text_model")
        .cloned()
        .collect::<Vec<_>>();
    let selectable = selectable
        .into_iter()
        .filter(|artifact| {
            if spec.quant.is_some()
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

    let excluded = initial_local
        .as_ref()
        .map(|(_, identity)| std::slice::from_ref(identity))
        .unwrap_or(&[]);
    let manual =
        find_best_matching_loose(&selectable, spec.quant, model_dirs, excluded, &mut warnings)?;
    let cached = find_best_matching_cached_hub(&selectable, spec.quant, &mut warnings)?;
    let materialized =
        |candidate: &local::ExactHostedLocal| system_time_secs(candidate.materialized);
    let (loose, loose_origin) = match (manual, cached) {
        (Some(manual), Some(cached)) if materialized(&cached) > materialized(&manual) => {
            (Some(cached), "hf_hub_cache_adoption")
        }
        (Some(manual), Some(_)) => (Some(manual), "manual_adoption"),
        (Some(manual), None) => (Some(manual), "manual_adoption"),
        (None, Some(cached)) => (Some(cached), "hf_hub_cache_adoption"),
        (None, None) => (None, "manual_adoption"),
    };
    if let Some((candidate, identity)) = initial_local {
        let loose_recency = loose
            .as_ref()
            .map(|candidate| (false, 0, materialized(candidate)))
            .unwrap_or((false, 0, 0));
        if loose.is_none() || bound_candidate_is_at_least_as_recent(&candidate, loose_recency.2) {
            let _lock = cache.lock_quant(&spec.repository, candidate.quant)?;
            if reverify_candidate_after_catalog(&candidate, identity, &mut warnings) {
                let (candidate, mmproj) = prepare_local_candidate_with_catalog(
                    candidate,
                    explicit_output,
                    model_dirs,
                    &catalog,
                    prepare_projector,
                    &mut warnings,
                )?;
                return Ok(candidate.into_resolved(mmproj, warnings));
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
    // Digest-bound loose bytes can disambiguate repositories that publish
    // more than one filename for the same quant. Do not reject on filename
    // ambiguity before checking bytes the operator already owns.
    let mut selected_requires_projector = false;
    let selected = if loose.is_none() {
        select_compatible_hosted(
            &selectable,
            spec.quant,
            recommended,
            |artifact| {
                let compatibility = validate_hub_gguf_header_compatibility(artifact)?;
                selected_requires_projector = compatibility.requires_projector;
                Ok(())
            },
            &mut warnings,
        )?
    } else {
        None
    };
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
    if let Some((candidate, _identity, _local_lock)) = select_local(
        spec,
        model_dirs,
        cache,
        Some(target_quant),
        hardware.available_memory_bytes,
        pool_budget_bytes,
        &mut warnings,
    )? {
        let selected_loose_materialized_at =
            loose.as_ref().map(|candidate| materialized(candidate));
        if post_lock_local_candidate_wins(&candidate, selected_loose_materialized_at) {
            let (candidate, mmproj) = prepare_local_candidate_with_catalog(
                candidate,
                explicit_output,
                model_dirs,
                &catalog,
                prepare_projector,
                &mut warnings,
            )?;
            return Ok(candidate.into_resolved(mmproj, warnings));
        }
    }
    let mut suppress_automatic_projector = false;
    let mut prepared_projector = None;
    let mut candidate = if let Some(loose) = loose {
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
                    PreparedProjectorSource::Existing(_) => {
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
        candidate
    } else if let Some(artifact) = selected {
        let destination = hosted_destination(&artifact, explicit_output)?;
        let text_destination_exact =
            verify_or_refuse_existing_destination(&destination, artifact.bytes, &artifact.sha256)?;
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
            let cached = download_hub_gguf(&artifact)?;
            materialize_hosted(&cached, &artifact, explicit_output, "hosted_download")?
        };
        prepared_projector = projector_plan;
        candidate
    } else {
        native_convert(
            &catalog,
            native_fallback_quant,
            explicit_output,
            native_product_bytes,
        )?
    };
    let (prepared, local_suppress_projector) =
        prepare_selected_local_decision(candidate, None, &mut warnings)?;
    candidate = prepared;
    suppress_automatic_projector |= local_suppress_projector;
    let mmproj = if let Some(plan) = prepared_projector {
        match materialize_prepared_projector(plan, &mut candidate, &mut warnings) {
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
                best_effort_projector_with_catalog(
                    &mut candidate,
                    model_dirs,
                    &catalog,
                    hosted_pair_requires_projector(
                        catalog.requires_projector,
                        selected_requires_projector,
                    ),
                    &mut warnings,
                )
            })
            .flatten()
    };
    Ok(candidate.into_resolved(mmproj, warnings))
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
    text_destination: &Path,
    catalog: &HubGgufCatalog,
    required: bool,
) -> Result<Option<(HubGgufArtifact, PathBuf)>> {
    if !required {
        return Ok(None);
    }
    let expected = expected_projector_sha256(&candidate.path)?;
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
    explicit_output: Option<&Path>,
    model_dirs: &[PathBuf],
    catalog: &HubGgufCatalog,
    prepare_projector: bool,
    warnings: &mut Vec<String>,
) -> Result<(Candidate, Option<PathBuf>)> {
    prepare_local_candidate_with_catalog_resolver(
        candidate,
        explicit_output,
        model_dirs,
        catalog,
        prepare_projector,
        warnings,
        resolve_hub_gguf_catalog,
    )
}

pub(super) fn prepare_local_candidate_with_catalog_resolver(
    mut candidate: Candidate,
    explicit_output: Option<&Path>,
    model_dirs: &[PathBuf],
    catalog: &HubGgufCatalog,
    prepare_projector: bool,
    warnings: &mut Vec<String>,
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
    let text_projector_required = text_requires_projector(&candidate.path)?;
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
    let projector_plan =
        planned_local_projector(&candidate, &text_destination, catalog, projector_required)
            .and_then(|plan| {
                plan.map(|(artifact, destination)| {
                    prepare_projector_action(artifact, destination, model_dirs)
                })
                .transpose()
            });
    let projector_plan = match projector_plan {
        Ok(Some(plan)) => {
            let pair_preflight = match &plan.source {
                PreparedProjectorSource::Existing(_) => {
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
    }
    let projector = match projector_plan {
        Some(plan) => match materialize_prepared_projector(plan, &mut candidate, warnings) {
            Ok(path) => Some(path),
            Err(error) => {
                warnings.push(format!(
                    "automatic mmproj preparation failed; serving text-only: {error}"
                ));
                None
            }
        },
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
    mut probe: impl FnMut(&HubGgufArtifact) -> Result<(), DownloadError>,
    warnings: &mut Vec<String>,
) -> Result<Option<HubGgufArtifact>> {
    let mut candidates = artifacts.to_vec();
    loop {
        let Some(artifact) = select_hosted(&candidates, exact, recommended)? else {
            return Ok(None);
        };
        match probe(&artifact) {
            Ok(()) => return Ok(Some(artifact)),
            Err(DownloadError::IncompatibleHostedGguf { reason }) => {
                warnings.push(format!(
                    "ignored incompatible hosted {} before payload transfer: {reason}",
                    artifact.filename
                ));
                candidates.retain(|candidate| candidate.filename != artifact.filename);
            }
            Err(error) => return Err(error.into()),
        }
    }
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
        quality_descending()
            .into_iter()
            .filter(|quant| quant_quality(*quant) <= quant_quality(desired))
            .collect()
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
    source: &Path,
    artifact: &HubGgufArtifact,
    explicit_output: Option<&Path>,
    origin: &str,
) -> Result<Candidate> {
    let destination = hosted_destination(artifact, explicit_output)?;
    materialize_preverified_exact(
        source,
        &destination,
        &artifact.repository,
        artifact.bytes,
        &artifact.sha256,
    )?;
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

fn hosted_destination(
    artifact: &HubGgufArtifact,
    explicit_output: Option<&Path>,
) -> Result<PathBuf> {
    let basename = safe_basename(&artifact.filename)?;
    let default = managed_revision_dir(
        &managed_model_root()?,
        &artifact.repository,
        &artifact.revision,
    )?
    .join(basename);
    resolve_output_path(explicit_output, default)
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

fn binding_from_candidate(candidate: &Candidate) -> Result<ManagedBinding> {
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
            hub_filename: artifact_filename,
            bytes: candidate.bytes,
            sha256: candidate.sha256.to_ascii_lowercase(),
        },
        projector,
    })
}

pub(super) fn native_convert(
    catalog: &HubGgufCatalog,
    quant: QuantType,
    explicit_output: Option<&Path>,
    exact_product_bytes: Option<u64>,
) -> Result<Candidate> {
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
        return Ok(authority);
    }
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
    let status = Command::new(std::env::current_exe().context("resolve current hf2q executable")?)
        .arg("convert")
        .arg(&catalog.repository)
        .arg("--revision")
        .arg(&catalog.revision)
        .arg("--quant")
        .arg(quant.as_str().to_ascii_lowercase())
        .arg("--output")
        .arg(&output)
        .arg("--no-clobber")
        .status()
        .context("launch native hf2q conversion")?;
    if !status.success() {
        bail!("native hf2q conversion failed with {status}");
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
    Ok(authority)
}
