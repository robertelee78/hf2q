use super::*;

pub(crate) fn resolve_repository(
    spec: &RepositoryModelSpec,
    explicit_output: Option<&Path>,
    model_dirs: &[PathBuf],
    cache: &mut ModelCache,
    hardware: &HardwareProfile,
    prepare_projector: bool,
) -> Result<ResolvedManagedModel> {
    let recommended = match spec.quant {
        Some(exact) => exact,
        None => select_quant(&GpuInfo::from_hardware_profile(hardware))?,
    };
    let mut warnings = Vec::new();
    if let Some((mut candidate, _local_lock)) =
        select_local(spec, model_dirs, cache, recommended, None, &mut warnings)?
    {
        candidate = prepare_selected_local(candidate, explicit_output, &mut warnings)?;
        let mmproj = if prepare_projector {
            resolve_projector(&mut candidate, model_dirs, &mut warnings)?
        } else {
            None
        };
        return Ok(candidate.into_resolved(mmproj, warnings));
    }

    let reference = HfModelReference::parse(&spec.repository, None)?;
    let catalog = resolve_hub_gguf_catalog(reference)
        .with_context(|| format!("resolve hosted GGUF metadata for {}", spec.repository))?;
    let selectable = catalog
        .artifacts
        .iter()
        .filter(|artifact| artifact.selectable && artifact.role == "text_model")
        .cloned()
        .collect::<Vec<_>>();

    let loose = find_best_matching_loose(&selectable, spec.quant, recommended, model_dirs)?;
    // Digest-bound loose bytes can disambiguate repositories that publish
    // more than one filename for the same quant. Do not reject on filename
    // ambiguity before checking bytes the operator already owns.
    let selected = if loose.is_none() {
        select_hosted(&selectable, spec.quant, recommended)?
    } else {
        None
    };
    let target_quant = loose
        .as_ref()
        .and_then(|(artifact, _)| artifact.quant_hint.as_deref())
        .or_else(|| {
            selected
                .as_ref()
                .and_then(|artifact| artifact.quant_hint.as_deref())
        })
        .and_then(|value| QuantType::from_canonical_str(value).ok())
        .unwrap_or_else(|| spec.quant.unwrap_or(recommended));
    let _resolution_lock = cache
        .lock_quant(&spec.repository, target_quant)
        .with_context(|| {
            format!(
                "lock managed resolution for {}:{}",
                spec.repository, target_quant
            )
        })?;
    if let Some((mut candidate, _local_lock)) = select_local(
        spec,
        model_dirs,
        cache,
        recommended,
        Some(target_quant),
        &mut warnings,
    )? {
        candidate = prepare_selected_local(candidate, explicit_output, &mut warnings)?;
        let mmproj = if prepare_projector {
            resolve_projector(&mut candidate, model_dirs, &mut warnings)?
        } else {
            None
        };
        return Ok(candidate.into_resolved(mmproj, warnings));
    }
    let mut candidate = if let Some((artifact, loose)) = loose {
        materialize_hosted(&loose, &artifact, explicit_output, "manual_adoption")?
    } else if let Some(artifact) = selected {
        let destination = hosted_destination(&artifact, explicit_output)?;
        if verify_or_refuse_existing_destination(&destination, artifact.bytes, &artifact.sha256)? {
            bind_hosted_destination(&destination, &artifact, "existing_destination")?
        } else {
            check_hub_artifact_plan(&artifact, &destination)?;
            let cached = download_hub_gguf(&artifact)?;
            materialize_hosted(&cached, &artifact, explicit_output, "hosted_download")?
        }
    } else if selectable.is_empty() {
        native_convert(&catalog, spec.quant.unwrap_or(recommended), explicit_output)?
    } else {
        unreachable!("select_hosted returns an actionable error when hosted choices exist")
    };
    candidate = prepare_selected_local(candidate, None, &mut warnings)?;
    let mmproj = prepare_projector
        .then(|| {
            best_effort_projector_with_catalog(&mut candidate, model_dirs, &catalog, &mut warnings)
        })
        .flatten();
    Ok(candidate.into_resolved(mmproj, warnings))
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
    let available = artifacts
        .iter()
        .filter_map(|artifact| artifact.quant_hint.as_deref())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join(", ");
    bail!("requested quant {desired} is not hosted; available supported quants: {available}")
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

pub(super) fn prepare_selected_local(
    mut candidate: Candidate,
    explicit_output: Option<&Path>,
    warnings: &mut Vec<String>,
) -> Result<Candidate> {
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
            materialize_preverified_exact(
                &candidate.path,
                &destination,
                &candidate.repository,
                candidate.bytes,
                &candidate.sha256,
            )?;
            let projector = match candidate.projector.as_ref() {
                Some((path, bytes, sha256))
                    if verify_candidate_projector(&candidate)?.is_some() =>
                {
                    let projector_destination = destination
                        .parent()
                        .context("selected destination has no parent")?
                        .join(
                            path.file_name()
                                .context("selected mmproj has no filename")?,
                        );
                    materialize_preverified_exact(
                        path,
                        &projector_destination,
                        &candidate.repository,
                        *bytes,
                        sha256,
                    )?;
                    Some((projector_destination, *bytes, sha256.clone()))
                }
                _ => None,
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
    Ok(candidate)
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

fn native_convert(
    catalog: &HubGgufCatalog,
    quant: QuantType,
    explicit_output: Option<&Path>,
) -> Result<Candidate> {
    let default = default_convert_output(
        &managed_model_root()?,
        &catalog.repository,
        &catalog.revision,
        quant.as_str(),
    )?;
    let output = resolve_output_path(explicit_output, default)?;
    let status = Command::new(std::env::current_exe().context("resolve current hf2q executable")?)
        .arg("convert")
        .arg(&catalog.repository)
        .arg("--revision")
        .arg(&catalog.revision)
        .arg("--quant")
        .arg(quant.as_str().to_ascii_lowercase())
        .arg("--output")
        .arg(&output)
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
