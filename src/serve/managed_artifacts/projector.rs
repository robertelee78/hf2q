use super::*;

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
            candidate, model_dirs, &catalog, warnings,
        )),
        Err(error) => {
            warnings.push(format!(
                "multimodal projector metadata unavailable; serving text-only: {error}"
            ));
            Ok(None)
        }
    }
}

pub(super) fn best_effort_projector_with_catalog(
    candidate: &mut Candidate,
    model_dirs: &[PathBuf],
    catalog: &HubGgufCatalog,
    warnings: &mut Vec<String>,
) -> Option<PathBuf> {
    match resolve_projector_with_catalog(candidate, model_dirs, catalog, warnings) {
        Ok(path) => path,
        Err(error) => {
            warnings.push(format!(
                "automatic mmproj preparation failed; serving text-only: {error}"
            ));
            None
        }
    }
}

fn resolve_projector_with_catalog(
    candidate: &mut Candidate,
    model_dirs: &[PathBuf],
    catalog: &HubGgufCatalog,
    warnings: &mut Vec<String>,
) -> Result<Option<PathBuf>> {
    if !text_requires_projector(&candidate.path)? {
        return Ok(None);
    }
    if let Some(path) = verify_candidate_projector(candidate)? {
        return Ok(Some(path));
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
    if !verify_or_refuse_existing_destination(&destination, artifact.bytes, &artifact.sha256)? {
        let source = match find_matching_loose(&artifact, model_dirs)? {
            Some(path) => path,
            None => {
                check_hub_artifact_plan(&artifact, &destination)?;
                download_hub_companion(&artifact)?
            }
        };
        materialize_preverified_exact(
            &source,
            &destination,
            &artifact.repository,
            artifact.bytes,
            &artifact.sha256,
        )?;
    }
    candidate.projector = Some((destination.clone(), artifact.bytes, artifact.sha256.clone()));
    if let Some(sidecar) = candidate.sidecar.as_ref() {
        let mut binding = read_binding(sidecar)?.context("managed text binding disappeared")?;
        binding.projector = Some(ArtifactBinding {
            local_filename: destination
                .file_name()
                .and_then(|name| name.to_str())
                .context("mmproj filename is not UTF-8")?
                .to_owned(),
            hub_filename: artifact.filename,
            bytes: artifact.bytes,
            sha256: artifact.sha256,
        });
        write_binding(sidecar, &binding)?;
    }
    Ok(Some(destination))
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
