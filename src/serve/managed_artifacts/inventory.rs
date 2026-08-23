use super::*;

pub(super) fn find_matching_loose(
    artifact: &HubGgufArtifact,
    model_dirs: &[PathBuf],
) -> Result<Option<PathBuf>> {
    for root in scan_roots(model_dirs)? {
        let mut found = None;
        visit_files(&root, |path, metadata| {
            if found.is_some()
                || metadata.len() != artifact.bytes
                || path
                    .extension()
                    .and_then(|v| v.to_str())
                    .is_none_or(|ext| !ext.eq_ignore_ascii_case("gguf"))
            {
                return Ok(());
            }
            if artifact.role == "text_model" {
                let expected = artifact
                    .quant_hint
                    .as_deref()
                    .and_then(|value| QuantType::from_canonical_str(value).ok());
                if quant_type_from_gguf_path(path).ok() != expected {
                    return Ok(());
                }
            }
            if crate::core::sha256::compute_file_sha256(path)?
                .eq_ignore_ascii_case(&artifact.sha256)
            {
                found = Some(path.to_path_buf());
            }
            Ok(())
        })?;
        if found.is_some() {
            return Ok(found);
        }
    }
    Ok(None)
}

pub(crate) fn print_inventory(model_dirs: &[PathBuf]) -> Result<()> {
    let cache_root = crate::serve::cache::default_root()?;
    let manifest_path = cache_root.join("manifest.json");
    let cache_manifest = manifest_path
        .is_file()
        .then(|| crate::serve::cache::read_manifest(&manifest_path))
        .transpose()?;
    let inventory = LocalArtifactInventory::for_serve(model_dirs)?;
    let local = inventory.discover(
        None,
        cache_manifest
            .as_ref()
            .map(|manifest| (cache_root.as_path(), manifest)),
    );
    let managed = scan_bindings(model_dirs, None)?;
    println!("REPOSITORY\tREVISION\tQUANT\tORIGIN\tLAST_USED\tMMPROJ\tPATH");
    let mut paths = BTreeSet::new();
    for candidate in managed {
        paths.insert(candidate.path.clone());
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            candidate.repository,
            short_revision(&candidate.revision),
            candidate.quant,
            candidate.origin,
            display_timestamp(candidate.last_used_at_secs),
            if candidate.projector.is_some() {
                "yes"
            } else {
                "no"
            },
            candidate.path.display()
        );
    }
    for artifact in local.artifacts {
        if paths.insert(artifact.path.clone()) {
            let last_used = cache_manifest
                .as_ref()
                .and_then(|manifest| manifest.models.get(&artifact.repository))
                .map_or(0, |model| model.last_accessed_secs);
            let cached_projector = cache_manifest
                .as_ref()
                .and_then(|manifest| manifest.models.get(&artifact.repository))
                .and_then(|model| model.quantizations.get(&artifact.quant_hint))
                .and_then(|entry| entry.mmproj_path.as_ref())
                .and_then(|path| {
                    projector_authority_from_receipt(path, &artifact.repository, &artifact.revision)
                        .ok()
                        .flatten()
                });
            let receipt_projector = paired_projector_path(&artifact.path).and_then(|path| {
                projector_authority_from_receipt(&path, &artifact.repository, &artifact.revision)
                    .ok()
                    .flatten()
            });
            println!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                artifact.repository,
                short_revision(&artifact.revision),
                artifact.quant_hint,
                artifact.provenance.as_str(),
                display_timestamp(last_used),
                if cached_projector.or(receipt_projector).is_some() {
                    "yes"
                } else {
                    "no"
                },
                artifact.path.display()
            );
        }
    }
    for root in scan_roots(model_dirs)? {
        visit_files(&root, |path, _| {
            if paths.contains(path)
                || path
                    .extension()
                    .and_then(|v| v.to_str())
                    .is_none_or(|ext| !ext.eq_ignore_ascii_case("gguf"))
            {
                return Ok(());
            }
            if let Ok(quant) = quant_type_from_gguf_path(path) {
                println!("?\t?\t{}\tunbound\t-\t?\t{}", quant, path.display());
            }
            Ok(())
        })?;
    }
    for warning in local.warnings {
        eprintln!("Warning: {warning}");
    }
    Ok(())
}

pub(super) fn scan_roots(model_dirs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut roots = vec![
        managed_model_root()?,
        std::env::current_dir()?.join("models"),
    ];
    roots.extend(model_dirs.iter().cloned());
    roots.sort();
    roots.dedup();
    Ok(roots)
}

pub(super) fn visit_files(
    root: &Path,
    mut visit: impl FnMut(&Path, &fs::Metadata) -> Result<()>,
) -> Result<()> {
    let metadata = match fs::symlink_metadata(root) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => metadata,
        _ => return Ok(()),
    };
    let _ = metadata;
    let mut queue = std::collections::VecDeque::from([(root.to_path_buf(), 0usize)]);
    let mut visited = 0usize;
    while let Some((directory, depth)) = queue.pop_front() {
        let mut entries = match fs::read_dir(&directory) {
            Ok(entries) => entries
                .filter_map(std::result::Result::ok)
                .collect::<Vec<_>>(),
            Err(_) => continue,
        };
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            visited += 1;
            if visited > MAX_SCAN_ENTRIES {
                return Ok(());
            }
            let metadata = match fs::symlink_metadata(entry.path()) {
                Ok(metadata) if !metadata.file_type().is_symlink() => metadata,
                _ => continue,
            };
            if metadata.is_dir() && depth < MAX_SCAN_DEPTH {
                queue.push_back((entry.path(), depth + 1));
            } else if metadata.is_file() {
                visit(&entry.path(), &metadata)?;
            }
        }
    }
    Ok(())
}
