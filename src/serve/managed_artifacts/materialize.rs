use super::*;

/// Check a final destination before any payload transfer. Exact bytes are an
/// idempotent hit; any other existing entry is a fail-closed conflict.
pub(super) fn verify_or_refuse_existing_destination(
    destination: &Path,
    bytes: u64,
    sha256: &str,
) -> Result<bool> {
    let metadata = match fs::symlink_metadata(destination) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() != bytes {
        bail!(
            "destination conflicts with the selected immutable artifact: {}",
            destination.display()
        );
    }
    if !crate::core::sha256::compute_file_sha256(destination)?.eq_ignore_ascii_case(sha256) {
        bail!(
            "destination conflicts with the selected immutable artifact: {}",
            destination.display()
        );
    }
    Ok(true)
}

#[cfg(test)]
pub(super) fn materialize_exact(
    source: &Path,
    destination: &Path,
    repository: &str,
    bytes: u64,
    sha256: &str,
) -> Result<()> {
    materialize_exact_with_link(
        source,
        destination,
        repository,
        bytes,
        sha256,
        |source, destination| fs::hard_link(source, destination),
    )
}

/// Materialize bytes whose complete SHA-256 was verified by the immediately
/// preceding selection/download step. A successful hard link preserves the
/// verified inode and therefore does not hash a multi-gigabyte payload twice.
pub(super) fn materialize_preverified_exact(
    source: &Path,
    destination: &Path,
    repository: &str,
    bytes: u64,
    sha256: &str,
) -> Result<()> {
    materialize_preverified_exact_with_link(
        source,
        destination,
        repository,
        bytes,
        sha256,
        |source, destination| fs::hard_link(source, destination),
    )
}

#[cfg(test)]
pub(super) fn materialize_exact_with_link(
    source: &Path,
    destination: &Path,
    repository: &str,
    bytes: u64,
    sha256: &str,
    link: impl FnOnce(&Path, &Path) -> std::io::Result<()>,
) -> Result<()> {
    let canonical = source
        .canonicalize()
        .with_context(|| format!("resolve materialization source {}", source.display()))?;
    let metadata = fs::metadata(&canonical)?;
    if !metadata.is_file()
        || metadata.len() != bytes
        || !crate::core::sha256::compute_file_sha256(&canonical)?.eq_ignore_ascii_case(sha256)
    {
        bail!("materialization source failed exact size/SHA-256 verification");
    }
    materialize_preverified_exact_with_link(
        &canonical,
        destination,
        repository,
        bytes,
        sha256,
        link,
    )
}

fn materialize_preverified_exact_with_link(
    source: &Path,
    destination: &Path,
    repository: &str,
    bytes: u64,
    sha256: &str,
    link: impl FnOnce(&Path, &Path) -> std::io::Result<()>,
) -> Result<()> {
    let source = source
        .canonicalize()
        .with_context(|| format!("resolve materialization source {}", source.display()))?;
    let source_metadata = fs::metadata(&source)?;
    if !source_metadata.is_file() || source_metadata.len() != bytes {
        bail!("materialization source is not the expected regular file");
    }
    if verify_or_refuse_existing_destination(destination, bytes, sha256)? {
        return Ok(());
    }
    let parent = destination
        .parent()
        .context("managed artifact has no parent")?;
    fs::create_dir_all(parent)?;
    let mut linked = false;
    match link(&source, destination) {
        Ok(()) => linked = true,
        Err(error) if error.raw_os_error() == Some(libc::EXDEV) => {
            check_hub_artifact_destination(repository, destination, bytes)?;
            let mut input = fs::File::open(&source)?;
            let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
            std::io::copy(&mut input, &mut temporary)?;
            temporary.as_file_mut().sync_all()?;
            if temporary.as_file().metadata()?.len() != bytes
                || !crate::core::sha256::compute_file_sha256(temporary.path())?
                    .eq_ignore_ascii_case(sha256)
            {
                bail!("copied artifact failed exact size/SHA-256 verification");
            }
            temporary
                .persist_noclobber(destination)
                .map_err(|error| error.error)?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            if verify_or_refuse_existing_destination(destination, bytes, sha256)? {
                return Ok(());
            }
            return Err(error.into());
        }
        Err(error) => return Err(error.into()),
    }
    let metadata = fs::symlink_metadata(destination)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() != bytes {
        bail!("materialized artifact failed regular-file/size verification");
    }
    // A preverified source is still a mutable path. Re-hash a newly
    // published hard link before writing authority so a same-size mutation
    // between selection and linking cannot bind the wrong bytes.
    if linked
        && !crate::core::sha256::compute_file_sha256(destination)?.eq_ignore_ascii_case(sha256)
    {
        fs::remove_file(destination).with_context(|| {
            format!(
                "remove materialized hard link after digest mismatch: {}",
                destination.display()
            )
        })?;
        bail!("materialized hard link failed final SHA-256 verification");
    }
    Ok(())
}
