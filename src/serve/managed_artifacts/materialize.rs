use super::*;

pub(super) enum PreparedLocalArtifact {
    Existing {
        retained: crate::core::bounded_file::StableRegularFile,
        destination: PathBuf,
    },
    Adoption {
        source: crate::core::bounded_file::StableRegularFile,
        destination_parent: crate::core::bounded_file::StableDirectory,
        destination_name: std::ffi::OsString,
        destination: PathBuf,
    },
}

impl PreparedLocalArtifact {
    pub(super) fn prepare(
        source: &Path,
        destination: &Path,
        bytes: u64,
        sha256: &str,
    ) -> Result<Self> {
        if verify_or_refuse_existing_destination(destination, bytes, sha256)? {
            let retained = retain_exact_digest(destination, bytes, sha256)?
                .context("exact local destination changed while retaining its authority")?;
            return Ok(Self::Existing {
                retained,
                destination: destination.to_path_buf(),
            });
        }
        let source = source
            .canonicalize()
            .with_context(|| format!("resolve materialization source {}", source.display()))?;
        let source = retain_exact_digest(&source, bytes, sha256)?
            .context("materialization source changed after exact verification")?;
        let parent = destination
            .parent()
            .context("managed artifact has no parent")?;
        let destination_parent =
            crate::core::bounded_file::StableDirectory::create_and_open(parent)
                .context("retain managed artifact destination parent")?;
        let destination_name = destination
            .file_name()
            .context("managed artifact destination has no filename")?
            .to_os_string();
        Ok(Self::Adoption {
            source,
            destination_parent,
            destination_name,
            destination: destination.to_path_buf(),
        })
    }

    pub(super) fn prepare_retained(
        source: crate::core::bounded_file::StableRegularFile,
        destination: &Path,
        bytes: u64,
        sha256: &str,
    ) -> Result<Self> {
        if verify_or_refuse_existing_destination(destination, bytes, sha256)? {
            let retained = retain_exact_digest(destination, bytes, sha256)?
                .context("exact local destination changed while retaining its authority")?;
            return Ok(Self::Existing {
                retained,
                destination: destination.to_path_buf(),
            });
        }
        if !source.is_stable()? {
            bail!("retained materialization source changed during planning");
        }
        let parent = destination
            .parent()
            .context("managed artifact has no parent")?;
        let destination_parent =
            crate::core::bounded_file::StableDirectory::create_and_open(parent)
                .context("retain managed artifact destination parent")?;
        let destination_name = destination
            .file_name()
            .context("managed artifact destination has no filename")?
            .to_os_string();
        Ok(Self::Adoption {
            source,
            destination_parent,
            destination_name,
            destination: destination.to_path_buf(),
        })
    }

    pub(super) fn destination(&self) -> &Path {
        match self {
            Self::Existing { destination, .. } | Self::Adoption { destination, .. } => destination,
        }
    }

    pub(super) fn needs_copy(&self) -> bool {
        matches!(self, Self::Adoption { .. })
    }

    pub(super) fn source_device_id(&self) -> Option<u64> {
        match self {
            Self::Existing { .. } => None,
            Self::Adoption { source, .. } => Some(source.device_id()),
        }
    }

    pub(super) fn destination_device_id(&self) -> Option<u64> {
        match self {
            Self::Existing { retained, .. } => Some(retained.device_id()),
            Self::Adoption {
                destination_parent, ..
            } => Some(destination_parent.device_id()),
        }
    }

    pub(super) fn destination_available_bytes(&self) -> Option<u64> {
        match self {
            Self::Existing { .. } => None,
            Self::Adoption {
                destination_parent, ..
            } => destination_parent.available_bytes(),
        }
    }

    pub(super) fn is_current(&self) -> Result<bool> {
        match self {
            Self::Existing { retained, .. } => Ok(retained.is_stable()?),
            Self::Adoption {
                source,
                destination_parent,
                ..
            } => Ok(source.is_stable()? && destination_parent.is_current()?),
        }
    }

    pub(super) fn materialize(self, repository: &str, bytes: u64, sha256: &str) -> Result<()> {
        match self {
            Self::Existing { retained, .. } => {
                if !retained.is_stable()? {
                    bail!("exact local destination changed before activation");
                }
                Ok(())
            }
            Self::Adoption {
                source,
                destination_parent,
                destination_name,
                ..
            } => materialize_retained_exact_at(
                source,
                destination_parent,
                &destination_name,
                repository,
                bytes,
                sha256,
            ),
        }
    }
}

fn retain_exact_digest(
    path: &Path,
    bytes: u64,
    sha256: &str,
) -> Result<Option<crate::core::bounded_file::StableRegularFile>> {
    let Some(mut retained) = crate::core::bounded_file::StableRegularFile::open_exact(path, bytes)?
    else {
        return Ok(None);
    };
    Ok(retained
        .sha256()?
        .is_some_and(|digest| digest.eq_ignore_ascii_case(sha256))
        .then_some(retained))
}

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
    if !crate::core::bounded_file::sha256_regular_nofollow_exact(destination, bytes)?
        .is_some_and(|digest| digest.eq_ignore_ascii_case(sha256))
    {
        bail!(
            "destination conflicts with the selected immutable artifact: {}",
            destination.display()
        );
    }
    Ok(true)
}

/// Materialize bytes whose complete SHA-256 was verified by the immediately
/// preceding selection/download step. On Apple filesystems a clone avoids a
/// second model-sized allocation; cross-filesystem fallback copies and hashes.
pub(super) fn materialize_preverified_exact(
    source: &Path,
    destination: &Path,
    repository: &str,
    bytes: u64,
    sha256: &str,
) -> Result<()> {
    let canonical = source
        .canonicalize()
        .with_context(|| format!("resolve materialization source {}", source.display()))?;
    let retained = crate::core::bounded_file::StableRegularFile::open_exact(&canonical, bytes)?
        .context("materialization source is not the expected regular file")?;
    materialize_retained_exact_with_proof(
        retained,
        destination,
        repository,
        bytes,
        sha256,
        RetainedProof::VerifyStagedClone,
    )
}

#[derive(Clone, Copy)]
enum RetainedProof {
    AlreadyVerified,
    VerifyStagedClone,
}

/// Adopt a source that was hashed and admitted through a retained descriptor.
/// The staged clone/copy is published create-if-absent, and the retained
/// source plus staged inode identities replace a second model-sized hash.
pub(super) fn materialize_retained_exact(
    source: crate::core::bounded_file::StableRegularFile,
    destination: &Path,
    repository: &str,
    bytes: u64,
    sha256: &str,
) -> Result<()> {
    materialize_retained_exact_with_proof(
        source,
        destination,
        repository,
        bytes,
        sha256,
        RetainedProof::AlreadyVerified,
    )
}

fn materialize_retained_exact_with_proof(
    source: crate::core::bounded_file::StableRegularFile,
    destination: &Path,
    repository: &str,
    bytes: u64,
    sha256: &str,
    proof: RetainedProof,
) -> Result<()> {
    let parent = destination
        .parent()
        .context("managed artifact has no parent")?;
    let destination_parent = crate::core::bounded_file::StableDirectory::create_and_open(parent)
        .context("retain managed artifact destination parent")?;
    let destination_name = destination
        .file_name()
        .context("managed artifact destination has no filename")?;
    materialize_retained_exact_at_with_proof(
        source,
        destination_parent,
        destination_name,
        repository,
        bytes,
        sha256,
        proof,
    )
}

pub(super) fn materialize_retained_exact_at(
    source: crate::core::bounded_file::StableRegularFile,
    destination_parent: crate::core::bounded_file::StableDirectory,
    destination_name: &std::ffi::OsStr,
    repository: &str,
    bytes: u64,
    sha256: &str,
) -> Result<()> {
    materialize_retained_exact_at_with_proof(
        source,
        destination_parent,
        destination_name,
        repository,
        bytes,
        sha256,
        RetainedProof::AlreadyVerified,
    )
}

pub(super) fn materialize_preverified_exact_at(
    source: &Path,
    destination_parent: crate::core::bounded_file::StableDirectory,
    destination_name: &std::ffi::OsStr,
    repository: &str,
    bytes: u64,
    sha256: &str,
) -> Result<()> {
    let canonical = source
        .canonicalize()
        .with_context(|| format!("resolve materialization source {}", source.display()))?;
    let retained = crate::core::bounded_file::StableRegularFile::open_exact(&canonical, bytes)?
        .context("materialization source is not the expected regular file")?;
    materialize_retained_exact_at_with_proof(
        retained,
        destination_parent,
        destination_name,
        repository,
        bytes,
        sha256,
        RetainedProof::VerifyStagedClone,
    )
}

fn materialize_retained_exact_at_with_proof(
    mut source: crate::core::bounded_file::StableRegularFile,
    destination_parent: crate::core::bounded_file::StableDirectory,
    destination_name: &std::ffi::OsStr,
    repository: &str,
    bytes: u64,
    sha256: &str,
    proof: RetainedProof,
) -> Result<()> {
    use rustix::fs::{AtFlags, Mode, OFlags, RenameFlags};

    if !source.is_stable()? || !destination_parent.is_current()? {
        bail!("materialization authority changed before retained adoption");
    }
    let canonical_destination = destination_parent.canonical_path().join(destination_name);
    if verify_or_refuse_existing_destination(&canonical_destination, bytes, sha256)? {
        return Ok(());
    }
    let parent_fd = destination_parent.try_clone()?;
    let staged_name =
        std::ffi::OsString::from(format!(".hf2q-adopt-{}.partial", uuid::Uuid::new_v4()));
    let mut guard = StagedAtGuard::new(parent_fd.try_clone()?, staged_name.clone());
    let (staged_file, cloned) = match source.clone_to_at(&parent_fd, &staged_name) {
        Ok(()) => {
            guard.arm();
            (
                fs::File::from(rustix::fs::openat(
                    &parent_fd,
                    &staged_name,
                    OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
                    Mode::empty(),
                )?),
                true,
            )
        }
        Err(error) if clone_requires_copy(&error) => {
            let _ = rustix::fs::unlinkat(&parent_fd, &staged_name, AtFlags::empty());
            check_hub_artifact_destination(repository, &canonical_destination, bytes)?;
            let mut output = fs::File::from(rustix::fs::openat(
                &parent_fd,
                &staged_name,
                OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::from_bits_truncate(0o600),
            )?);
            guard.arm();
            let (copied_bytes, copied_sha256) = source
                .copy_and_hash(&mut output)?
                .context("materialization source changed during retained copy")?;
            if copied_bytes != bytes || !copied_sha256.eq_ignore_ascii_case(sha256) {
                bail!("retained copy failed exact size/SHA-256 verification");
            }
            output.sync_all()?;
            (output, false)
        }
        Err(error) => {
            let _ = rustix::fs::unlinkat(&parent_fd, &staged_name, AtFlags::empty());
            return Err(error.into());
        }
    };
    let staged_path = destination_parent.canonical_path().join(&staged_name);
    let mut staged = crate::core::bounded_file::StableRegularFile::from_open_file(
        staged_file,
        &staged_path,
        bytes,
    )?
    .context("retained adoption did not produce the expected regular file")?;
    if cloned
        && matches!(proof, RetainedProof::VerifyStagedClone)
        && !staged
            .sha256()?
            .is_some_and(|digest| digest.eq_ignore_ascii_case(sha256))
    {
        bail!("staged clone failed exact SHA-256 verification");
    }
    if !source.is_stable()? || !staged.descriptor_is_stable()? {
        bail!("materialization source or staged clone changed before publication");
    }
    match rustix::fs::renameat_with(
        &parent_fd,
        &staged_name,
        &parent_fd,
        destination_name,
        RenameFlags::NOREPLACE,
    ) {
        Ok(()) => {}
        Err(error) if error == rustix::io::Errno::EXIST => {
            if verify_or_refuse_existing_destination(&canonical_destination, bytes, sha256)? {
                return Ok(());
            }
            return Err(std::io::Error::from_raw_os_error(error.raw_os_error()).into());
        }
        Err(error) => return Err(std::io::Error::from_raw_os_error(error.raw_os_error()).into()),
    }
    guard.disarm();
    let published = staged.current_descriptor_matches_path(&canonical_destination)?;
    let source_stable = source.is_stable()?;
    let public_parent_stable = destination_parent.is_current()?;
    if !published || !source_stable || !public_parent_stable {
        bail!("retained artifact publication changed before authority binding");
    }
    Ok(())
}

struct StagedAtGuard {
    directory: fs::File,
    name: std::ffi::OsString,
    armed: bool,
}

impl StagedAtGuard {
    fn new(directory: fs::File, name: std::ffi::OsString) -> Self {
        Self {
            directory,
            name,
            armed: false,
        }
    }

    fn arm(&mut self) {
        self.armed = true;
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for StagedAtGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = rustix::fs::unlinkat(&self.directory, &self.name, rustix::fs::AtFlags::empty());
        }
    }
}

pub(super) fn clone_requires_copy(error: &std::io::Error) -> bool {
    error
        .raw_os_error()
        .is_some_and(|code| [libc::EXDEV, libc::ENOTSUP, libc::EOPNOTSUPP].contains(&code))
}
