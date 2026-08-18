use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;

use rustix::fs::{self, FileType, Mode, OFlags};
use sha2::{Digest, Sha256};
use uuid::{Uuid, Variant, Version};

use super::locked::LockedInstallation;
use super::metadata::MetadataStateAuthorization;
use super::unix::{self, EntryIdentity};
use super::InstallStateError;
use crate::distribution::schema::Sha256Digest;

const DOWNLOADS_DIRECTORY: &str = "downloads";
const STAGE_PREFIX: &str = ".artifact-";

#[derive(Debug, thiserror::Error)]
pub(in crate::distribution) enum ArtifactStageError {
    #[error("the update staging device has insufficient free space")]
    StorageFull,
    #[error("the streamed release archive does not match its authenticated descriptor")]
    Integrity,
    #[error(transparent)]
    InstallState(InstallStateError),
    #[error("release archive staging I/O failed")]
    Io(#[source] std::io::Error),
}

impl ArtifactStageError {
    fn io(error: std::io::Error) -> Self {
        match error.raw_os_error() {
            Some(code) if code == libc::ENOSPC || code == libc::EDQUOT => Self::StorageFull,
            _ => Self::Io(error),
        }
    }

    fn errno(error: rustix::io::Errno) -> Self {
        Self::io(std::io::Error::from_raw_os_error(error.raw_os_error()))
    }
}

impl From<InstallStateError> for ArtifactStageError {
    fn from(error: InstallStateError) -> Self {
        match &error {
            InstallStateError::Io { source, .. }
                if matches!(source.raw_os_error(), Some(libc::ENOSPC | libc::EDQUOT)) =>
            {
                Self::StorageFull
            }
            _ => Self::InstallState(error),
        }
    }
}

/// Same-process-only, unlinked archive staging.
///
/// The file has no pathname after construction. Dropping this value releases
/// every staged byte; it grants no extraction or installation authority.
pub(in crate::distribution) struct EphemeralArtifactStage {
    file: File,
    identity: EntryIdentity,
    expected_length: u64,
    written: u64,
    hasher: Sha256,
}

/// Exact length/SHA-256 proof over one private, unlinked file descriptor.
///
/// This wrapper implements only `Read` and `Seek`; distribution siblings
/// cannot append to or replace its authenticated bytes.
pub(in crate::distribution) struct VerifiedArchiveFile {
    file: File,
    identity: EntryIdentity,
    length: u64,
    sha256: Sha256Digest,
}

impl std::fmt::Debug for EphemeralArtifactStage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EphemeralArtifactStage")
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for VerifiedArchiveFile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VerifiedArchiveFile")
            .finish_non_exhaustive()
    }
}

pub(in crate::distribution) fn create_ephemeral_artifact_stage(
    authorization: &MetadataStateAuthorization,
    expected_length: u64,
) -> Result<EphemeralArtifactStage, ArtifactStageError> {
    if expected_length == 0 {
        return Err(ArtifactStageError::Integrity);
    }
    let locked = LockedInstallation::acquire(&authorization.root.path)?;
    create_ephemeral_artifact_stage_under_lock(&locked, expected_length)
}

pub(super) fn create_ephemeral_artifact_stage_under_lock(
    locked: &LockedInstallation,
    expected_length: u64,
) -> Result<EphemeralArtifactStage, ArtifactStageError> {
    if expected_length == 0
        || expected_length > crate::distribution::schema::MAX_RELEASE_ARCHIVE_BYTES
    {
        return Err(ArtifactStageError::Integrity);
    }
    let downloads = unix::ensure_private_directory(locked.update(), DOWNLOADS_DIRECTORY)?;
    recover_named_residue(&downloads)?;

    let name = format!("{STAGE_PREFIX}{}", Uuid::new_v4().simple());
    let fd = fs::openat(
        downloads.fd(),
        &name,
        OFlags::RDWR
            | OFlags::CREATE
            | OFlags::EXCL
            | OFlags::NOFOLLOW
            | OFlags::NONBLOCK
            | OFlags::CLOEXEC,
        Mode::from_raw_mode(0o600),
    )
    .map_err(ArtifactStageError::errno)?;
    let file = File::from(fd);
    unix::sync_directory(&downloads)?;

    let linked = fs::fstat(&file).map_err(ArtifactStageError::errno)?;
    unix::require_regular_policy(&linked, 0o600, downloads.device())?;
    let linked_identity = unix::identity(&linked);
    unix::remove_named_regular_file(&downloads, &name, linked_identity)?;
    unix::sync_directory(&downloads)?;

    let unlinked = fs::fstat(&file).map_err(ArtifactStageError::errno)?;
    let identity = require_unlinked_private_file(&unlinked, downloads.device())?;
    if identity.device != linked_identity.device || identity.inode != linked_identity.inode {
        return Err(InstallStateError::InvalidLayout(
            "artifact staging file changed while unlinking",
        )
        .into());
    }
    Ok(EphemeralArtifactStage {
        file,
        identity,
        expected_length,
        written: 0,
        hasher: Sha256::new(),
    })
}

impl EphemeralArtifactStage {
    pub(in crate::distribution) fn write_chunk(
        &mut self,
        bytes: &[u8],
    ) -> Result<(), ArtifactStageError> {
        let next = self
            .written
            .checked_add(bytes.len() as u64)
            .ok_or(ArtifactStageError::Integrity)?;
        if next > self.expected_length {
            return Err(ArtifactStageError::Integrity);
        }
        let mut offset = 0;
        while offset < bytes.len() {
            let count = self
                .file
                .write(&bytes[offset..])
                .map_err(ArtifactStageError::io)?;
            if count == 0 {
                return Err(ArtifactStageError::io(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "artifact stage made no write progress",
                )));
            }
            self.hasher.update(&bytes[offset..offset + count]);
            self.written += count as u64;
            offset += count;
        }
        Ok(())
    }

    pub(in crate::distribution) fn finish(
        mut self,
        expected_sha256: &Sha256Digest,
    ) -> Result<VerifiedArchiveFile, ArtifactStageError> {
        if self.written != self.expected_length
            || hex::encode(self.hasher.clone().finalize()) != expected_sha256.as_str()
        {
            return Err(ArtifactStageError::Integrity);
        }
        fs::fsync(&self.file).map_err(ArtifactStageError::errno)?;
        let after_write = fs::fstat(&self.file).map_err(ArtifactStageError::errno)?;
        let after_identity = require_unlinked_private_file(&after_write, self.identity.device)?;
        if after_identity.device != self.identity.device
            || after_identity.inode != self.identity.inode
            || after_identity.size != self.expected_length
        {
            return Err(ArtifactStageError::Integrity);
        }

        self.file
            .seek(SeekFrom::Start(0))
            .map_err(ArtifactStageError::io)?;
        let mut hasher = Sha256::new();
        let mut total = 0_u64;
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let count = self
                .file
                .read(&mut buffer)
                .map_err(ArtifactStageError::io)?;
            if count == 0 {
                break;
            }
            total = total
                .checked_add(count as u64)
                .ok_or(ArtifactStageError::Integrity)?;
            if total > self.expected_length {
                return Err(ArtifactStageError::Integrity);
            }
            hasher.update(&buffer[..count]);
        }
        let after_read = fs::fstat(&self.file).map_err(ArtifactStageError::errno)?;
        let reread_identity = require_unlinked_private_file(&after_read, self.identity.device)?;
        if reread_identity.device != self.identity.device
            || reread_identity.inode != self.identity.inode
            || reread_identity.size != self.expected_length
            || total != self.expected_length
            || hex::encode(hasher.finalize()) != expected_sha256.as_str()
        {
            return Err(ArtifactStageError::Integrity);
        }
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(ArtifactStageError::io)?;
        Ok(VerifiedArchiveFile {
            file: self.file,
            identity: reread_identity,
            length: self.expected_length,
            sha256: expected_sha256.clone(),
        })
    }
}

impl VerifiedArchiveFile {
    pub(in crate::distribution) fn length(&self) -> u64 {
        self.length
    }

    pub(in crate::distribution) fn sha256(&self) -> &Sha256Digest {
        &self.sha256
    }

    pub(in crate::distribution) fn revalidate(&self) -> Result<(), ArtifactStageError> {
        let before = fs::fstat(&self.file).map_err(ArtifactStageError::errno)?;
        let before_identity = require_unlinked_private_file(&before, self.identity.device)?;
        if before_identity.device != self.identity.device
            || before_identity.inode != self.identity.inode
            || before_identity.size != self.length
        {
            return Err(ArtifactStageError::Integrity);
        }
        let mut hasher = Sha256::new();
        let mut total = 0_u64;
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let count = self
                .file
                .read_at(&mut buffer, total)
                .map_err(ArtifactStageError::io)?;
            if count == 0 {
                break;
            }
            total = total
                .checked_add(count as u64)
                .ok_or(ArtifactStageError::Integrity)?;
            if total > self.length {
                return Err(ArtifactStageError::Integrity);
            }
            hasher.update(&buffer[..count]);
        }
        let after = fs::fstat(&self.file).map_err(ArtifactStageError::errno)?;
        let after_identity = require_unlinked_private_file(&after, self.identity.device)?;
        if after_identity != before_identity
            || total != self.length
            || hex::encode(hasher.finalize()) != self.sha256.as_str()
        {
            return Err(ArtifactStageError::Integrity);
        }
        Ok(())
    }
}

impl Read for VerifiedArchiveFile {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.file.read(buffer)
    }
}

impl Seek for VerifiedArchiveFile {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.file.seek(position)
    }
}

fn recover_named_residue(downloads: &unix::Directory) -> Result<(), ArtifactStageError> {
    let names = unix::list_names_bounded(downloads, 1)?;
    let Some(name) = names.into_iter().next() else {
        return Ok(());
    };
    let suffix = name
        .strip_prefix(STAGE_PREFIX)
        .ok_or(InstallStateError::InvalidLayout(
            "download staging contains an unexpected entry",
        ))?;
    let uuid = Uuid::parse_str(suffix).map_err(|_| {
        InstallStateError::InvalidLayout("download staging contains an invalid transaction name")
    })?;
    if suffix.len() != 32
        || uuid.simple().to_string() != suffix
        || uuid.get_version() != Some(Version::Random)
        || uuid.get_variant() != Variant::RFC4122
    {
        return Err(InstallStateError::InvalidLayout(
            "download staging contains an invalid transaction name",
        )
        .into());
    }
    let identity = unix::entry_identity(downloads, &name)?.ok_or(
        InstallStateError::InvalidLayout("download staging entry disappeared"),
    )?;
    if identity.file_type != FileType::RegularFile
        || identity.uid != rustix::process::geteuid().as_raw()
        || identity.mode != 0o600
        || identity.links != 1
        || identity.device != downloads.device()
        || identity.size != 0
    {
        return Err(InstallStateError::InvalidLayout(
            "download staging residue has unsafe identity",
        )
        .into());
    }
    unix::remove_named_regular_file(downloads, &name, identity)?;
    unix::sync_directory(downloads)?;
    Ok(())
}

fn require_unlinked_private_file(
    stat: &rustix::fs::Stat,
    expected_device: u64,
) -> Result<EntryIdentity, ArtifactStageError> {
    let identity = unix::identity(stat);
    if identity.file_type != FileType::RegularFile
        || identity.uid != rustix::process::geteuid().as_raw()
        || identity.mode != 0o600
        || identity.links != 0
        || identity.device != expected_device
    {
        return Err(InstallStateError::InvalidLayout(
            "unlinked artifact stage has unsafe identity",
        )
        .into());
    }
    Ok(identity)
}

#[cfg(test)]
mod tests;
