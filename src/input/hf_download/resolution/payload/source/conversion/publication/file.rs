use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use rustix::fs::{self, AtFlags, Mode, OFlags};

use super::{publication_error, publication_require, ModelPreparationPublicationError};
use unix::{
    entry_exists_at, full_sync, full_sync_named_file, named_file_identity, open_exact_directory,
    open_named_file, publication_as_io, remove_named_file, require_created_file,
    require_directory_rebound, require_private_stat, stat_identity, sync_directory, validate_name,
    verify_named_file, verify_named_identity, verify_open_file, Directory, PRIVATE_MODE,
};

mod unix;
pub(super) use unix::Identity;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PublicationBarrier {
    ParentOpened,
    PartialCreated,
    PartialPrefixVerified,
    PartialWritten,
    PartialFullSynced,
    PartialDirectorySynced,
    FinalLinked,
    LinkDirectorySynced,
    PartialUnlinked,
    UnlinkDirectorySynced,
    FinalFullSynced,
    AdoptedParentSynced,
    ParentRebound,
}

pub(super) fn directory_identity(
    path: &Path,
) -> Result<Identity, ModelPreparationPublicationError> {
    Ok(open_exact_directory(path, None)?.identity)
}

pub(super) fn publish_exact_private_file(
    parent: &Path,
    expected_parent: Identity,
    final_name: &str,
    partial_name: &str,
    expected: &[u8],
    cap: usize,
) -> Result<(), ModelPreparationPublicationError> {
    publish_exact_private_file_with(
        parent,
        expected_parent,
        final_name,
        partial_name,
        expected,
        cap,
        |_| Ok(()),
    )
}

fn publish_exact_private_file_with(
    parent_path: &Path,
    expected_parent: Identity,
    final_name: &str,
    partial_name: &str,
    expected: &[u8],
    cap: usize,
    mut barrier: impl FnMut(PublicationBarrier) -> Result<(), ModelPreparationPublicationError>,
) -> Result<(), ModelPreparationPublicationError> {
    publication_require(
        !expected.is_empty() && expected.len() <= cap,
        "published model record is empty or exceeds its cap",
    )?;
    validate_name(final_name)?;
    validate_name(partial_name)?;
    let parent = open_exact_directory(parent_path, Some(expected_parent))?;
    barrier(PublicationBarrier::ParentOpened)?;

    if entry_exists_at(&parent, final_name)? {
        finish_existing_publication(&parent, final_name, partial_name, expected, &mut barrier)?;
        require_directory_rebound(parent_path, expected_parent)?;
        barrier(PublicationBarrier::ParentRebound)?;
        return Ok(());
    }

    let partial_identity = if entry_exists_at(&parent, partial_name)? {
        resume_partial(&parent, partial_name, expected, &mut barrier)?
    } else {
        create_partial(&parent, partial_name, expected, &mut barrier)?
    };
    verify_named_file(&parent, partial_name, partial_identity, expected, 1)?;
    fs::linkat(
        &parent.fd,
        partial_name,
        &parent.fd,
        final_name,
        AtFlags::empty(),
    )
    .map_err(std::io::Error::from)?;
    barrier(PublicationBarrier::FinalLinked)?;
    sync_directory(&parent)?;
    barrier(PublicationBarrier::LinkDirectorySynced)?;
    verify_named_file(&parent, partial_name, partial_identity, expected, 2)?;
    verify_named_file(&parent, final_name, partial_identity, expected, 2)?;
    remove_named_file(&parent, partial_name, partial_identity)?;
    barrier(PublicationBarrier::PartialUnlinked)?;
    sync_directory(&parent)?;
    barrier(PublicationBarrier::UnlinkDirectorySynced)?;
    verify_named_file(&parent, final_name, partial_identity, expected, 1)?;
    full_sync_named_file(&parent, final_name, partial_identity, expected)?;
    barrier(PublicationBarrier::FinalFullSynced)?;
    require_directory_rebound(parent_path, expected_parent)?;
    barrier(PublicationBarrier::ParentRebound)
}

pub(super) fn read_exact_private_file(
    parent_path: &Path,
    expected_parent: Identity,
    name: &str,
    expected: &[u8],
    cap: usize,
) -> Result<Vec<u8>, ModelPreparationPublicationError> {
    publication_require(
        !expected.is_empty() && expected.len() <= cap,
        "model record is empty or exceeds its cap",
    )?;
    validate_name(name)?;
    let parent = open_exact_directory(parent_path, Some(expected_parent))?;
    let identity = named_file_identity(&parent, name, 1, Some(expected.len() as u64))?;
    verify_named_file(&parent, name, identity, expected, 1)?;
    require_directory_rebound(parent_path, expected_parent)?;
    Ok(expected.to_vec())
}

pub(super) fn read_bounded_owned_file(
    parent_path: &Path,
    expected_parent: Identity,
    name: &str,
    cap: usize,
) -> Result<Vec<u8>, ModelPreparationPublicationError> {
    validate_name(name)?;
    let parent = open_exact_directory(parent_path, Some(expected_parent))?;
    let identity = named_file_identity(&parent, name, 1, None)?;
    let mut file = open_named_file(&parent, name, identity, true)?;
    let metadata = file.metadata()?;
    publication_require(metadata.len() <= cap as u64, "model record exceeds its cap")?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    (&mut file).take(cap as u64 + 1).read_to_end(&mut bytes)?;
    publication_require(bytes.len() <= cap, "model record exceeds its cap")?;
    let after = stat_identity(&fs::fstat(&file).map_err(std::io::Error::from)?);
    publication_require(after == identity, "model record changed while reading")?;
    verify_named_identity(&parent, name, identity)?;
    require_directory_rebound(parent_path, expected_parent)?;
    Ok(bytes)
}

pub(super) fn entry_exists(path: &Path) -> Result<bool, std::io::Error> {
    let parent_path = path
        .parent()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing parent"))?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid record name")
        })?;
    let parent = open_exact_directory(parent_path, None).map_err(publication_as_io)?;
    entry_exists_at(&parent, name)
}

fn finish_existing_publication(
    parent: &Directory,
    final_name: &str,
    partial_name: &str,
    expected: &[u8],
    barrier: &mut impl FnMut(PublicationBarrier) -> Result<(), ModelPreparationPublicationError>,
) -> Result<(), ModelPreparationPublicationError> {
    if entry_exists_at(parent, partial_name)? {
        let partial_identity =
            named_file_identity(parent, partial_name, 2, Some(expected.len() as u64))?;
        let final_identity =
            named_file_identity(parent, final_name, 2, Some(expected.len() as u64))?;
        publication_require(
            partial_identity == final_identity,
            "published record and residue are not the same exact inode",
        )?;
        verify_named_file(parent, final_name, final_identity, expected, 2)?;
        verify_named_file(parent, partial_name, partial_identity, expected, 2)?;
        remove_named_file(parent, partial_name, partial_identity)?;
        barrier(PublicationBarrier::PartialUnlinked)?;
        sync_directory(parent)?;
        barrier(PublicationBarrier::UnlinkDirectorySynced)?;
        verify_named_file(parent, final_name, final_identity, expected, 1)?;
        full_sync_named_file(parent, final_name, final_identity, expected)?;
        barrier(PublicationBarrier::FinalFullSynced)?;
    } else {
        let final_identity =
            named_file_identity(parent, final_name, 1, Some(expected.len() as u64))?;
        verify_named_file(parent, final_name, final_identity, expected, 1)?;
        full_sync_named_file(parent, final_name, final_identity, expected)?;
        barrier(PublicationBarrier::FinalFullSynced)?;
        sync_directory(parent)?;
        barrier(PublicationBarrier::AdoptedParentSynced)?;
        verify_named_file(parent, final_name, final_identity, expected, 1)?;
        full_sync_named_file(parent, final_name, final_identity, expected)?;
        barrier(PublicationBarrier::FinalFullSynced)?;
    }
    Ok(())
}

fn create_partial(
    parent: &Directory,
    name: &str,
    expected: &[u8],
    barrier: &mut impl FnMut(PublicationBarrier) -> Result<(), ModelPreparationPublicationError>,
) -> Result<Identity, ModelPreparationPublicationError> {
    let fd = fs::openat(
        &parent.fd,
        name,
        OFlags::RDWR
            | OFlags::CREATE
            | OFlags::EXCL
            | OFlags::NOFOLLOW
            | OFlags::NONBLOCK
            | OFlags::CLOEXEC,
        Mode::from_raw_mode(PRIVATE_MODE),
    )
    .map_err(std::io::Error::from)?;
    let mut file = File::from(fd);
    let created = fs::fstat(&file).map_err(std::io::Error::from)?;
    require_created_file(&created, parent.identity.device())?;
    fs::fchmod(&file, Mode::from_raw_mode(PRIVATE_MODE)).map_err(std::io::Error::from)?;
    let identity = stat_identity(&created);
    require_private_stat(
        &fs::fstat(&file).map_err(std::io::Error::from)?,
        parent.identity.device(),
        1,
    )?;
    verify_named_identity(parent, name, identity)?;
    barrier(PublicationBarrier::PartialCreated)?;
    file.write_all(expected)?;
    barrier(PublicationBarrier::PartialWritten)?;
    full_sync(&file)?;
    barrier(PublicationBarrier::PartialFullSynced)?;
    verify_open_file(&mut file, identity, expected, 1, parent.identity.device())?;
    sync_directory(parent)?;
    barrier(PublicationBarrier::PartialDirectorySynced)?;
    verify_named_file(parent, name, identity, expected, 1)?;
    Ok(identity)
}

fn resume_partial(
    parent: &Directory,
    name: &str,
    expected: &[u8],
    barrier: &mut impl FnMut(PublicationBarrier) -> Result<(), ModelPreparationPublicationError>,
) -> Result<Identity, ModelPreparationPublicationError> {
    let identity = named_file_identity(parent, name, 1, None)?;
    let mut file = open_named_file(parent, name, identity, false)?;
    let before = file.metadata()?;
    publication_require(
        before.len() <= expected.len() as u64,
        "partial model record is longer than expected",
    )?;
    let existing_len = usize::try_from(before.len())
        .map_err(|_| publication_error("partial model record length does not fit usize"))?;
    let mut existing = vec![0u8; existing_len];
    file.read_exact(&mut existing)?;
    publication_require(
        existing == expected[..existing_len],
        "partial model record is not an exact prefix",
    )?;
    barrier(PublicationBarrier::PartialPrefixVerified)?;
    let current = fs::fstat(&file).map_err(std::io::Error::from)?;
    require_private_stat(&current, parent.identity.device(), 1)?;
    publication_require(
        stat_identity(&current) == identity && current.st_size as u64 == before.len(),
        "partial model record changed after its prefix was measured",
    )?;
    verify_named_identity(parent, name, identity)?;
    file.seek(SeekFrom::Start(existing_len as u64))?;
    file.write_all(&expected[existing_len..])?;
    barrier(PublicationBarrier::PartialWritten)?;
    full_sync(&file)?;
    barrier(PublicationBarrier::PartialFullSynced)?;
    verify_open_file(&mut file, identity, expected, 1, parent.identity.device())?;
    verify_named_file(parent, name, identity, expected, 1)?;
    Ok(identity)
}

#[cfg(test)]
mod tests;
