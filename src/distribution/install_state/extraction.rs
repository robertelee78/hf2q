use std::collections::BTreeMap;
use std::io::Read;

use sha2::{Digest, Sha256};

use super::locked::LockedInstallation;
use super::unix::{self, Directory, EntryIdentity};
use super::InstallStateError;
use crate::distribution::schema::{
    BundleFileV1, ReleaseManifestV1, Sha256Digest, MAX_BUNDLE_DIRECTORIES,
};
use crate::distribution::update_auth::ExtractionStageAuthorization;

mod io;

#[cfg(test)]
use io::abort_after_next_extraction_write;
use io::{reconstruct_exact, verify_file};

const EXTRACTIONS: &str = "extractions";
const MANIFEST_NAME: &str = "release-manifest.json";
const MAX_RETAINED_EXTRACTIONS: usize = 8;

#[derive(Debug, thiserror::Error)]
pub(in crate::distribution) enum ExtractionError {
    #[error("the extraction staging device has insufficient free space")]
    StorageFull,
    #[error("the private extraction tree does not match the authenticated release")]
    Integrity,
    #[error(transparent)]
    InstallState(InstallStateError),
    #[error("release extraction I/O failed")]
    Io(#[source] std::io::Error),
}

impl ExtractionError {
    fn write_io(error: std::io::Error) -> Self {
        match error.raw_os_error() {
            Some(code) if code == libc::ENOSPC || code == libc::EDQUOT => Self::StorageFull,
            _ => Self::Io(error),
        }
    }

    fn read_io(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<InstallStateError> for ExtractionError {
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

#[derive(Clone)]
struct ExpectedFile {
    path: String,
    size: u64,
    sha256: String,
}

#[derive(Default)]
struct ExpectedDirectory {
    children: BTreeMap<String, ExpectedNode>,
}

enum ExpectedNode {
    Directory(ExpectedDirectory),
    File(usize),
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum FileState {
    #[default]
    Absent,
    Partial(u64),
    Complete,
}

/// Lock-borrowing, exact-replay release extraction transaction.
pub(in crate::distribution) struct ReleaseExtractionStage<'lock> {
    locked: &'lock LockedInstallation,
    extractions: Directory,
    stage: Directory,
    stage_name: String,
    files: Vec<ExpectedFile>,
    directories: Vec<String>,
    states: Vec<FileState>,
    next: usize,
}

/// Inert private tree. It grants no path, file descriptor, publication, or
/// activation authority and remains mode 0600/0700 throughout this slice.
pub(in crate::distribution) struct ExtractedReleaseTree {
    _extractions: Directory,
    _stage: Directory,
    stage_name: String,
}

impl std::fmt::Debug for ReleaseExtractionStage<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReleaseExtractionStage")
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for ExtractedReleaseTree {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExtractedReleaseTree")
            .finish_non_exhaustive()
    }
}

pub(super) fn open_release_extraction<'lock>(
    locked: &'lock LockedInstallation,
    authorization: ExtractionStageAuthorization,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
) -> Result<ReleaseExtractionStage<'lock>, ExtractionError> {
    if manifest.version() != authorization.version()
        || manifest
            .to_deterministic_json()
            .map_err(|_| ExtractionError::Integrity)?
            != exact_manifest
        || manifest.derived_directories().len() > MAX_BUNDLE_DIRECTORIES
    {
        return Err(ExtractionError::Integrity);
    }
    let stage_name = authorization.stage_name();
    let extractions = unix::ensure_private_directory(locked.update(), EXTRACTIONS)?;
    validate_retained_stages(&extractions, &stage_name)?;
    let stage = unix::ensure_private_directory(&extractions, &stage_name)?;
    let files = expected_files(exact_manifest, manifest);
    let directories = manifest.derived_directories().to_vec();
    let tree = expected_tree(&directories, &files)?;
    let states = scan_tree(&stage, &tree, &files)?;
    validate_prefix(&states)?;
    Ok(ReleaseExtractionStage {
        locked,
        extractions,
        stage,
        stage_name,
        files,
        directories,
        states,
        next: 0,
    })
}

impl ReleaseExtractionStage<'_> {
    pub(in crate::distribution) fn resume_manifest(
        &mut self,
        source: &mut dyn Read,
    ) -> Result<(), ExtractionError> {
        self.resume_next(MANIFEST_NAME, source)
    }

    pub(in crate::distribution) fn resume_payload(
        &mut self,
        file: &BundleFileV1,
        source: &mut dyn Read,
    ) -> Result<(), ExtractionError> {
        let expected = self
            .files
            .get(self.next)
            .ok_or(ExtractionError::Integrity)?;
        if expected.path != file.path().as_str()
            || expected.size != file.size()
            || expected.sha256 != file.sha256().as_str()
        {
            return Err(ExtractionError::Integrity);
        }
        self.resume_next(file.path().as_str(), source)
    }

    fn resume_next(&mut self, path: &str, source: &mut dyn Read) -> Result<(), ExtractionError> {
        let expected = self
            .files
            .get(self.next)
            .filter(|expected| expected.path == path)
            .cloned()
            .ok_or(ExtractionError::Integrity)?;
        let (parent, name) = open_or_create_parent(&self.stage, path)?;
        let (file, identity) = match self.states[self.next] {
            FileState::Absent => unix::create_private_regular_file(&parent, name)?,
            FileState::Partial(_) | FileState::Complete => {
                unix::open_private_regular_file(&parent, name)?
            }
        };
        if identity.size > expected.size {
            return Err(ExtractionError::Integrity);
        }
        unix::verify_named_identity(&parent, name, identity)?;
        reconstruct_exact(&file, identity, source, &expected)?;
        unix::verify_named_identity(&parent, name, identity_with_size(identity, expected.size))?;
        self.states[self.next] = FileState::Complete;
        self.next += 1;
        Ok(())
    }

    pub(in crate::distribution) fn finish(self) -> Result<ExtractedReleaseTree, ExtractionError> {
        if self.next != self.files.len()
            || self
                .states
                .iter()
                .any(|state| *state != FileState::Complete)
        {
            return Err(ExtractionError::Integrity);
        }
        let tree = expected_tree(&self.directories, &self.files)?;
        let states = scan_tree(&self.stage, &tree, &self.files)?;
        if states.iter().any(|state| *state != FileState::Complete) {
            return Err(ExtractionError::Integrity);
        }
        for expected in &self.files {
            let (parent, name) = open_existing_parent(&self.stage, &expected.path)?;
            let (file, identity) = unix::open_private_regular_file(&parent, name)?;
            verify_file(&file, identity, expected)?;
            unix::verify_named_identity(&parent, name, identity)?;
            unix::full_sync_file(&file)?;
            unix::verify_named_identity(&parent, name, identity)?;
        }
        for path in self.directories.iter().rev() {
            let directory = open_existing_directory(&self.stage, path)?;
            unix::sync_directory(&directory)?;
        }
        unix::sync_directory(&self.stage)?;
        unix::sync_directory(&self.extractions)?;
        unix::sync_directory(self.locked.update())?;
        unix::sync_directory(self.locked.root())?;

        let live = self.locked.reopen()?;
        let live_extractions =
            unix::open_directory_at(&live.update, EXTRACTIONS, Some(0o700), true)?;
        if !live_extractions.same_object(&self.extractions) {
            return Err(ExtractionError::Integrity);
        }
        let live_stage =
            unix::open_directory_at(&live_extractions, &self.stage_name, Some(0o700), true)?;
        if !live_stage.same_object(&self.stage) {
            return Err(ExtractionError::Integrity);
        }
        let live_states = scan_tree(&live_stage, &tree, &self.files)?;
        if live_states
            .iter()
            .any(|state| *state != FileState::Complete)
        {
            return Err(ExtractionError::Integrity);
        }
        self.locked.full_sync_endpoint()?;
        Ok(ExtractedReleaseTree {
            _extractions: live_extractions,
            _stage: live_stage,
            stage_name: self.stage_name,
        })
    }
}

fn expected_files(exact_manifest: &[u8], manifest: &ReleaseManifestV1) -> Vec<ExpectedFile> {
    let mut files = Vec::with_capacity(manifest.files().len() + 1);
    files.push(ExpectedFile {
        path: MANIFEST_NAME.to_owned(),
        size: exact_manifest.len() as u64,
        sha256: hex::encode(Sha256::digest(exact_manifest)),
    });
    files.extend(manifest.files().iter().map(|file| ExpectedFile {
        path: file.path().as_str().to_owned(),
        size: file.size(),
        sha256: file.sha256().as_str().to_owned(),
    }));
    files
}

fn expected_tree(
    directories: &[String],
    files: &[ExpectedFile],
) -> Result<ExpectedDirectory, ExtractionError> {
    let mut root = ExpectedDirectory::default();
    for path in directories {
        let components: Vec<_> = path.split('/').collect();
        insert_directory(&mut root, &components)?;
    }
    for (index, file) in files.iter().enumerate() {
        let components: Vec<_> = file.path.split('/').collect();
        insert_file(&mut root, &components, index)?;
    }
    Ok(root)
}

fn insert_directory(
    directory: &mut ExpectedDirectory,
    components: &[&str],
) -> Result<(), ExtractionError> {
    let Some((component, remaining)) = components.split_first() else {
        return Ok(());
    };
    let node = directory
        .children
        .entry((*component).to_owned())
        .or_insert_with(|| ExpectedNode::Directory(ExpectedDirectory::default()));
    let ExpectedNode::Directory(child) = node else {
        return Err(ExtractionError::Integrity);
    };
    insert_directory(child, remaining)
}

fn insert_file(
    directory: &mut ExpectedDirectory,
    components: &[&str],
    index: usize,
) -> Result<(), ExtractionError> {
    let (component, remaining) = components.split_first().ok_or(ExtractionError::Integrity)?;
    if !remaining.is_empty() {
        let Some(ExpectedNode::Directory(child)) = directory.children.get_mut(*component) else {
            return Err(ExtractionError::Integrity);
        };
        return insert_file(child, remaining, index);
    }
    if directory
        .children
        .insert((*component).to_owned(), ExpectedNode::File(index))
        .is_some()
    {
        return Err(ExtractionError::Integrity);
    }
    Ok(())
}

fn scan_tree(
    root: &Directory,
    expected: &ExpectedDirectory,
    files: &[ExpectedFile],
) -> Result<Vec<FileState>, ExtractionError> {
    let mut states = vec![FileState::Absent; files.len()];
    scan_directory(root, expected, files, &mut states)?;
    Ok(states)
}

fn scan_directory(
    directory: &Directory,
    expected: &ExpectedDirectory,
    files: &[ExpectedFile],
    states: &mut [FileState],
) -> Result<(), ExtractionError> {
    let names = unix::list_names_bounded(directory, expected.children.len())?;
    for name in names {
        let node = expected
            .children
            .get(&name)
            .ok_or(ExtractionError::Integrity)?;
        match node {
            ExpectedNode::Directory(child) => {
                let opened = unix::open_directory_at(directory, &name, Some(0o700), true)?;
                scan_directory(&opened, child, files, states)?;
            }
            ExpectedNode::File(index) => {
                let (file, identity) = unix::open_private_regular_file(directory, &name)?;
                let expected = files.get(*index).ok_or(ExtractionError::Integrity)?;
                if identity.size > expected.size {
                    return Err(ExtractionError::Integrity);
                }
                states[*index] = if identity.size == expected.size {
                    match verify_file(&file, identity, expected) {
                        Ok(()) => FileState::Complete,
                        // A storage crash may preserve the length before every
                        // data block. This exact expected private file remains
                        // repairable from the authenticated archive stream.
                        Err(ExtractionError::Integrity) => FileState::Partial(identity.size),
                        Err(error) => return Err(error),
                    }
                } else {
                    FileState::Partial(identity.size)
                };
            }
        }
    }
    Ok(())
}

fn validate_prefix(states: &[FileState]) -> Result<(), ExtractionError> {
    let mut incomplete = false;
    for state in states {
        match state {
            FileState::Complete if incomplete => return Err(ExtractionError::Integrity),
            FileState::Complete => {}
            FileState::Partial(_) if incomplete => return Err(ExtractionError::Integrity),
            FileState::Partial(_) | FileState::Absent => {
                incomplete = true;
            }
        }
    }
    Ok(())
}

fn open_or_create_parent<'a>(
    root: &Directory,
    path: &'a str,
) -> Result<(Directory, &'a str), ExtractionError> {
    let (parent, name) = split_parent(path)?;
    let mut directory = unix::duplicate_directory(root)?;
    for component in parent.split('/').filter(|component| !component.is_empty()) {
        directory = unix::ensure_private_directory(&directory, component)?;
    }
    Ok((directory, name))
}

fn open_existing_parent<'a>(
    root: &Directory,
    path: &'a str,
) -> Result<(Directory, &'a str), ExtractionError> {
    let (parent, name) = split_parent(path)?;
    let directory = open_existing_directory(root, parent)?;
    Ok((directory, name))
}

fn open_existing_directory(root: &Directory, path: &str) -> Result<Directory, ExtractionError> {
    let mut directory = unix::duplicate_directory(root)?;
    for component in path.split('/').filter(|component| !component.is_empty()) {
        directory = unix::open_directory_at(&directory, component, Some(0o700), true)?;
    }
    Ok(directory)
}

fn split_parent(path: &str) -> Result<(&str, &str), ExtractionError> {
    match path.rsplit_once('/') {
        Some((parent, name)) if !parent.is_empty() && !name.is_empty() => Ok((parent, name)),
        None if !path.is_empty() => Ok(("", path)),
        _ => Err(ExtractionError::Integrity),
    }
}

fn validate_retained_stages(
    extractions: &Directory,
    requested: &str,
) -> Result<(), ExtractionError> {
    let names = unix::list_names_bounded(extractions, MAX_RETAINED_EXTRACTIONS + 1)?;
    if names.len() > MAX_RETAINED_EXTRACTIONS
        || (names.len() == MAX_RETAINED_EXTRACTIONS && !names.contains(requested))
    {
        return Err(ExtractionError::Integrity);
    }
    for name in names {
        validate_stage_name(&name)?;
        let _ = unix::open_directory_at(extractions, &name, Some(0o700), true)?;
    }
    Ok(())
}

fn validate_stage_name(name: &str) -> Result<(), ExtractionError> {
    let rest = name
        .strip_prefix(".extract-v")
        .ok_or(ExtractionError::Integrity)?;
    let (version, digest) = rest.rsplit_once('-').ok_or(ExtractionError::Integrity)?;
    crate::distribution::schema::ReleaseVersion::parse_stable("extraction.version", version.into())
        .map_err(|_| ExtractionError::Integrity)?;
    Sha256Digest::parse("extraction.archive_sha256", digest.into())
        .map_err(|_| ExtractionError::Integrity)?;
    Ok(())
}

fn identity_with_size(mut identity: EntryIdentity, size: u64) -> EntryIdentity {
    identity.size = size;
    identity
}

#[cfg(test)]
#[path = "extraction_tests.rs"]
mod tests;

#[cfg(test)]
impl ExtractedReleaseTree {
    pub(super) fn stage_name_for_test(&self) -> &str {
        &self.stage_name
    }
}
