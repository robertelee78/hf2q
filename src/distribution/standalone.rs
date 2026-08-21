//! Small standalone install lifecycle for the native hf2q executable.
//!
//! The public updater authenticates exact downloaded bytes through Apple trust
//! before calling the local publisher. The hidden installer performs the same
//! checks in its generated shell entry point. This module owns the local lock,
//! channel marker, atomic executable replacement, one previous executable,
//! rollback, and uninstall.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use rustix::fs::FlockOperation;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const ACTIVE_NAME: &str = "hf2q";
const MARKER_NAME: &str = ".hf2q-standalone.json";
const PREVIOUS_NAME: &str = ".hf2q-previous";
const LOCK_NAME: &str = ".hf2q-standalone.lock";
const CANDIDATE_PARTIAL: &str = ".hf2q-candidate.partial";
const MARKER_PARTIAL: &str = ".hf2q-standalone.partial";
const PREVIOUS_PARTIAL: &str = ".hf2q-previous.partial";
const ROLLBACK_PARTIAL: &str = ".hf2q-rollback.partial";
const MARKER_KIND: &str = "hf2q.install-channel";
const MARKER_SCHEMA: u32 = 1;
const MAX_MARKER_BYTES: usize = 4 * 1024;
const MAX_BINARY_BYTES: u64 = 512 * 1024 * 1024;
const COPY_BUFFER_BYTES: usize = 64 * 1024;

#[derive(Debug, thiserror::Error)]
pub(crate) enum StandaloneError {
    #[error("another hf2q standalone install transition is running")]
    Busy,
    #[error("invalid standalone installation: {0}")]
    Invalid(&'static str),
    #[error("standalone candidate digest does not match the expected release bytes")]
    DigestMismatch,
    #[error("invalid standalone release record: {0}")]
    ReleaseRecord(&'static str),
    #[error("standalone update transport failed: {0}")]
    Network(String),
    #[error("standalone release trust check failed: {0}")]
    Trust(&'static str),
    #[error("standalone transition committed, but final durability is unknown: {0}")]
    CommittedDurabilityUnknown(String),
    #[error("standalone filesystem operation `{operation}` failed: {source}")]
    Io {
        operation: &'static str,
        #[source]
        source: std::io::Error,
    },
}

impl StandaloneError {
    fn io(operation: &'static str, source: std::io::Error) -> Self {
        Self::Io { operation, source }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StandaloneMarkerV1 {
    kind: String,
    schema_version: u32,
    package: String,
    channel: String,
}

impl StandaloneMarkerV1 {
    fn canonical() -> Self {
        Self {
            kind: MARKER_KIND.to_owned(),
            schema_version: MARKER_SCHEMA,
            package: "hf2q".to_owned(),
            channel: "standalone".to_owned(),
        }
    }

    fn bytes() -> Result<Vec<u8>, StandaloneError> {
        let mut bytes = serde_json::to_vec(&Self::canonical())
            .map_err(|_| StandaloneError::Invalid("channel marker could not be encoded"))?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    fn parse(bytes: &[u8]) -> Result<Self, StandaloneError> {
        if bytes.is_empty() || bytes.len() > MAX_MARKER_BYTES {
            return Err(StandaloneError::Invalid(
                "channel marker size is outside the supported bound",
            ));
        }
        let marker: Self = serde_json::from_slice(bytes)
            .map_err(|_| StandaloneError::Invalid("channel marker is not valid schema-1 JSON"))?;
        if marker != Self::canonical() || bytes != Self::bytes()? {
            return Err(StandaloneError::Invalid(
                "channel marker is not the canonical standalone marker",
            ));
        }
        Ok(marker)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CandidateExpectation {
    size: u64,
    sha256: [u8; 32],
}

impl CandidateExpectation {
    pub(crate) fn new(size: u64, sha256: [u8; 32]) -> Result<Self, StandaloneError> {
        if size == 0 || size > MAX_BINARY_BYTES {
            return Err(StandaloneError::Invalid(
                "candidate size is outside the supported bound",
            ));
        }
        Ok(Self { size, sha256 })
    }

    pub(crate) fn from_hex(size: u64, sha256: &str) -> Result<Self, StandaloneError> {
        if sha256.len() != 64
            || !sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        {
            return Err(StandaloneError::Invalid(
                "candidate SHA-256 must be 64 lowercase hexadecimal characters",
            ));
        }
        let mut digest = [0_u8; 32];
        hex::decode_to_slice(sha256, &mut digest).map_err(|_| {
            StandaloneError::Invalid(
                "candidate SHA-256 must be 64 lowercase hexadecimal characters",
            )
        })?;
        Self::new(size, digest)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PublishOutcome {
    Installed,
    Updated,
}

pub(crate) fn verify_running_installation(executable: &Path) -> Result<PathBuf, StandaloneError> {
    let executable = fs::canonicalize(executable)
        .map_err(|error| StandaloneError::io("canonicalize running executable", error))?;
    let install_directory = executable.parent().ok_or(StandaloneError::Invalid(
        "running executable has no standalone installation directory",
    ))?;
    let directory = InstallDirectory::open(install_directory)?;
    read_marker(&directory)?;
    require_owned_executable(&directory, ACTIVE_NAME)?;
    let active = fs::canonicalize(directory.child(ACTIVE_NAME))
        .map_err(|error| StandaloneError::io("canonicalize installed executable", error))?;
    if active != executable {
        return Err(StandaloneError::Invalid(
            "running executable is not the standalone-owned active hf2q binary",
        ));
    }
    directory.revalidate()?;
    Ok(install_directory.to_owned())
}

#[derive(Debug)]
struct InstallDirectory {
    path: PathBuf,
    handle: File,
    device: u64,
    inode: u64,
}

impl InstallDirectory {
    fn open(path: &Path) -> Result<Self, StandaloneError> {
        if !path.is_absolute() {
            return Err(StandaloneError::Invalid(
                "install directory must be an absolute path",
            ));
        }
        let canonical = fs::canonicalize(path)
            .map_err(|error| StandaloneError::io("canonicalize install directory", error))?;
        if canonical != path {
            return Err(StandaloneError::Invalid(
                "install directory must be canonical and contain no symlink component",
            ));
        }
        let handle = File::open(path)
            .map_err(|error| StandaloneError::io("open install directory", error))?;
        let metadata = handle
            .metadata()
            .map_err(|error| StandaloneError::io("inspect install directory", error))?;
        if !metadata.is_dir()
            || metadata.uid() != rustix::process::geteuid().as_raw()
            || metadata.mode() & 0o022 != 0
        {
            return Err(StandaloneError::Invalid(
                "install directory must be current-user-owned and not group/world-writable",
            ));
        }
        Ok(Self {
            path: path.to_owned(),
            handle,
            device: metadata.dev(),
            inode: metadata.ino(),
        })
    }

    fn child(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }

    fn revalidate(&self) -> Result<(), StandaloneError> {
        let metadata = fs::metadata(&self.path)
            .map_err(|error| StandaloneError::io("reopen install directory", error))?;
        if !metadata.is_dir()
            || metadata.dev() != self.device
            || metadata.ino() != self.inode
            || metadata.uid() != rustix::process::geteuid().as_raw()
            || metadata.mode() & 0o022 != 0
        {
            return Err(StandaloneError::Invalid(
                "install directory changed during the transition",
            ));
        }
        Ok(())
    }

    fn sync(&self) -> Result<(), StandaloneError> {
        sync_file(&self.handle, "sync install directory")
    }
}

struct InstallLock {
    file: File,
}

impl InstallLock {
    fn acquire(directory: &InstallDirectory) -> Result<Self, StandaloneError> {
        let path = directory.child(LOCK_NAME);
        let (file, created) = match open_new_private(&path, 0o600) {
            Ok(file) => (file, true),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                (open_existing_nofollow(&path, true)?, false)
            }
            Err(error) => return Err(StandaloneError::io("create install lock", error)),
        };
        if created {
            file.set_permissions(fs::Permissions::from_mode(0o600))
                .map_err(|error| StandaloneError::io("set install-lock mode", error))?;
            sync_file(&file, "sync install lock")?;
            directory.sync()?;
        }
        require_owned_regular(&file, directory, 0o600, Some(0), "install lock")?;
        rustix::fs::flock(&file, FlockOperation::NonBlockingLockExclusive).map_err(|error| {
            if error == rustix::io::Errno::WOULDBLOCK {
                StandaloneError::Busy
            } else {
                StandaloneError::io(
                    "acquire install lock",
                    std::io::Error::from_raw_os_error(error.raw_os_error()),
                )
            }
        })?;
        directory.revalidate()?;
        Ok(Self { file })
    }

    fn sync(&self) -> Result<(), StandaloneError> {
        sync_file(&self.file, "sync install-lock endpoint")
    }
}

pub(crate) fn publish_verified_candidate(
    install_directory: &Path,
    candidate: &Path,
    expectation: &CandidateExpectation,
) -> Result<PublishOutcome, StandaloneError> {
    let directory = InstallDirectory::open(install_directory)?;
    let lock = InstallLock::acquire(&directory)?;
    cleanup_exact_partials(&directory)?;
    let marker_exists = directory.child(MARKER_NAME).symlink_metadata().is_ok();
    let active_exists = directory.child(ACTIVE_NAME).symlink_metadata().is_ok();
    if active_exists && !marker_exists {
        return Err(StandaloneError::Invalid(
            "existing hf2q executable is not owned by the standalone channel",
        ));
    }
    copy_candidate(&directory, candidate, expectation, CANDIDATE_PARTIAL)?;
    if marker_exists {
        read_marker(&directory)?;
    } else {
        write_marker(&directory)?;
    }
    if active_exists {
        copy_owned_executable(&directory, ACTIVE_NAME, PREVIOUS_PARTIAL)?;
        fs::rename(
            directory.child(PREVIOUS_PARTIAL),
            directory.child(PREVIOUS_NAME),
        )
        .map_err(|error| StandaloneError::io("retain previous executable", error))?;
        directory.sync()?;
    }

    directory.revalidate()?;
    fs::rename(
        directory.child(CANDIDATE_PARTIAL),
        directory.child(ACTIVE_NAME),
    )
    .map_err(|error| StandaloneError::io("activate standalone executable", error))?;
    let committed = || -> Result<(), StandaloneError> {
        directory.sync()?;
        require_named_executable(&directory, ACTIVE_NAME, expectation)?;
        read_marker(&directory)?;
        lock.sync()?;
        directory.revalidate()?;
        Ok(())
    };
    committed().map_err(|error| StandaloneError::CommittedDurabilityUnknown(error.to_string()))?;
    Ok(if active_exists {
        PublishOutcome::Updated
    } else {
        PublishOutcome::Installed
    })
}

pub(crate) fn rollback(install_directory: &Path) -> Result<(), StandaloneError> {
    let directory = InstallDirectory::open(install_directory)?;
    let lock = InstallLock::acquire(&directory)?;
    cleanup_exact_partials(&directory)?;
    read_marker(&directory)?;
    require_owned_executable(&directory, ACTIVE_NAME)?;
    require_owned_executable(&directory, PREVIOUS_NAME)?;
    copy_owned_executable(&directory, ACTIVE_NAME, ROLLBACK_PARTIAL)?;
    fs::rename(directory.child(PREVIOUS_NAME), directory.child(ACTIVE_NAME))
        .map_err(|error| StandaloneError::io("activate retained executable", error))?;
    let committed = || -> Result<(), StandaloneError> {
        directory.sync()?;
        fs::rename(
            directory.child(ROLLBACK_PARTIAL),
            directory.child(PREVIOUS_NAME),
        )
        .map_err(|error| StandaloneError::io("retain replaced executable", error))?;
        directory.sync()?;
        require_owned_executable(&directory, ACTIVE_NAME)?;
        require_owned_executable(&directory, PREVIOUS_NAME)?;
        read_marker(&directory)?;
        lock.sync()?;
        directory.revalidate()?;
        Ok(())
    };
    committed().map_err(|error| StandaloneError::CommittedDurabilityUnknown(error.to_string()))
}

pub(crate) fn uninstall(install_directory: &Path) -> Result<(), StandaloneError> {
    let directory = InstallDirectory::open(install_directory)?;
    let lock = InstallLock::acquire(&directory)?;
    read_marker(&directory)?;
    remove_owned_executable_if_present(&directory, PREVIOUS_NAME)?;
    remove_owned_executable_if_present(&directory, ACTIVE_NAME)?;
    remove_marker(&directory)?;
    cleanup_exact_partials(&directory)?;
    directory.sync()?;
    lock.sync()?;
    drop(lock);
    remove_owned_lock(&directory)?;
    directory.sync()?;
    directory.revalidate()
}

fn write_marker(directory: &InstallDirectory) -> Result<(), StandaloneError> {
    let bytes = StandaloneMarkerV1::bytes()?;
    let path = directory.child(MARKER_PARTIAL);
    let mut file = open_new_private(&path, 0o600)
        .map_err(|error| StandaloneError::io("create channel marker", error))?;
    file.set_permissions(fs::Permissions::from_mode(0o600))
        .map_err(|error| StandaloneError::io("set channel-marker mode", error))?;
    file.write_all(&bytes)
        .map_err(|error| StandaloneError::io("write channel marker", error))?;
    sync_file(&file, "sync channel marker")?;
    require_owned_regular(
        &file,
        directory,
        0o600,
        Some(bytes.len() as u64),
        "channel marker",
    )?;
    fs::rename(path, directory.child(MARKER_NAME))
        .map_err(|error| StandaloneError::io("publish channel marker", error))?;
    directory.sync()?;
    read_marker(directory).map(|_| ())
}

fn read_marker(directory: &InstallDirectory) -> Result<StandaloneMarkerV1, StandaloneError> {
    let path = directory.child(MARKER_NAME);
    let mut file = open_existing_nofollow(&path, false)?;
    let metadata = require_owned_regular(&file, directory, 0o600, None, "channel marker")?;
    if metadata.len() == 0 || metadata.len() > MAX_MARKER_BYTES as u64 {
        return Err(StandaloneError::Invalid(
            "channel marker size is outside the supported bound",
        ));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.read_to_end(&mut bytes)
        .map_err(|error| StandaloneError::io("read channel marker", error))?;
    let after = file
        .metadata()
        .map_err(|error| StandaloneError::io("reinspect channel marker", error))?;
    if after.dev() != metadata.dev()
        || after.ino() != metadata.ino()
        || after.len() != metadata.len()
    {
        return Err(StandaloneError::Invalid(
            "channel marker changed while it was read",
        ));
    }
    StandaloneMarkerV1::parse(&bytes)
}

fn copy_candidate(
    directory: &InstallDirectory,
    source: &Path,
    expectation: &CandidateExpectation,
    partial_name: &str,
) -> Result<(), StandaloneError> {
    let mut source = open_existing_nofollow(source, false)?;
    let metadata = source
        .metadata()
        .map_err(|error| StandaloneError::io("inspect candidate executable", error))?;
    if !metadata.is_file() || metadata.len() != expectation.size || metadata.nlink() != 1 {
        return Err(StandaloneError::Invalid(
            "candidate executable is not one exact regular file",
        ));
    }
    let path = directory.child(partial_name);
    let mut destination = open_new_private(&path, 0o555)
        .map_err(|error| StandaloneError::io("create candidate partial", error))?;
    destination
        .set_permissions(fs::Permissions::from_mode(0o555))
        .map_err(|error| StandaloneError::io("set candidate mode", error))?;
    let digest = copy_and_hash(&mut source, &mut destination, expectation.size)?;
    if digest != expectation.sha256 {
        drop(destination);
        let _ = fs::remove_file(path);
        return Err(StandaloneError::DigestMismatch);
    }
    sync_file(&destination, "sync candidate executable")?;
    require_owned_regular(
        &destination,
        directory,
        0o555,
        Some(expectation.size),
        "candidate executable",
    )?;
    Ok(())
}

fn copy_owned_executable(
    directory: &InstallDirectory,
    source_name: &str,
    destination_name: &str,
) -> Result<(), StandaloneError> {
    let mut source = open_existing_nofollow(&directory.child(source_name), false)?;
    let metadata = require_owned_regular(&source, directory, 0o555, None, "installed executable")?;
    if metadata.len() == 0 || metadata.len() > MAX_BINARY_BYTES {
        return Err(StandaloneError::Invalid(
            "installed executable size is outside the supported bound",
        ));
    }
    let mut destination = open_new_private(&directory.child(destination_name), 0o555)
        .map_err(|error| StandaloneError::io("create retained executable", error))?;
    destination
        .set_permissions(fs::Permissions::from_mode(0o555))
        .map_err(|error| StandaloneError::io("set retained-executable mode", error))?;
    let mut remaining = metadata.len();
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    while remaining > 0 {
        let take = usize::try_from(remaining.min(COPY_BUFFER_BYTES as u64))
            .map_err(|_| StandaloneError::Invalid("executable size conversion failed"))?;
        source
            .read_exact(&mut buffer[..take])
            .map_err(|error| StandaloneError::io("read installed executable", error))?;
        destination
            .write_all(&buffer[..take])
            .map_err(|error| StandaloneError::io("write retained executable", error))?;
        remaining -= take as u64;
    }
    sync_file(&destination, "sync retained executable")?;
    require_owned_regular(
        &destination,
        directory,
        0o555,
        Some(metadata.len()),
        "retained executable",
    )?;
    Ok(())
}

fn copy_and_hash(
    source: &mut File,
    destination: &mut File,
    size: u64,
) -> Result<[u8; 32], StandaloneError> {
    source
        .seek(SeekFrom::Start(0))
        .map_err(|error| StandaloneError::io("seek candidate executable", error))?;
    let mut hasher = Sha256::new();
    let mut remaining = size;
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    while remaining > 0 {
        let take = usize::try_from(remaining.min(COPY_BUFFER_BYTES as u64))
            .map_err(|_| StandaloneError::Invalid("candidate size conversion failed"))?;
        source
            .read_exact(&mut buffer[..take])
            .map_err(|error| StandaloneError::io("read candidate executable", error))?;
        hasher.update(&buffer[..take]);
        destination
            .write_all(&buffer[..take])
            .map_err(|error| StandaloneError::io("write candidate executable", error))?;
        remaining -= take as u64;
    }
    let source_after = source
        .metadata()
        .map_err(|error| StandaloneError::io("reinspect candidate executable", error))?;
    if source_after.len() != size {
        return Err(StandaloneError::Invalid(
            "candidate executable changed while it was copied",
        ));
    }
    Ok(hasher.finalize().into())
}

fn require_named_executable(
    directory: &InstallDirectory,
    name: &str,
    expectation: &CandidateExpectation,
) -> Result<(), StandaloneError> {
    let mut file = open_existing_nofollow(&directory.child(name), false)?;
    let metadata = require_owned_regular(
        &file,
        directory,
        0o555,
        Some(expectation.size),
        "installed executable",
    )?;
    let digest = hash_exact(&mut file, expectation.size)?;
    if digest != expectation.sha256 || metadata.len() != expectation.size {
        return Err(StandaloneError::DigestMismatch);
    }
    Ok(())
}

fn hash_exact(source: &mut File, size: u64) -> Result<[u8; 32], StandaloneError> {
    source
        .seek(SeekFrom::Start(0))
        .map_err(|error| StandaloneError::io("seek installed executable", error))?;
    let mut hasher = Sha256::new();
    let mut remaining = size;
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    while remaining > 0 {
        let take = usize::try_from(remaining.min(COPY_BUFFER_BYTES as u64))
            .map_err(|_| StandaloneError::Invalid("executable size conversion failed"))?;
        source
            .read_exact(&mut buffer[..take])
            .map_err(|error| StandaloneError::io("read installed executable", error))?;
        hasher.update(&buffer[..take]);
        remaining -= take as u64;
    }
    Ok(hasher.finalize().into())
}

fn require_owned_executable(
    directory: &InstallDirectory,
    name: &str,
) -> Result<(), StandaloneError> {
    let file = open_existing_nofollow(&directory.child(name), false)?;
    let metadata = require_owned_regular(&file, directory, 0o555, None, "installed executable")?;
    if metadata.len() == 0 || metadata.len() > MAX_BINARY_BYTES {
        return Err(StandaloneError::Invalid(
            "installed executable size is outside the supported bound",
        ));
    }
    Ok(())
}

fn remove_owned_executable_if_present(
    directory: &InstallDirectory,
    name: &str,
) -> Result<(), StandaloneError> {
    let path = directory.child(name);
    match path.symlink_metadata() {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(StandaloneError::io("inspect executable", error)),
    }
    require_owned_executable(directory, name)?;
    fs::remove_file(path).map_err(|error| StandaloneError::io("remove executable", error))
}

fn remove_marker(directory: &InstallDirectory) -> Result<(), StandaloneError> {
    read_marker(directory)?;
    fs::remove_file(directory.child(MARKER_NAME))
        .map_err(|error| StandaloneError::io("remove channel marker", error))
}

fn remove_owned_lock(directory: &InstallDirectory) -> Result<(), StandaloneError> {
    let file = open_existing_nofollow(&directory.child(LOCK_NAME), true)?;
    require_owned_regular(&file, directory, 0o600, Some(0), "install lock")?;
    fs::remove_file(directory.child(LOCK_NAME))
        .map_err(|error| StandaloneError::io("remove install lock", error))
}

fn cleanup_exact_partials(directory: &InstallDirectory) -> Result<(), StandaloneError> {
    for name in [
        CANDIDATE_PARTIAL,
        MARKER_PARTIAL,
        PREVIOUS_PARTIAL,
        ROLLBACK_PARTIAL,
    ] {
        let path = directory.child(name);
        let metadata = match path.symlink_metadata() {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(StandaloneError::io("inspect partial file", error)),
        };
        if !metadata.is_file()
            || metadata.uid() != rustix::process::geteuid().as_raw()
            || metadata.nlink() != 1
            || metadata.dev() != directory.device
        {
            return Err(StandaloneError::Invalid(
                "reserved partial name contains an unowned filesystem node",
            ));
        }
        fs::remove_file(path)
            .map_err(|error| StandaloneError::io("remove stale partial file", error))?;
    }
    directory.sync()
}

#[path = "standalone/update.rs"]
mod update;
pub(crate) use update::{run_update, UpdateOutcome};

fn open_new_private(path: &Path, mode: u32) -> std::io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .mode(mode)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
}

fn open_existing_nofollow(path: &Path, writable: bool) -> Result<File, StandaloneError> {
    OpenOptions::new()
        .read(true)
        .write(writable)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|error| StandaloneError::io("open standalone-owned file", error))
}

fn require_owned_regular(
    file: &File,
    directory: &InstallDirectory,
    mode: u32,
    size: Option<u64>,
    what: &'static str,
) -> Result<fs::Metadata, StandaloneError> {
    let metadata = file
        .metadata()
        .map_err(|error| StandaloneError::io("inspect standalone-owned file", error))?;
    if !metadata.is_file()
        || metadata.uid() != rustix::process::geteuid().as_raw()
        || metadata.nlink() != 1
        || metadata.dev() != directory.device
        || metadata.mode() & 0o7777 != mode
        || size.is_some_and(|expected| metadata.len() != expected)
    {
        return Err(StandaloneError::Invalid(what));
    }
    Ok(metadata)
}

fn sync_file(file: &File, operation: &'static str) -> Result<(), StandaloneError> {
    #[cfg(target_os = "macos")]
    {
        rustix::fs::fcntl_fullfsync(file).map_err(|error| {
            StandaloneError::io(
                operation,
                std::io::Error::from_raw_os_error(error.raw_os_error()),
            )
        })
    }
    #[cfg(not(target_os = "macos"))]
    {
        file.sync_all()
            .map_err(|error| StandaloneError::io(operation, error))
    }
}

#[cfg(test)]
#[path = "standalone_tests.rs"]
mod tests;
