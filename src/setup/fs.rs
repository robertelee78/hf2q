use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::AsFd;
use std::path::{Path, PathBuf};

use rustix::fs::{self, AtFlags};

use self::unix::{
    acquire_lock, create_private_file, ensure_directory, entry_identity, full_sync, full_sync_lock,
    io_error, open_existing_directory, open_or_create_root, open_private_file, private_identity,
    reopen_root, sync_directory, verify_directory, verify_lock, verify_named, verify_root,
    Directory, Identity,
};
use super::schema::{ConfigV1, MAX_CONFIG_BYTES};
use super::SetupError;

mod unix;

const CONFIG_NAME: &str = "config.toml";
const PARTIAL_NAME: &str = ".config.toml.partial";
const LOCK_NAME: &str = ".config.toml.lock";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SetupBarrier {
    RootOpened,
    LockAcquired,
    PartialPrefixVerified,
    PartialSynced,
    BeforeRename,
    ConfigRenamed,
    RootSynced,
    ConfigFullSynced,
    SessionDirectoriesSynced,
    EndpointFullSynced,
}

impl SetupBarrier {
    #[cfg(test)]
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::RootOpened => "root-opened",
            Self::LockAcquired => "lock-acquired",
            Self::PartialPrefixVerified => "partial-prefix-verified",
            Self::PartialSynced => "partial-synced",
            Self::BeforeRename => "before-rename",
            Self::ConfigRenamed => "config-renamed",
            Self::RootSynced => "root-synced",
            Self::ConfigFullSynced => "config-full-synced",
            Self::SessionDirectoriesSynced => "session-directories-synced",
            Self::EndpointFullSynced => "endpoint-full-synced",
        }
    }
}

pub(super) struct ExistingConfig {
    root: Option<Directory>,
    config: Option<ConfigV1>,
    bytes: Option<Vec<u8>>,
    identity: Option<Identity>,
}

impl ExistingConfig {
    pub(super) fn config(&self) -> Option<&ConfigV1> {
        self.config.as_ref()
    }

    fn matches(&self, other: &Self) -> bool {
        self.config == other.config && self.bytes == other.bytes && self.identity == other.identity
    }
}

struct SessionDirectories {
    cache: Directory,
    sessions: Directory,
}

#[allow(dead_code)]
pub(super) struct RuntimeConfigBinding {
    root_path: PathBuf,
    root: Directory,
    config_bytes: Vec<u8>,
    config_identity: Identity,
    directories: SessionDirectories,
    state_binding: crate::distribution::SetupStateRootBinding,
}

#[allow(dead_code)]
impl RuntimeConfigBinding {
    pub(super) fn revalidate(&self) -> Result<(), SetupError> {
        self.state_binding
            .revalidate()
            .map_err(|error| SetupError::Filesystem(error.to_string()))?;
        verify_root(&self.root_path, &self.root)?;
        let observed =
            read_optional_snapshot_with_hook(&self.root, CONFIG_NAME, MAX_CONFIG_BYTES, || {})?
                .ok_or(SetupError::Missing)?;
        if observed.0 != self.config_bytes || observed.1 != self.config_identity {
            return Err(SetupError::Filesystem(
                "runtime config.toml changed after authorization".to_owned(),
            ));
        }
        verify_session_directories(&self.root, &self.directories)?;
        verify_root(&self.root_path, &self.root)?;
        self.state_binding
            .revalidate()
            .map_err(|error| SetupError::Filesystem(error.to_string()))
    }

    pub(in crate::setup) fn session_directory_fd(&self) -> std::os::fd::BorrowedFd<'_> {
        self.directories.sessions.fd.as_fd()
    }

    #[cfg(test)]
    pub(super) fn retained_regular_files_are_read_only_for_test(&self) -> Result<bool, SetupError> {
        self.state_binding
            .retained_regular_files_are_read_only_for_test()
            .map_err(|error| SetupError::Filesystem(error.to_string()))
    }
}

#[allow(dead_code)]
pub(super) fn authorize_runtime_config(
    root_path: &Path,
) -> Result<Option<(ConfigV1, RuntimeConfigBinding)>, SetupError> {
    let state_binding = crate::distribution::verify_setup_state_root(root_path)
        .map_err(|error| SetupError::Filesystem(error.to_string()))?;
    let root = match reopen_root(root_path) {
        Ok(root) => root,
        Err(SetupError::Missing) => {
            state_binding
                .revalidate()
                .map_err(|error| SetupError::Filesystem(error.to_string()))?;
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let observed =
        match read_optional_snapshot_with_hook(&root, CONFIG_NAME, MAX_CONFIG_BYTES, || {})? {
            Some(observed) => observed,
            None => {
                verify_root(root_path, &root)?;
                state_binding
                    .revalidate()
                    .map_err(|error| SetupError::Filesystem(error.to_string()))?;
                return Ok(None);
            }
        };
    let config = ConfigV1::parse(&observed.0)?;
    let cache = open_existing_directory(&root, "cache")?;
    let sessions = open_existing_directory(&cache, "sessions")?;
    let binding = RuntimeConfigBinding {
        root_path: root_path.to_owned(),
        root,
        config_bytes: observed.0,
        config_identity: observed.1,
        directories: SessionDirectories { cache, sessions },
        state_binding,
    };
    binding.revalidate()?;
    Ok(Some((config, binding)))
}

pub(super) fn observe_existing_config(root_path: &Path) -> Result<ExistingConfig, SetupError> {
    observe_existing_config_with_hook(root_path, || {})
}

fn observe_existing_config_with_hook(
    root_path: &Path,
    hook: impl FnOnce(),
) -> Result<ExistingConfig, SetupError> {
    let root = match reopen_root(root_path) {
        Ok(root) => root,
        Err(SetupError::Missing) => {
            return Ok(ExistingConfig {
                root: None,
                config: None,
                bytes: None,
                identity: None,
            });
        }
        Err(error) => return Err(error),
    };
    let observed = read_optional_snapshot_with_hook(&root, CONFIG_NAME, MAX_CONFIG_BYTES, hook)?;
    let (bytes, identity) = match observed {
        Some((bytes, identity)) => (Some(bytes), Some(identity)),
        None => (None, None),
    };
    let config = bytes.as_deref().map(ConfigV1::parse).transpose()?;
    verify_root(root_path, &root)?;
    Ok(ExistingConfig {
        root: Some(root),
        config,
        bytes,
        identity,
    })
}

#[cfg(test)]
pub(super) fn read_existing_config_with_test_hook(
    root_path: &Path,
    hook: impl FnOnce(),
) -> Result<Option<ConfigV1>, SetupError> {
    Ok(observe_existing_config_with_hook(root_path, hook)?.config)
}

pub(super) fn persist(
    root_path: &Path,
    config: &ConfigV1,
    expected: &[u8],
    observed: &ExistingConfig,
    state_binding: &crate::distribution::SetupStateRootBinding,
) -> Result<bool, SetupError> {
    persist_with_hook(
        root_path,
        config,
        expected,
        observed,
        state_binding,
        |barrier| {
            abort_at_test_barrier(barrier);
            Ok(())
        },
    )
}

fn persist_with_hook(
    root_path: &Path,
    config: &ConfigV1,
    expected: &[u8],
    observed: &ExistingConfig,
    state_binding: &crate::distribution::SetupStateRootBinding,
    mut hook: impl FnMut(SetupBarrier) -> Result<(), SetupError>,
) -> Result<bool, SetupError> {
    let (root, root_created) = if observed.root.is_some() {
        (reopen_root(root_path)?, false)
    } else {
        open_or_create_root(root_path)?
    };
    match &observed.root {
        Some(expected) if !root_created && root.same_object(expected) => {}
        None if root_created => {}
        _ => {
            return Err(SetupError::Filesystem(
                "state root changed after setup displayed its policy defaults".to_owned(),
            ));
        }
    }
    hook(SetupBarrier::RootOpened)?;
    let lock = acquire_lock(&root, LOCK_NAME)?;
    hook(SetupBarrier::LockAcquired)?;
    let live = reopen_root(root_path)?;
    if !live.same_object(&root) {
        return Err(SetupError::Filesystem(
            "state root changed while acquiring the setup lock".to_owned(),
        ));
    }
    verify_lock(&root, LOCK_NAME, &lock)?;
    state_binding
        .revalidate()
        .map_err(|error| SetupError::Filesystem(error.to_string()))?;

    let live_observed = observe_existing_config_from_root(&root)?;
    if !observed.matches(&live_observed) {
        return Err(SetupError::Filesystem(
            "config.toml changed after setup displayed its policy defaults".to_owned(),
        ));
    }
    if let Some(bytes) = live_observed.bytes.as_deref() {
        if bytes == expected {
            remove_partial_if_private(&root)?;
            let directories = ensure_session_directories(&root)?;
            hook(SetupBarrier::SessionDirectoriesSynced)?;
            verify_root(root_path, &root)?;
            verify_committed_config(&root, config, expected, live_observed.identity)?;
            verify_session_directories(&root, &directories)?;
            verify_lock(&root, LOCK_NAME, &lock)?;
            full_sync_lock(&lock)?;
            hook(SetupBarrier::EndpointFullSynced)?;
            verify_root(root_path, &root)?;
            state_binding
                .revalidate()
                .map_err(|error| SetupError::Filesystem(error.to_string()))?;
            verify_committed_config(&root, config, expected, live_observed.identity)?;
            verify_session_directories(&root, &directories)?;
            verify_lock(&root, LOCK_NAME, &lock)?;
            return Ok(false);
        }
    }

    verify_lock(&root, LOCK_NAME, &lock)?;
    let partial_identity = write_partial(&root, expected, || {
        hook(SetupBarrier::PartialPrefixVerified)?;
        verify_lock(&root, LOCK_NAME, &lock)
    })?;
    hook(SetupBarrier::PartialSynced)?;
    let before = entry_identity(&root, CONFIG_NAME)?;
    if before != live_observed.identity {
        return Err(SetupError::Filesystem(
            "config.toml changed while the setup partial was prepared".to_owned(),
        ));
    }
    verify_root(root_path, &root)?;
    if let Some(identity) = before {
        verify_named(&root, CONFIG_NAME, identity)?;
    }
    hook(SetupBarrier::BeforeRename)?;
    verify_root(root_path, &root)?;
    if entry_identity(&root, CONFIG_NAME)? != before {
        return Err(SetupError::Filesystem(
            "config.toml changed immediately before replacement".to_owned(),
        ));
    }
    verify_lock(&root, LOCK_NAME, &lock)?;
    state_binding
        .revalidate()
        .map_err(|error| SetupError::Filesystem(error.to_string()))?;
    verify_named(&root, PARTIAL_NAME, partial_identity)?;
    fs::renameat(root.fd.as_fd(), PARTIAL_NAME, root.fd.as_fd(), CONFIG_NAME)
        .map_err(|error| io_error("atomically replace config.toml", error))?;

    let committed = (|| {
        verify_named(&root, CONFIG_NAME, partial_identity)?;
        hook(SetupBarrier::ConfigRenamed)?;
        sync_directory(&root)?;
        hook(SetupBarrier::RootSynced)?;
        verify_root(root_path, &root)?;
        verify_committed_config(&root, config, expected, Some(partial_identity))?;
        hook(SetupBarrier::ConfigFullSynced)?;
        let directories = ensure_session_directories(&root)?;
        hook(SetupBarrier::SessionDirectoriesSynced)?;
        sync_directory(&root)?;
        verify_root(root_path, &root)?;
        state_binding
            .revalidate()
            .map_err(|error| SetupError::Filesystem(error.to_string()))?;
        verify_committed_config(&root, config, expected, Some(partial_identity))?;
        verify_session_directories(&root, &directories)?;
        verify_lock(&root, LOCK_NAME, &lock)?;
        full_sync_lock(&lock)?;
        hook(SetupBarrier::EndpointFullSynced)?;
        verify_root(root_path, &root)?;
        state_binding
            .revalidate()
            .map_err(|error| SetupError::Filesystem(error.to_string()))?;
        verify_committed_config(&root, config, expected, Some(partial_identity))?;
        verify_session_directories(&root, &directories)?;
        verify_lock(&root, LOCK_NAME, &lock)
    })();
    committed.map_err(|error| SetupError::DurabilityUnknown(error.to_string()))?;
    Ok(true)
}

#[cfg(test)]
pub(super) fn persist_with_test_hook(
    root_path: &Path,
    config: &ConfigV1,
    expected: &[u8],
    hook: impl FnMut(SetupBarrier) -> Result<(), SetupError>,
) -> Result<bool, SetupError> {
    let observed = observe_existing_config(root_path)?;
    let state_binding = crate::distribution::verify_setup_state_root(root_path)
        .map_err(|error| SetupError::Filesystem(error.to_string()))?;
    persist_with_hook(root_path, config, expected, &observed, &state_binding, hook)
}

#[cfg(test)]
pub(super) fn persist_observed_with_test_hook(
    root_path: &Path,
    config: &ConfigV1,
    expected: &[u8],
    observed: &ExistingConfig,
    hook: impl FnMut(SetupBarrier) -> Result<(), SetupError>,
) -> Result<bool, SetupError> {
    let state_binding = crate::distribution::verify_setup_state_root(root_path)
        .map_err(|error| SetupError::Filesystem(error.to_string()))?;
    persist_with_hook(root_path, config, expected, observed, &state_binding, hook)
}

#[cfg(test)]
fn abort_at_test_barrier(barrier: SetupBarrier) {
    super::tests::abort_at_setup_barrier(barrier);
}

#[cfg(not(test))]
fn abort_at_test_barrier(_barrier: SetupBarrier) {}

fn ensure_session_directories(root: &Directory) -> Result<SessionDirectories, SetupError> {
    let cache = ensure_directory(root, "cache")?;
    let sessions = ensure_directory(&cache, "sessions")?;
    sync_directory(&sessions)?;
    sync_directory(&cache)?;
    sync_directory(root)?;
    let directories = SessionDirectories { cache, sessions };
    verify_session_directories(root, &directories)?;
    Ok(directories)
}

fn verify_session_directories(
    root: &Directory,
    directories: &SessionDirectories,
) -> Result<(), SetupError> {
    verify_directory(&directories.cache, "sessions", &directories.sessions)?;
    verify_directory(root, "cache", &directories.cache)
}

fn verify_committed_config(
    root: &Directory,
    config: &ConfigV1,
    expected: &[u8],
    expected_identity: Option<Identity>,
) -> Result<(), SetupError> {
    let bytes = read_required(root, CONFIG_NAME, MAX_CONFIG_BYTES)?;
    if bytes != expected || ConfigV1::parse(&bytes)? != *config {
        return Err(SetupError::Filesystem(
            "committed config.toml does not match the requested config".to_owned(),
        ));
    }
    let file = open_private_file(root, CONFIG_NAME, true)?;
    full_sync(&file)?;
    let identity = private_identity(&file, root)?;
    if expected_identity.is_some_and(|expected| expected != identity) {
        return Err(SetupError::Filesystem(
            "committed config.toml is not the renamed setup partial".to_owned(),
        ));
    }
    verify_named(root, CONFIG_NAME, identity)
}

fn write_partial(
    root: &Directory,
    expected: &[u8],
    hook: impl FnOnce() -> Result<(), SetupError>,
) -> Result<Identity, SetupError> {
    let mut file = match open_private_file(root, PARTIAL_NAME, true) {
        Ok(file) => file,
        Err(SetupError::Missing) => create_private_file(root, PARTIAL_NAME)?,
        Err(error) => return Err(error),
    };
    let mut current = read_open_file(&mut file, MAX_CONFIG_BYTES)?;
    let measured = private_identity(&file, root)?;
    if measured.size != current.len() as u64 {
        return Err(SetupError::Filesystem(
            "setup partial changed while its prefix was measured".to_owned(),
        ));
    }
    verify_named(root, PARTIAL_NAME, measured)?;
    hook()?;
    let rechecked = private_identity(&file, root)?;
    if rechecked != measured || rechecked.size != current.len() as u64 {
        return Err(SetupError::Filesystem(
            "setup partial changed after its prefix was measured".to_owned(),
        ));
    }
    verify_named(root, PARTIAL_NAME, rechecked)?;
    if !expected.starts_with(&current) {
        fs::unlinkat(root.fd.as_fd(), PARTIAL_NAME, AtFlags::empty())
            .map_err(|error| io_error("remove conflicting setup partial", error))?;
        sync_directory(root)?;
        file = create_private_file(root, PARTIAL_NAME)?;
        current.clear();
    }
    let identity = private_identity(&file, root)?;
    if current.len() < expected.len() {
        if identity.size != current.len() as u64 {
            return Err(SetupError::Filesystem(
                "setup partial length changed before resume".to_owned(),
            ));
        }
        verify_named(root, PARTIAL_NAME, identity)?;
        file.seek(SeekFrom::Start(current.len() as u64))?;
        file.write_all(&expected[current.len()..])?;
    }
    file.flush()?;
    full_sync(&file)?;
    let after = private_identity(&file, root)?;
    if identity.device != after.device || identity.inode != after.inode {
        return Err(SetupError::Filesystem(
            "setup partial changed while it was written".to_owned(),
        ));
    }
    verify_named(root, PARTIAL_NAME, after)?;
    let bytes = read_open_file(&mut file, MAX_CONFIG_BYTES)?;
    if bytes != expected {
        return Err(SetupError::Filesystem(
            "setup partial does not contain the requested config".to_owned(),
        ));
    }
    let final_identity = private_identity(&file, root)?;
    if after != final_identity {
        return Err(SetupError::Filesystem(
            "setup partial changed while it was verified".to_owned(),
        ));
    }
    verify_named(root, PARTIAL_NAME, final_identity)?;
    sync_directory(root)?;
    verify_named(root, PARTIAL_NAME, final_identity)?;
    Ok(final_identity)
}

fn read_optional(root: &Directory, name: &str, cap: usize) -> Result<Option<Vec<u8>>, SetupError> {
    read_optional_with_hook(root, name, cap, || {})
}

fn read_optional_with_hook(
    root: &Directory,
    name: &str,
    cap: usize,
    hook: impl FnOnce(),
) -> Result<Option<Vec<u8>>, SetupError> {
    Ok(read_optional_snapshot_with_hook(root, name, cap, hook)?.map(|(bytes, _)| bytes))
}

fn read_optional_snapshot_with_hook(
    root: &Directory,
    name: &str,
    cap: usize,
    hook: impl FnOnce(),
) -> Result<Option<(Vec<u8>, Identity)>, SetupError> {
    match open_private_file(root, name, false) {
        Ok(mut file) => {
            let before = private_identity(&file, root)?;
            let bytes = read_open_file(&mut file, cap)?;
            hook();
            let after = private_identity(&file, root)?;
            if before != after {
                return Err(SetupError::Filesystem(
                    "setup file changed while it was read".to_owned(),
                ));
            }
            verify_named(root, name, after)?;
            Ok(Some((bytes, after)))
        }
        Err(SetupError::Missing) => Ok(None),
        Err(error) => Err(error),
    }
}

fn observe_existing_config_from_root(root: &Directory) -> Result<ExistingConfig, SetupError> {
    let observed = read_optional_snapshot_with_hook(root, CONFIG_NAME, MAX_CONFIG_BYTES, || {})?;
    let (bytes, identity) = match observed {
        Some((bytes, identity)) => (Some(bytes), Some(identity)),
        None => (None, None),
    };
    let config = bytes.as_deref().map(ConfigV1::parse).transpose()?;
    Ok(ExistingConfig {
        root: None,
        config,
        bytes,
        identity,
    })
}

#[cfg(test)]
pub(super) fn hold_setup_lock(root_path: &Path) -> Result<unix::SetupLock, SetupError> {
    let (root, _) = open_or_create_root(root_path)?;
    acquire_lock(&root, LOCK_NAME)
}

fn remove_partial_if_private(root: &Directory) -> Result<(), SetupError> {
    let file = match open_private_file(root, PARTIAL_NAME, true) {
        Ok(file) => file,
        Err(SetupError::Missing) => return Ok(()),
        Err(error) => return Err(error),
    };
    let identity = private_identity(&file, root)?;
    verify_named(root, PARTIAL_NAME, identity)?;
    fs::unlinkat(root.fd.as_fd(), PARTIAL_NAME, AtFlags::empty())
        .map_err(|error| io_error("remove completed setup partial", error))?;
    sync_directory(root)
}

fn read_required(root: &Directory, name: &str, cap: usize) -> Result<Vec<u8>, SetupError> {
    read_optional(root, name, cap)?.ok_or(SetupError::Missing)
}

fn read_open_file(file: &mut File, cap: usize) -> Result<Vec<u8>, SetupError> {
    file.seek(SeekFrom::Start(0))?;
    let mut bytes = Vec::new();
    file.take((cap + 1) as u64).read_to_end(&mut bytes)?;
    if bytes.len() > cap {
        return Err(SetupError::InvalidConfig(format!(
            "config input exceeds {cap} bytes"
        )));
    }
    Ok(bytes)
}
