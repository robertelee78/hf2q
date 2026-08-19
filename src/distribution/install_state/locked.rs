use std::fs::File;
use std::path::{Path, PathBuf};

use super::unix::{self, Directory, EntryIdentity};
use super::InstallStateError;

/// Descriptor-backed authority for one lock-held installation-state root.
///
/// This capability is intentionally neither cloneable nor serializable.  It
/// is the common exclusion boundary for activation and update-metadata state.
#[derive(Debug)]
pub(super) struct LockedInstallation {
    root_path: PathBuf,
    root: Directory,
    update: Directory,
    _lock: File,
    lock_identity: EntryIdentity,
}

#[derive(Debug)]
pub(super) struct LiveLockedNamespace {
    pub(super) root: Directory,
    pub(super) update: Directory,
}

impl LockedInstallation {
    pub(super) fn acquire(root_path: &Path) -> Result<Self, InstallStateError> {
        let root = unix::open_or_create_root(root_path)?;
        // Root/update are the only race-safe bootstrap mutations before the
        // shared installation lock is held.
        let update = unix::ensure_private_directory(&root, "update")?;
        let (lock, lock_identity) = unix::acquire_nonblocking_lock(&update)?;
        file_system_preflight(&update)?;
        Ok(Self {
            root_path: root_path.to_owned(),
            root,
            update,
            _lock: lock,
            lock_identity,
        })
    }

    pub(super) fn root(&self) -> &Directory {
        &self.root
    }

    pub(super) fn update(&self) -> &Directory {
        &self.update
    }

    pub(super) fn lock_identity(&self) -> EntryIdentity {
        self.lock_identity
    }

    /// Flush all prior state-root directory barriers to stable media through
    /// the exact lock file that anchors this installation transaction.
    ///
    /// This is the recovery endpoint when no selected metadata selector
    /// exists yet, so there is no `current.json` file available to full-sync.
    pub(super) fn full_sync_endpoint(&self) -> Result<(), InstallStateError> {
        let _ = self.reopen()?;
        unix::full_sync_file(&self._lock)
    }

    /// Reopen every named component from the originally authorized path.
    ///
    /// Holding a descriptor is not enough: a same-user namespace replacement
    /// must not let a stale directory authorize a commit in detached state.
    pub(super) fn reopen(&self) -> Result<LiveLockedNamespace, InstallStateError> {
        let root = unix::open_existing_root(&self.root_path)?;
        if !root.same_object(&self.root) {
            return Err(InstallStateError::InvalidLayout(
                "named state root changed while its lock was held",
            ));
        }
        let update = unix::open_directory_at(&root, "update", Some(0o700), true)?;
        if !update.same_object(&self.update) {
            return Err(InstallStateError::InvalidLayout(
                "named update directory changed after preparation",
            ));
        }
        unix::verify_named_identity(&update, "install.lock", self.lock_identity)?;
        Ok(LiveLockedNamespace { root, update })
    }
}

fn file_system_preflight(update: &Directory) -> Result<(), InstallStateError> {
    super::file::write_or_resume_private_file(update, ".noreplace-source", b"")?;
    super::file::write_or_resume_private_file(update, ".noreplace-target", b"")?;
    unix::preflight_noreplace(update, ".noreplace-source", ".noreplace-target")
}
