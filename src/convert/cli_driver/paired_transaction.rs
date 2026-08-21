//! Crash-recoverable publication for one text GGUF and its projector.
//!
//! The text GGUF is always promoted last.  A durable journal plus
//! same-filesystem backups makes every earlier state recoverable. The text
//! rename becomes the commit marker only after its destination and source
//! directories have both been synchronized.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use crate::core::paired_artifact::{
    canonical_parent, file_name, journal_path, path_entry_exists, read_pair_journal,
    sync_directory, sync_path, FileIdentity, PairArtifactError, PairJournalMember, PairLock,
    PairMemberRole, PairTextLock, PairTransactionJournal, PAIR_JOURNAL_SCHEMA_VERSION,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PublishFailpoint {
    AfterJournal,
    AfterBackup(PairMemberRole),
    AfterPromoteRename(PairMemberRole),
    AfterPromote(PairMemberRole),
}

pub(super) struct PairWorkspace {
    transaction_id: String,
    parent: PathBuf,
    root: PathBuf,
    text: PathBuf,
}

impl PairWorkspace {
    pub(super) fn create(text: &Path) -> Result<Self, PairArtifactError> {
        recover_pending(text)?;
        let parent = canonical_parent(text)?;
        let text = parent.join(file_name(text)?);
        let transaction_id = uuid::Uuid::new_v4().to_string();
        let root = parent.join(format!(".hf2q-pair-{transaction_id}"));
        fs::create_dir(&root).map_err(|source| PairArtifactError::Io {
            path: root.clone(),
            source,
        })?;
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).map_err(|source| {
            PairArtifactError::Io {
                path: root.clone(),
                source,
            }
        })?;
        let backup = root.join("backup");
        fs::create_dir(&backup).map_err(|source| PairArtifactError::Io {
            path: backup.clone(),
            source,
        })?;
        fs::set_permissions(&backup, fs::Permissions::from_mode(0o700)).map_err(|source| {
            PairArtifactError::Io {
                path: backup.clone(),
                source,
            }
        })?;
        sync_directory(&root)?;
        sync_directory(&parent)?;
        Ok(Self {
            transaction_id,
            parent,
            root,
            text,
        })
    }

    pub(super) fn transaction_id(&self) -> &str {
        &self.transaction_id
    }

    pub(super) fn staged_path(&self, role: PairMemberRole) -> PathBuf {
        self.root.join(role.private_name())
    }

    pub(super) fn discard_unpublished(&self) {
        if !path_entry_exists(&journal_path(&self.text)).unwrap_or(true) {
            if let Err(error) = remove_transaction_root(&self.parent, &self.root) {
                tracing::warn!(error = %error, root = %self.root.display(), "could not remove unpublished pair workspace");
            }
        }
    }

    pub(super) fn publish(
        &self,
        destinations: &[(PairMemberRole, PathBuf)],
    ) -> Result<(), PairArtifactError> {
        self.publish_inner(destinations, None, false)
    }

    #[cfg(test)]
    fn publish_crash_at(
        &self,
        destinations: &[(PairMemberRole, PathBuf)],
        failpoint: PublishFailpoint,
    ) -> Result<(), PairArtifactError> {
        self.publish_inner(destinations, Some(failpoint), true)
    }

    fn publish_inner(
        &self,
        destinations: &[(PairMemberRole, PathBuf)],
        failpoint: Option<PublishFailpoint>,
        simulate_crash: bool,
    ) -> Result<(), PairArtifactError> {
        let _lock = PairLock::exclusive(&self.text)?;
        let pre_recovery_text_lock = PairTextLock::exclusive_if_present(&self.text)?;
        recover_locked(&self.text, &self.parent)?;
        // Recovery may replace the final text inode (candidate -> prior) or
        // recreate it from backup. Release the pre-recovery inode only after
        // recovery, then lock whatever inode is current before starting the
        // new journal/mutation. A fallback reader that wins this handoff can
        // only make us wait; no publication mutation has begun yet.
        drop(pre_recovery_text_lock);
        let _current_text_lock = PairTextLock::exclusive_if_present(&self.text)?;
        let journal_file = journal_path(&self.text);
        if path_entry_exists(&journal_file)? {
            return Err(PairArtifactError::Invalid(
                "pair transaction journal still exists after recovery".into(),
            ));
        }
        validate_destination_order(destinations)?;

        let parent_metadata =
            fs::metadata(&self.parent).map_err(|source| PairArtifactError::Io {
                path: self.parent.clone(),
                source,
            })?;
        let mut members = Vec::with_capacity(destinations.len());
        for (role, destination) in destinations {
            if canonical_parent(destination)? != self.parent {
                return Err(PairArtifactError::Invalid(
                    "all pair destinations must share the canonical text directory".into(),
                ));
            }
            let staged = self.staged_path(*role);
            let candidate = FileIdentity::from_path(&staged)?;
            if let Some(candidate) = candidate.as_ref() {
                sync_path(&staged)?;
                let staged_metadata =
                    fs::symlink_metadata(&staged).map_err(|source| PairArtifactError::Io {
                        path: staged.clone(),
                        source,
                    })?;
                if !staged_metadata.is_file()
                    || staged_metadata.uid() != rustix::process::geteuid().as_raw()
                    || staged_metadata.nlink() != 1
                    || staged_metadata.dev() != parent_metadata.dev()
                    || &FileIdentity::from_path(&staged)?.ok_or_else(|| {
                        PairArtifactError::Invalid(format!(
                            "staged pair member disappeared: {}",
                            staged.display()
                        ))
                    })? != candidate
                {
                    return Err(PairArtifactError::Invalid(format!(
                        "staged pair member is not one stable owned same-filesystem regular file: {}",
                        staged.display()
                    )));
                }
            } else if matches!(*role, PairMemberRole::Projector | PairMemberRole::Text) {
                return Err(PairArtifactError::Invalid(format!(
                    "staged required pair member is missing: {}",
                    staged.display()
                )));
            }
            let final_path = self.parent.join(file_name(destination)?);
            let prior = FileIdentity::from_path(&final_path)?;
            if let Some(prior_metadata) = fs::symlink_metadata(&final_path).ok() {
                if !prior_metadata.is_file()
                    || prior_metadata.file_type().is_symlink()
                    || prior_metadata.uid() != rustix::process::geteuid().as_raw()
                    || prior_metadata.nlink() != 1
                    || prior_metadata.dev() != parent_metadata.dev()
                {
                    return Err(PairArtifactError::Invalid(format!(
                        "existing pair destination is not one owned same-filesystem regular file: {}",
                        final_path.display()
                    )));
                }
            }
            let final_name = file_name(destination)?
                .to_str()
                .ok_or_else(|| {
                    PairArtifactError::Invalid(
                        "paired conversion destinations must have UTF-8 filenames".into(),
                    )
                })?
                .to_owned();
            members.push(PairJournalMember {
                role: *role,
                final_name,
                prior,
                candidate,
            });
        }
        let journal = PairTransactionJournal {
            schema_version: PAIR_JOURNAL_SCHEMA_VERSION,
            transaction_id: self.transaction_id.clone(),
            transaction_root: file_name(&self.root)?
                .to_str()
                .ok_or_else(|| {
                    PairArtifactError::Invalid("pair transaction root is not UTF-8".into())
                })?
                .to_owned(),
            members,
        };
        journal.validate()?;
        write_journal(&journal_file, &journal)?;

        let mut text_commit_durable = false;
        let mut candidate_text_lock = None;
        let mutation_result = (|| {
            if failpoint == Some(PublishFailpoint::AfterJournal) {
                return Err(PairArtifactError::Invalid(
                    "injected failure after pair journal".into(),
                ));
            }
            for member in &journal.members {
                if member.prior.is_some() {
                    let final_path = journal.final_path(&self.parent, member);
                    let backup_path = journal.backup_path(&self.parent, member.role);
                    fs::rename(&final_path, &backup_path).map_err(|source| {
                        PairArtifactError::Io {
                            path: final_path,
                            source,
                        }
                    })?;
                    sync_directory(&self.root.join("backup"))?;
                    sync_directory(&self.parent)?;
                }
                if failpoint == Some(PublishFailpoint::AfterBackup(member.role)) {
                    return Err(PairArtifactError::Invalid(
                        "injected failure after pair backup".into(),
                    ));
                }
            }

            for member in &journal.members {
                if member.candidate.is_none() {
                    continue;
                }
                let staged = journal.staged_path(&self.parent, member.role);
                let final_path = journal.final_path(&self.parent, member);
                if member.role == PairMemberRole::Text {
                    // Lock the private candidate inode before rename. The
                    // lock follows that inode into its final name, leaving no
                    // visibility gap for read-only/cross-user readers.
                    candidate_text_lock = Some(PairTextLock::exclusive(&staged)?);
                }
                fs::rename(&staged, &final_path).map_err(|source| PairArtifactError::Io {
                    path: final_path.clone(),
                    source,
                })?;
                if failpoint == Some(PublishFailpoint::AfterPromoteRename(member.role)) {
                    return Err(PairArtifactError::Invalid(format!(
                        "injected failure after pair {:?} rename",
                        member.role
                    )));
                }
                sync_directory(&self.parent)?;
                sync_directory(&self.root)?;
                if member.role == PairMemberRole::Text {
                    text_commit_durable = true;
                }
                if failpoint == Some(PublishFailpoint::AfterPromote(member.role)) {
                    return Err(PairArtifactError::Invalid(format!(
                        "injected failure after pair {:?} promotion",
                        member.role
                    )));
                }
            }
            Ok(())
        })();

        if let Err(error) = mutation_result {
            if !simulate_crash {
                let recovery = if text_commit_durable {
                    cleanup_committed(&journal_file, &self.parent, &journal)
                } else {
                    rollback(&journal_file, &self.parent, &journal)
                };
                if let Err(recovery) = recovery {
                    return Err(PairArtifactError::Invalid(format!(
                        "pair publication failed ({error}); immediate recovery also failed ({recovery})"
                    )));
                }
            }
            return Err(error);
        }

        if let Err(error) = cleanup_committed(&journal_file, &self.parent, &journal) {
            tracing::warn!(
                error = %error,
                journal = %journal_file.display(),
                "pair committed; journal cleanup remains recoverable"
            );
        }
        drop(candidate_text_lock);
        Ok(())
    }
}

pub(super) fn recover_pending(text: &Path) -> Result<(), PairArtifactError> {
    let parent = canonical_parent(text)?;
    let text = parent.join(file_name(text)?);
    let _lock = PairLock::exclusive(&text)?;
    let _text_lock = PairTextLock::exclusive_if_present(&text)?;
    recover_locked(&text, &parent)
}

fn recover_locked(text: &Path, parent: &Path) -> Result<(), PairArtifactError> {
    let journal_file = journal_path(text);
    if !path_entry_exists(&journal_file)? {
        return Ok(());
    }
    let journal = read_pair_journal(&journal_file)?;
    validate_recovery_binding(text, &journal)?;
    if journal.committed(parent)? {
        cleanup_committed(&journal_file, parent, &journal)
    } else if journal.rolled_back(parent)? {
        cleanup_terminal(&journal_file, parent, &journal)
    } else {
        validate_recovery_root(parent, &journal)?;
        rollback(&journal_file, parent, &journal)
    }
}

fn validate_recovery_binding(
    text: &Path,
    journal: &PairTransactionJournal,
) -> Result<(), PairArtifactError> {
    let text_name = file_name(text)?
        .to_str()
        .ok_or_else(|| PairArtifactError::Invalid("pair text filename is not UTF-8".into()))?;
    if journal
        .members
        .iter()
        .find(|member| member.role == PairMemberRole::Text)
        .map(|member| member.final_name.as_str())
        != Some(text_name)
    {
        return Err(PairArtifactError::Invalid(
            "pair journal text member does not match its journal filename".into(),
        ));
    }
    Ok(())
}

fn validate_recovery_root(
    parent: &Path,
    journal: &PairTransactionJournal,
) -> Result<(), PairArtifactError> {
    let parent_metadata = fs::metadata(parent).map_err(|source| PairArtifactError::Io {
        path: parent.to_path_buf(),
        source,
    })?;
    for directory in [
        journal.root_path(parent),
        journal.root_path(parent).join("backup"),
    ] {
        let metadata =
            fs::symlink_metadata(&directory).map_err(|source| PairArtifactError::Io {
                path: directory.clone(),
                source,
            })?;
        if !metadata.is_dir()
            || metadata.uid() != rustix::process::geteuid().as_raw()
            || metadata.dev() != parent_metadata.dev()
            || metadata.mode() & 0o7777 != 0o700
        {
            return Err(PairArtifactError::Invalid(format!(
                "pair recovery directory is not an owned private same-filesystem directory: {}",
                directory.display()
            )));
        }
    }
    Ok(())
}

fn rollback(
    journal_file: &Path,
    parent: &Path,
    journal: &PairTransactionJournal,
) -> Result<(), PairArtifactError> {
    for member in journal.members.iter().rev() {
        let final_path = journal.final_path(parent, member);
        match FileIdentity::from_path(&final_path)? {
            Some(identity) if member.candidate.as_ref() == Some(&identity) => {
                fs::remove_file(&final_path).map_err(|source| PairArtifactError::Io {
                    path: final_path.clone(),
                    source,
                })?;
                sync_directory(parent)?;
            }
            Some(identity) if member.prior.as_ref() == Some(&identity) => {}
            Some(_) => {
                return Err(PairArtifactError::Invalid(format!(
                    "refusing to recover over an unknown pair member: {}",
                    final_path.display()
                )));
            }
            None => {}
        }
    }

    for member in &journal.members {
        let final_path = journal.final_path(parent, member);
        let backup_path = journal.backup_path(parent, member.role);
        match member.prior.as_ref() {
            Some(prior) => {
                if prior.matches_path(&final_path)? {
                    continue;
                }
                if !prior.matches_path(&backup_path)? {
                    return Err(PairArtifactError::Invalid(format!(
                        "pair recovery cannot find the prior {:?} member",
                        member.role
                    )));
                }
                if final_path.exists() {
                    return Err(PairArtifactError::Invalid(format!(
                        "pair recovery found an unexpected final member: {}",
                        final_path.display()
                    )));
                }
                fs::rename(&backup_path, &final_path).map_err(|source| PairArtifactError::Io {
                    path: final_path.clone(),
                    source,
                })?;
                sync_directory(parent)?;
                sync_directory(&journal.root_path(parent).join("backup"))?;
            }
            None => {
                if FileIdentity::from_path(&final_path)?.is_some()
                    || FileIdentity::from_path(&backup_path)?.is_some()
                {
                    return Err(PairArtifactError::Invalid(format!(
                        "pair recovery found a member absent from the baseline: {}",
                        final_path.display()
                    )));
                }
            }
        }
    }
    cleanup_terminal(journal_file, parent, journal)
}

fn cleanup_committed(
    journal_file: &Path,
    parent: &Path,
    journal: &PairTransactionJournal,
) -> Result<(), PairArtifactError> {
    if !journal.committed(parent)? {
        return Err(PairArtifactError::Invalid(
            "cannot clean forward an uncommitted pair transaction".into(),
        ));
    }
    cleanup_terminal(journal_file, parent, journal)
}

fn cleanup_terminal(
    journal_file: &Path,
    parent: &Path,
    journal: &PairTransactionJournal,
) -> Result<(), PairArtifactError> {
    // The journal is the recovery obligation. Remove and persist it before
    // deleting private staging/backups: a crash can then leave only a safe
    // orphan, never a journal whose required recovery state has vanished.
    remove_journal(journal_file, parent)?;
    if let Err(error) = remove_transaction_root(parent, &journal.root_path(parent)) {
        tracing::warn!(
            error = %error,
            root = %journal.root_path(parent).display(),
            "pair terminal state is durable; private transaction orphan could not be removed"
        );
    }
    Ok(())
}

fn remove_journal(path: &Path, parent: &Path) -> Result<(), PairArtifactError> {
    match fs::remove_file(path) {
        Ok(()) => sync_directory(parent),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(PairArtifactError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn remove_transaction_root(parent: &Path, root: &Path) -> Result<(), PairArtifactError> {
    if root.parent() != Some(parent)
        || !file_name(root)?
            .to_str()
            .is_some_and(|name| name.starts_with(".hf2q-pair-"))
    {
        return Err(PairArtifactError::Invalid(
            "refusing to remove a non-transaction directory".into(),
        ));
    }
    let metadata = match fs::symlink_metadata(root) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(source) => {
            return Err(PairArtifactError::Io {
                path: root.to_path_buf(),
                source,
            });
        }
    };
    let parent_metadata = fs::metadata(parent).map_err(|source| PairArtifactError::Io {
        path: parent.to_path_buf(),
        source,
    })?;
    if !metadata.is_dir()
        || metadata.uid() != rustix::process::geteuid().as_raw()
        || metadata.dev() != parent_metadata.dev()
        || metadata.mode() & 0o7777 != 0o700
    {
        return Err(PairArtifactError::Invalid(
            "refusing to remove an unowned pair transaction directory".into(),
        ));
    }
    fs::remove_dir_all(root).map_err(|source| PairArtifactError::Io {
        path: root.to_path_buf(),
        source,
    })?;
    sync_directory(parent)
}

fn write_journal(path: &Path, journal: &PairTransactionJournal) -> Result<(), PairArtifactError> {
    let parent = canonical_parent(path)?;
    let temporary = parent.join(format!(
        ".hf2q-pair-journal-{}.partial",
        journal.transaction_id
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(&temporary)
        .map_err(|source| PairArtifactError::Io {
            path: temporary.clone(),
            source,
        })?;
    serde_json::to_writer_pretty(&mut file, journal)?;
    file.write_all(b"\n")
        .map_err(|source| PairArtifactError::Io {
            path: temporary.clone(),
            source,
        })?;
    drop(file);
    sync_path(&temporary)?;
    fs::rename(&temporary, path).map_err(|source| PairArtifactError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    sync_directory(&parent)
}

fn validate_destination_order(
    destinations: &[(PairMemberRole, PathBuf)],
) -> Result<(), PairArtifactError> {
    if destinations.is_empty()
        || destinations.last().map(|(role, _)| *role) != Some(PairMemberRole::Text)
        || !destinations
            .iter()
            .any(|(role, _)| *role == PairMemberRole::Projector)
    {
        return Err(PairArtifactError::Invalid(
            "pair publication requires projector and text-last ordering".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    fn write(path: &Path, value: &[u8]) {
        fs::write(path, value).unwrap();
    }

    fn destinations(dir: &Path) -> Vec<(PairMemberRole, PathBuf)> {
        vec![
            (PairMemberRole::Projector, dir.join("model-mmproj.gguf")),
            (
                PairMemberRole::ProjectorReceipt,
                dir.join("model-mmproj.gguf.receipt.json"),
            ),
            (
                PairMemberRole::ProjectorTensorReceipt,
                dir.join("model-mmproj.gguf.tensor-conversion.json"),
            ),
            (
                PairMemberRole::TextReceipt,
                dir.join("model.gguf.receipt.json"),
            ),
            (
                PairMemberRole::TextTensorReceipt,
                dir.join("model.gguf.tensor-conversion.json"),
            ),
            (PairMemberRole::Text, dir.join("model.gguf")),
        ]
    }

    fn seed_old_and_new(workspace: &PairWorkspace, destinations: &[(PairMemberRole, PathBuf)]) {
        for (role, final_path) in destinations {
            write(final_path, format!("old-{role:?}").as_bytes());
            write(
                &workspace.staged_path(*role),
                format!("new-{role:?}").as_bytes(),
            );
        }
    }

    #[test]
    fn every_precommit_crash_restores_the_complete_old_pair() {
        let failpoints = [
            PublishFailpoint::AfterJournal,
            PublishFailpoint::AfterBackup(PairMemberRole::Projector),
            PublishFailpoint::AfterBackup(PairMemberRole::ProjectorReceipt),
            PublishFailpoint::AfterBackup(PairMemberRole::ProjectorTensorReceipt),
            PublishFailpoint::AfterBackup(PairMemberRole::TextReceipt),
            PublishFailpoint::AfterBackup(PairMemberRole::TextTensorReceipt),
            PublishFailpoint::AfterBackup(PairMemberRole::Text),
            PublishFailpoint::AfterPromoteRename(PairMemberRole::Projector),
            PublishFailpoint::AfterPromoteRename(PairMemberRole::ProjectorReceipt),
            PublishFailpoint::AfterPromoteRename(PairMemberRole::ProjectorTensorReceipt),
            PublishFailpoint::AfterPromoteRename(PairMemberRole::TextReceipt),
            PublishFailpoint::AfterPromoteRename(PairMemberRole::TextTensorReceipt),
            PublishFailpoint::AfterPromote(PairMemberRole::Projector),
            PublishFailpoint::AfterPromote(PairMemberRole::ProjectorReceipt),
            PublishFailpoint::AfterPromote(PairMemberRole::ProjectorTensorReceipt),
            PublishFailpoint::AfterPromote(PairMemberRole::TextReceipt),
            PublishFailpoint::AfterPromote(PairMemberRole::TextTensorReceipt),
        ];
        for failpoint in failpoints {
            let dir = tempfile::tempdir().unwrap();
            let text = dir.path().join("model.gguf");
            let workspace = PairWorkspace::create(&text).unwrap();
            let destinations = destinations(dir.path());
            seed_old_and_new(&workspace, &destinations);
            workspace
                .publish_crash_at(&destinations, failpoint)
                .unwrap_err();
            recover_pending(&text).unwrap();
            for (role, final_path) in &destinations {
                assert_eq!(
                    fs::read(final_path).unwrap(),
                    format!("old-{role:?}").as_bytes(),
                    "failpoint {failpoint:?} role {role:?}"
                );
            }
            assert!(!journal_path(&text).exists());
        }
    }

    #[test]
    fn every_precommit_in_process_failure_immediately_restores_the_complete_old_pair() {
        let failpoints = [
            PublishFailpoint::AfterJournal,
            PublishFailpoint::AfterBackup(PairMemberRole::Projector),
            PublishFailpoint::AfterBackup(PairMemberRole::ProjectorReceipt),
            PublishFailpoint::AfterBackup(PairMemberRole::ProjectorTensorReceipt),
            PublishFailpoint::AfterBackup(PairMemberRole::TextReceipt),
            PublishFailpoint::AfterBackup(PairMemberRole::TextTensorReceipt),
            PublishFailpoint::AfterBackup(PairMemberRole::Text),
            PublishFailpoint::AfterPromoteRename(PairMemberRole::Projector),
            PublishFailpoint::AfterPromoteRename(PairMemberRole::ProjectorReceipt),
            PublishFailpoint::AfterPromoteRename(PairMemberRole::ProjectorTensorReceipt),
            PublishFailpoint::AfterPromoteRename(PairMemberRole::TextReceipt),
            PublishFailpoint::AfterPromoteRename(PairMemberRole::TextTensorReceipt),
            PublishFailpoint::AfterPromoteRename(PairMemberRole::Text),
            PublishFailpoint::AfterPromote(PairMemberRole::Projector),
            PublishFailpoint::AfterPromote(PairMemberRole::ProjectorReceipt),
            PublishFailpoint::AfterPromote(PairMemberRole::ProjectorTensorReceipt),
            PublishFailpoint::AfterPromote(PairMemberRole::TextReceipt),
            PublishFailpoint::AfterPromote(PairMemberRole::TextTensorReceipt),
        ];
        for failpoint in failpoints {
            let dir = tempfile::tempdir().unwrap();
            let text = dir.path().join("model.gguf");
            let workspace = PairWorkspace::create(&text).unwrap();
            let destinations = destinations(dir.path());
            seed_old_and_new(&workspace, &destinations);
            workspace
                .publish_inner(&destinations, Some(failpoint), false)
                .unwrap_err();
            for (role, final_path) in &destinations {
                assert_eq!(
                    fs::read(final_path).unwrap(),
                    format!("old-{role:?}").as_bytes(),
                    "failpoint {failpoint:?} role {role:?}"
                );
            }
            assert!(!journal_path(&text).exists());
        }
    }

    #[test]
    fn crash_after_text_commit_cleans_forward_to_the_complete_new_pair() {
        let dir = tempfile::tempdir().unwrap();
        let text = dir.path().join("model.gguf");
        let workspace = PairWorkspace::create(&text).unwrap();
        let destinations = destinations(dir.path());
        seed_old_and_new(&workspace, &destinations);
        workspace
            .publish_crash_at(
                &destinations,
                PublishFailpoint::AfterPromote(PairMemberRole::Text),
            )
            .unwrap_err();
        recover_pending(&text).unwrap();
        for (role, final_path) in &destinations {
            assert_eq!(
                fs::read(final_path).unwrap(),
                format!("new-{role:?}").as_bytes()
            );
        }
        assert!(!journal_path(&text).exists());
    }

    #[test]
    fn fresh_destination_precommit_crash_returns_to_absent() {
        let dir = tempfile::tempdir().unwrap();
        let text = dir.path().join("model.gguf");
        let workspace = PairWorkspace::create(&text).unwrap();
        let destinations = destinations(dir.path());
        for (role, _) in &destinations {
            write(
                &workspace.staged_path(*role),
                format!("new-{role:?}").as_bytes(),
            );
        }
        workspace
            .publish_crash_at(
                &destinations,
                PublishFailpoint::AfterPromote(PairMemberRole::TextTensorReceipt),
            )
            .unwrap_err();
        recover_pending(&text).unwrap();
        for (_, final_path) in &destinations {
            assert!(!final_path.exists());
        }
    }

    #[test]
    fn committed_local_pair_removes_prior_remote_receipts() {
        let dir = tempfile::tempdir().unwrap();
        let text = dir.path().join("model.gguf");
        let workspace = PairWorkspace::create(&text).unwrap();
        let destinations = destinations(dir.path());
        for (role, final_path) in &destinations {
            write(final_path, format!("old-{role:?}").as_bytes());
            if matches!(role, PairMemberRole::Projector | PairMemberRole::Text) {
                write(
                    &workspace.staged_path(*role),
                    format!("new-{role:?}").as_bytes(),
                );
            }
        }

        workspace.publish(&destinations).unwrap();

        for (role, final_path) in &destinations {
            match role {
                PairMemberRole::Projector | PairMemberRole::Text => assert_eq!(
                    fs::read(final_path).unwrap(),
                    format!("new-{role:?}").as_bytes()
                ),
                PairMemberRole::ProjectorReceipt
                | PairMemberRole::ProjectorTensorReceipt
                | PairMemberRole::TextReceipt
                | PairMemberRole::TextTensorReceipt => assert!(!final_path.exists()),
            }
        }
        assert!(!journal_path(&text).exists());
    }

    #[test]
    fn committed_terminal_journal_recovers_after_backup_directory_is_gone() {
        let dir = tempfile::tempdir().unwrap();
        let text = dir.path().join("model.gguf");
        let workspace = PairWorkspace::create(&text).unwrap();
        let destinations = destinations(dir.path());
        seed_old_and_new(&workspace, &destinations);
        workspace
            .publish_crash_at(
                &destinations,
                PublishFailpoint::AfterPromote(PairMemberRole::Text),
            )
            .unwrap_err();
        fs::remove_dir_all(workspace.root.join("backup")).unwrap();

        recover_pending(&text).unwrap();

        for (role, final_path) in &destinations {
            assert_eq!(
                fs::read(final_path).unwrap(),
                format!("new-{role:?}").as_bytes()
            );
        }
        assert!(!journal_path(&text).exists());
    }

    #[test]
    fn rolled_back_terminal_journal_recovers_after_transaction_root_is_gone() {
        let dir = tempfile::tempdir().unwrap();
        let text = dir.path().join("model.gguf");
        let workspace = PairWorkspace::create(&text).unwrap();
        let destinations = destinations(dir.path());
        seed_old_and_new(&workspace, &destinations);
        workspace
            .publish_crash_at(&destinations, PublishFailpoint::AfterJournal)
            .unwrap_err();
        fs::remove_dir_all(&workspace.root).unwrap();

        recover_pending(&text).unwrap();

        for (role, final_path) in &destinations {
            assert_eq!(
                fs::read(final_path).unwrap(),
                format!("old-{role:?}").as_bytes()
            );
        }
        assert!(!journal_path(&text).exists());
    }

    #[test]
    fn recovery_rejects_a_symlink_journal_without_touching_its_target() {
        let dir = tempfile::tempdir().unwrap();
        let text = dir.path().join("model.gguf");
        let sibling = dir.path().join("do-not-touch.json");
        write(&text, b"old-text");
        write(&sibling, b"sibling");
        symlink(&sibling, journal_path(&text)).unwrap();

        let error = recover_pending(&text).unwrap_err().to_string();

        assert!(error.contains("pair I/O"), "unexpected error: {error}");
        assert_eq!(fs::read(&text).unwrap(), b"old-text");
        assert_eq!(fs::read(&sibling).unwrap(), b"sibling");
    }

    #[test]
    fn recovery_rejects_a_journal_bound_to_another_text_name() {
        let dir = tempfile::tempdir().unwrap();
        let text = dir.path().join("model.gguf");
        let sibling = dir.path().join("do-not-touch.gguf");
        write(&text, b"old-text");
        write(&sibling, b"sibling");
        let workspace = PairWorkspace::create(&text).unwrap();
        write(
            &workspace.staged_path(PairMemberRole::Projector),
            b"new-projector",
        );
        write(&workspace.staged_path(PairMemberRole::Text), b"new-text");
        let journal = PairTransactionJournal {
            schema_version: PAIR_JOURNAL_SCHEMA_VERSION,
            transaction_id: workspace.transaction_id.clone(),
            transaction_root: file_name(&workspace.root)
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned(),
            members: vec![
                PairJournalMember {
                    role: PairMemberRole::Projector,
                    final_name: "model-mmproj.gguf".into(),
                    prior: None,
                    candidate: FileIdentity::from_path(
                        &workspace.staged_path(PairMemberRole::Projector),
                    )
                    .unwrap(),
                },
                PairJournalMember {
                    role: PairMemberRole::Text,
                    final_name: "do-not-touch.gguf".into(),
                    prior: FileIdentity::from_path(&sibling).unwrap(),
                    candidate: FileIdentity::from_path(
                        &workspace.staged_path(PairMemberRole::Text),
                    )
                    .unwrap(),
                },
            ],
        };
        write_journal(&journal_path(&text), &journal).unwrap();

        let error = recover_pending(&text).unwrap_err().to_string();

        assert!(
            error.contains("does not match"),
            "unexpected error: {error}"
        );
        assert_eq!(fs::read(&text).unwrap(), b"old-text");
        assert_eq!(fs::read(&sibling).unwrap(), b"sibling");
    }

    #[test]
    fn publication_rejects_symlink_destinations_without_touching_the_target() {
        let dir = tempfile::tempdir().unwrap();
        let text = dir.path().join("model.gguf");
        let sibling = dir.path().join("do-not-touch.gguf");
        let workspace = PairWorkspace::create(&text).unwrap();
        let destinations = destinations(dir.path());
        for (role, _) in &destinations {
            write(
                &workspace.staged_path(*role),
                format!("new-{role:?}").as_bytes(),
            );
        }
        write(&sibling, b"sibling");
        let projector = destinations
            .iter()
            .find(|(role, _)| *role == PairMemberRole::Projector)
            .unwrap()
            .1
            .clone();
        symlink(&sibling, &projector).unwrap();

        let error = workspace.publish(&destinations).unwrap_err().to_string();

        assert!(error.contains("regular file"), "unexpected error: {error}");
        assert_eq!(fs::read(&sibling).unwrap(), b"sibling");
        assert!(!journal_path(&text).exists());
    }
}
