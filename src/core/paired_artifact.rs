//! Shared multimodal-pair metadata, locking, and crash-journal schema.
//!
//! Pair publication itself lives in the converter.  This module owns the
//! small contract shared with serving: one lock endpoint, one generation on
//! both GGUF members, and a durable journal which is reader-visible only
//! while publication recovery is pending.

use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};

use mlx_native::gguf::GgufFile;
use rustix::fs::FlockOperation;
use serde::{Deserialize, Serialize};

use crate::core::provenance::KEY_MMPROJ_SHA256;

pub(crate) const PAIR_JOURNAL_SCHEMA_VERSION: u32 = 1;
pub(crate) const PAIR_METADATA_SCHEMA_VERSION: &str = "1";
pub(crate) const KEY_PAIR_SCHEMA_VERSION: &str = "hf2q.pair_schema_version";
pub(crate) const KEY_PAIR_GENERATION: &str = "hf2q.pair_generation";
const MAX_PAIR_JOURNAL_BYTES: u64 = 64 * 1024;

#[derive(Debug, thiserror::Error)]
pub(crate) enum PairArtifactError {
    #[error("pair I/O at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid multimodal pair: {0}")]
    Invalid(String),
    #[error("parse pair transaction journal: {0}")]
    Journal(#[from] serde_json::Error),
}

fn io(path: &Path, source: std::io::Error) -> PairArtifactError {
    PairArtifactError::Io {
        path: path.to_path_buf(),
        source,
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PairMemberRole {
    Projector,
    ProjectorReceipt,
    ProjectorTensorReceipt,
    TextReceipt,
    TextTensorReceipt,
    Text,
}

impl PairMemberRole {
    pub(crate) fn private_name(self) -> &'static str {
        match self {
            Self::Projector => "projector.gguf",
            Self::ProjectorReceipt => "projector.gguf.receipt.json",
            Self::ProjectorTensorReceipt => "projector.gguf.tensor-conversion.json",
            Self::TextReceipt => "text.gguf.receipt.json",
            Self::TextTensorReceipt => "text.gguf.tensor-conversion.json",
            Self::Text => "text.gguf",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct FileIdentity {
    pub(crate) device: u64,
    pub(crate) inode: u64,
    pub(crate) mode: u32,
    pub(crate) links: u64,
    pub(crate) size: u64,
}

impl FileIdentity {
    fn from_file(file: &File, path: &Path) -> Result<Self, PairArtifactError> {
        let metadata = file.metadata().map_err(|error| io(path, error))?;
        Ok(Self::from_metadata(&metadata))
    }

    fn from_metadata(metadata: &fs::Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            mode: metadata.mode(),
            links: metadata.nlink(),
            size: metadata.len(),
        }
    }

    pub(crate) fn from_path(path: &Path) -> Result<Option<Self>, PairArtifactError> {
        match fs::symlink_metadata(path) {
            Ok(metadata) => Ok(Some(Self::from_metadata(&metadata))),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(io(path, error)),
        }
    }

    pub(crate) fn matches_path(&self, path: &Path) -> Result<bool, PairArtifactError> {
        Ok(Self::from_path(path)?.as_ref() == Some(self))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PairJournalMember {
    pub(crate) role: PairMemberRole,
    pub(crate) final_name: String,
    pub(crate) prior: Option<FileIdentity>,
    pub(crate) candidate: Option<FileIdentity>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PairTransactionJournal {
    pub(crate) schema_version: u32,
    pub(crate) transaction_id: String,
    pub(crate) transaction_root: String,
    pub(crate) members: Vec<PairJournalMember>,
}

impl PairTransactionJournal {
    pub(crate) fn validate(&self) -> Result<(), PairArtifactError> {
        if self.schema_version != PAIR_JOURNAL_SCHEMA_VERSION {
            return Err(PairArtifactError::Invalid(format!(
                "unsupported pair journal schema {}",
                self.schema_version
            )));
        }
        validate_transaction_id(&self.transaction_id)?;
        validate_single_name(&self.transaction_root, "transaction root")?;
        if self.transaction_root != format!(".hf2q-pair-{}", self.transaction_id) {
            return Err(PairArtifactError::Invalid(
                "pair journal transaction root does not match its transaction id".into(),
            ));
        }
        if self.members.is_empty()
            || self.members.last().map(|member| member.role) != Some(PairMemberRole::Text)
        {
            return Err(PairArtifactError::Invalid(
                "pair journal must publish text last".into(),
            ));
        }
        let mut roles = Vec::with_capacity(self.members.len());
        let mut names = Vec::with_capacity(self.members.len());
        for member in &self.members {
            validate_single_name(&member.final_name, "pair member")?;
            if matches!(
                member.role,
                PairMemberRole::Projector | PairMemberRole::Text
            ) && member.candidate.is_none()
            {
                return Err(PairArtifactError::Invalid(
                    "pair journal required artifact has no candidate identity".into(),
                ));
            }
            if let Some(candidate) = member.candidate.as_ref() {
                if candidate.mode & u32::from(libc::S_IFMT) != u32::from(libc::S_IFREG)
                    || candidate.links != 1
                    || candidate.size == 0
                {
                    return Err(PairArtifactError::Invalid(
                        "pair journal candidate is not one nonempty regular file identity".into(),
                    ));
                }
            }
            if let Some(prior) = member.prior.as_ref() {
                if prior.mode & u32::from(libc::S_IFMT) != u32::from(libc::S_IFREG)
                    || prior.links != 1
                {
                    return Err(PairArtifactError::Invalid(
                        "pair journal prior member is not one regular file identity".into(),
                    ));
                }
            }
            if roles.contains(&member.role) || names.contains(&member.final_name) {
                return Err(PairArtifactError::Invalid(
                    "pair journal contains duplicate member roles or paths".into(),
                ));
            }
            roles.push(member.role);
            names.push(member.final_name.clone());
        }
        if !roles.contains(&PairMemberRole::Projector) {
            return Err(PairArtifactError::Invalid(
                "pair journal has no projector member".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn root_path(&self, parent: &Path) -> PathBuf {
        parent.join(&self.transaction_root)
    }

    pub(crate) fn staged_path(&self, parent: &Path, role: PairMemberRole) -> PathBuf {
        self.root_path(parent).join(role.private_name())
    }

    pub(crate) fn backup_path(&self, parent: &Path, role: PairMemberRole) -> PathBuf {
        self.root_path(parent)
            .join("backup")
            .join(role.private_name())
    }

    pub(crate) fn final_path(&self, parent: &Path, member: &PairJournalMember) -> PathBuf {
        parent.join(&member.final_name)
    }

    pub(crate) fn committed(&self, parent: &Path) -> Result<bool, PairArtifactError> {
        self.validate()?;
        for member in &self.members {
            let final_path = self.final_path(parent, member);
            let matches_candidate = match member.candidate.as_ref() {
                Some(candidate) => candidate.matches_path(&final_path)?,
                None => FileIdentity::from_path(&final_path)?.is_none(),
            };
            if !matches_candidate {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(crate) fn rolled_back(&self, parent: &Path) -> Result<bool, PairArtifactError> {
        self.validate()?;
        for member in &self.members {
            let final_path = self.final_path(parent, member);
            let matches_prior = match member.prior.as_ref() {
                Some(prior) => prior.matches_path(&final_path)?,
                None => FileIdentity::from_path(&final_path)?.is_none(),
            };
            if !matches_prior {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

pub(crate) fn journal_path(text: &Path) -> PathBuf {
    append_suffix(text, ".pair.txn.json")
}

pub(crate) fn lock_path(text: &Path) -> PathBuf {
    append_suffix(text, ".pair.lock")
}

fn append_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(suffix);
    PathBuf::from(name)
}

pub(crate) struct PairLock {
    _file: File,
}

impl PairLock {
    pub(crate) fn shared(text: &Path) -> Result<Self, PairArtifactError> {
        Self::acquire(text, FlockOperation::LockShared)
    }

    pub(crate) fn exclusive(text: &Path) -> Result<Self, PairArtifactError> {
        Self::acquire(text, FlockOperation::LockExclusive)
    }

    fn acquire(text: &Path, operation: FlockOperation) -> Result<Self, PairArtifactError> {
        let parent = canonical_parent(text)?;
        let path = lock_path(&parent.join(file_name(text)?));
        let mut created = false;
        let file = match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&path)
        {
            Ok(file) => {
                created = true;
                file
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => OpenOptions::new()
                .read(true)
                .write(true)
                .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
                .open(&path)
                .map_err(|error| io(&path, error))?,
            Err(error) => return Err(io(&path, error)),
        };
        if created {
            file.set_permissions(fs::Permissions::from_mode(0o600))
                .map_err(|error| io(&path, error))?;
            full_sync(&file).map_err(|error| io(&path, error))?;
            sync_directory(&parent)?;
        }
        let metadata = file.metadata().map_err(|error| io(&path, error))?;
        let parent_metadata = fs::metadata(&parent).map_err(|error| io(&parent, error))?;
        if !metadata.is_file()
            || metadata.uid() != rustix::process::geteuid().as_raw()
            || metadata.nlink() != 1
            || metadata.dev() != parent_metadata.dev()
            || metadata.len() != 0
            || metadata.mode() & 0o7777 != 0o600
        {
            return Err(PairArtifactError::Invalid(format!(
                "pair lock is not an owned private regular file: {}",
                path.display()
            )));
        }
        rustix::fs::flock(&file, operation).map_err(|error| {
            io(
                &path,
                std::io::Error::from_raw_os_error(error.raw_os_error()),
            )
        })?;
        let locked_identity = FileIdentity::from_file(&file, &path)?;
        if !locked_identity.matches_path(&path)? {
            return Err(PairArtifactError::Invalid(format!(
                "pair lock path changed while it was being acquired: {}",
                path.display()
            )));
        }
        Ok(Self { _file: file })
    }
}

/// Advisory lock on the current text GGUF inode.
///
/// Writers take this exclusively in addition to the stable sibling lock.
/// Readers use it only when the sibling lock cannot be created/opened (for
/// example, a read-only model mount or a lock owned by another user).
pub(crate) struct PairTextLock {
    _file: File,
}

impl PairTextLock {
    pub(crate) fn shared(text: &Path) -> Result<Self, PairArtifactError> {
        Self::acquire(text, FlockOperation::LockShared)
    }

    pub(crate) fn exclusive(text: &Path) -> Result<Self, PairArtifactError> {
        Self::acquire(text, FlockOperation::LockExclusive)
    }

    pub(crate) fn exclusive_if_present(text: &Path) -> Result<Option<Self>, PairArtifactError> {
        match Self::exclusive(text) {
            Ok(lock) => Ok(Some(lock)),
            Err(PairArtifactError::Io { source, .. })
                if source.kind() == std::io::ErrorKind::NotFound =>
            {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    fn acquire(text: &Path, operation: FlockOperation) -> Result<Self, PairArtifactError> {
        let parent = canonical_parent(text)?;
        let path = parent.join(file_name(text)?);
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&path)
            .map_err(|error| io(&path, error))?;
        let metadata = file.metadata().map_err(|error| io(&path, error))?;
        let parent_metadata = fs::metadata(&parent).map_err(|error| io(&parent, error))?;
        if !metadata.is_file()
            || metadata.nlink() != 1
            || metadata.dev() != parent_metadata.dev()
            || metadata.len() == 0
        {
            return Err(PairArtifactError::Invalid(format!(
                "pair text lock target is not one nonempty same-filesystem regular file: {}",
                path.display()
            )));
        }
        rustix::fs::flock(&file, operation).map_err(|error| {
            io(
                &path,
                std::io::Error::from_raw_os_error(error.raw_os_error()),
            )
        })?;
        let locked_identity = FileIdentity::from_file(&file, &path)?;
        if !locked_identity.matches_path(&path)? {
            return Err(PairArtifactError::Invalid(format!(
                "pair text path changed while its inode was being locked: {}",
                path.display()
            )));
        }
        Ok(Self { _file: file })
    }
}

enum PairReadLock {
    Sibling { _lock: PairLock },
    Text { _lock: PairTextLock },
}

pub(crate) struct PairReadGuard {
    _lock: PairReadLock,
    text: PathBuf,
    projector: PathBuf,
}

impl PairReadGuard {
    pub(crate) fn acquire(text: &Path, projector: &Path) -> Result<Self, PairArtifactError> {
        let lock = match PairLock::shared(text) {
            Ok(lock) => PairReadLock::Sibling { _lock: lock },
            Err(sibling_error) => {
                let text_lock = PairTextLock::shared(text).map_err(|text_error| {
                    PairArtifactError::Invalid(format!(
                        "sibling pair lock unavailable ({sibling_error}); text-inode fallback also failed ({text_error})"
                    ))
                })?;
                tracing::debug!(
                    error = %sibling_error,
                    text = %text.display(),
                    "using shared text-inode lock for read-only or cross-user multimodal pair"
                );
                PairReadLock::Text { _lock: text_lock }
            }
        };
        Ok(Self {
            _lock: lock,
            text: canonical_parent(text)?.join(file_name(text)?),
            projector: canonical_parent(projector)?.join(file_name(projector)?),
        })
    }

    pub(crate) fn validate(
        &self,
        text_gguf: &GgufFile,
        projector_gguf: &GgufFile,
        projector_sha256: &str,
    ) -> Result<(), PairArtifactError> {
        let text_generation = metadata_nonempty(text_gguf, KEY_PAIR_GENERATION);
        let projector_generation = metadata_nonempty(projector_gguf, KEY_PAIR_GENERATION);
        let text_schema = metadata_nonempty(text_gguf, KEY_PAIR_SCHEMA_VERSION);
        let projector_schema = metadata_nonempty(projector_gguf, KEY_PAIR_SCHEMA_VERSION);
        let expected_projector = metadata_nonempty(text_gguf, KEY_MMPROJ_SHA256);
        validate_metadata_values(
            text_generation.as_deref(),
            projector_generation.as_deref(),
            text_schema.as_deref(),
            projector_schema.as_deref(),
            expected_projector.as_deref(),
            projector_sha256,
        )?;
        if text_generation.is_some()
            && canonical_parent(&self.text)? != canonical_parent(&self.projector)?
        {
            return Err(PairArtifactError::Invalid(
                "generation-marked text and projector must share one directory".into(),
            ));
        }

        let journal = journal_path(&self.text);
        if path_entry_exists(&journal)? {
            let parsed = read_pair_journal(&journal)?;
            if text_generation.as_deref() != Some(parsed.transaction_id.as_str())
                || !parsed.committed(&canonical_parent(&self.text)?)?
            {
                return Err(PairArtifactError::Invalid(
                    "incomplete_pair_transaction: recovery is required before loading this pair"
                        .into(),
                ));
            }
        }
        Ok(())
    }
}

pub(crate) fn path_entry_exists(path: &Path) -> Result<bool, PairArtifactError> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(io(path, error)),
    }
}

pub(crate) fn read_pair_journal(path: &Path) -> Result<PairTransactionJournal, PairArtifactError> {
    let parent = canonical_parent(path)?;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .map_err(|error| io(path, error))?;
    let metadata = file.metadata().map_err(|error| io(path, error))?;
    let parent_metadata = fs::metadata(&parent).map_err(|error| io(&parent, error))?;
    if !metadata.is_file()
        || metadata.uid() != rustix::process::geteuid().as_raw()
        || metadata.nlink() != 1
        || metadata.dev() != parent_metadata.dev()
        || metadata.mode() & 0o7777 != 0o600
        || metadata.len() == 0
        || metadata.len() > MAX_PAIR_JOURNAL_BYTES
    {
        return Err(PairArtifactError::Invalid(format!(
            "pair journal is not one bounded owned private regular file: {}",
            path.display()
        )));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_PAIR_JOURNAL_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| io(path, error))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(PairArtifactError::Invalid(
            "pair journal changed while it was being read".into(),
        ));
    }
    let journal: PairTransactionJournal = serde_json::from_slice(&bytes)?;
    journal.validate()?;
    Ok(journal)
}

fn validate_metadata_values(
    text_generation: Option<&str>,
    projector_generation: Option<&str>,
    text_schema: Option<&str>,
    projector_schema: Option<&str>,
    expected_projector: Option<&str>,
    projector_sha256: &str,
) -> Result<(), PairArtifactError> {
    match (text_generation, projector_generation) {
        (None, None) => {
            if text_schema.is_some() || projector_schema.is_some() {
                return Err(PairArtifactError::Invalid(
                    "pair schema metadata exists without a generation".into(),
                ));
            }
        }
        (Some(text_generation), Some(projector_generation)) => {
            validate_transaction_id(text_generation)?;
            if text_generation != projector_generation {
                return Err(PairArtifactError::Invalid(
                    "text and projector generations do not match".into(),
                ));
            }
            if text_schema.as_deref() != Some(PAIR_METADATA_SCHEMA_VERSION)
                || projector_schema.as_deref() != Some(PAIR_METADATA_SCHEMA_VERSION)
            {
                return Err(PairArtifactError::Invalid(
                    "generation-marked pair has a missing or unsupported schema".into(),
                ));
            }
        }
        _ => {
            return Err(PairArtifactError::Invalid(
                "only one GGUF member has pair-generation metadata".into(),
            ));
        }
    }

    if let Some(expected) = expected_projector {
        validate_sha256(expected)?;
        if !expected.eq_ignore_ascii_case(projector_sha256) {
            return Err(PairArtifactError::Invalid(
                "projector digest does not match the text GGUF binding".into(),
            ));
        }
    } else if text_generation.is_some() {
        return Err(PairArtifactError::Invalid(
            "generation-marked text GGUF has no projector digest binding".into(),
        ));
    }

    Ok(())
}

pub(crate) fn canonical_parent(path: &Path) -> Result<PathBuf, PairArtifactError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::canonicalize(parent).map_err(|error| io(parent, error))
}

pub(crate) fn file_name(path: &Path) -> Result<&std::ffi::OsStr, PairArtifactError> {
    path.file_name()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| PairArtifactError::Invalid("pair path has no filename".into()))
}

pub(crate) fn sync_path(path: &Path) -> Result<(), PairArtifactError> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .map_err(|error| io(path, error))?;
    full_sync(&file).map_err(|error| io(path, error))
}

pub(crate) fn sync_directory(path: &Path) -> Result<(), PairArtifactError> {
    let file = File::open(path).map_err(|error| io(path, error))?;
    file.sync_all().map_err(|error| io(path, error))
}

fn full_sync(file: &File) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        rustix::fs::fcntl_fullfsync(file)
            .map_err(|error| std::io::Error::from_raw_os_error(error.raw_os_error()))
    }
    #[cfg(not(target_os = "macos"))]
    {
        file.sync_all()
    }
}

fn metadata_nonempty(gguf: &GgufFile, key: &str) -> Option<String> {
    gguf.metadata_string(key)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn validate_transaction_id(value: &str) -> Result<(), PairArtifactError> {
    let parsed = uuid::Uuid::parse_str(value)
        .map_err(|_| PairArtifactError::Invalid("pair generation is not a UUID".into()))?;
    if parsed.to_string() != value {
        return Err(PairArtifactError::Invalid(
            "pair generation is not canonical lowercase UUID text".into(),
        ));
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), PairArtifactError> {
    if value.len() != 64 || !value.chars().all(|character| character.is_ascii_hexdigit()) {
        return Err(PairArtifactError::Invalid(
            "text GGUF projector binding is not a SHA-256".into(),
        ));
    }
    Ok(())
}

fn validate_single_name(value: &str, what: &str) -> Result<(), PairArtifactError> {
    let path = Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || path.components().count() != 1
        || !matches!(path.components().next(), Some(Component::Normal(_)))
    {
        return Err(PairArtifactError::Invalid(format!(
            "{what} must be one relative filename"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn journal_rejects_traversal_and_requires_text_last() {
        let identity = FileIdentity {
            device: 1,
            inode: 2,
            mode: u32::from(libc::S_IFREG),
            links: 1,
            size: 3,
        };
        let tx = uuid::Uuid::new_v4().to_string();
        let mut journal = PairTransactionJournal {
            schema_version: PAIR_JOURNAL_SCHEMA_VERSION,
            transaction_id: tx.clone(),
            transaction_root: format!(".hf2q-pair-{tx}"),
            members: vec![
                PairJournalMember {
                    role: PairMemberRole::Projector,
                    final_name: "model-mmproj.gguf".into(),
                    prior: None,
                    candidate: Some(identity.clone()),
                },
                PairJournalMember {
                    role: PairMemberRole::Text,
                    final_name: "model.gguf".into(),
                    prior: None,
                    candidate: Some(identity),
                },
            ],
        };
        journal.validate().unwrap();
        journal.members[0].final_name = "../escape".into();
        assert!(journal.validate().is_err());
        journal.members[0].final_name = "model-mmproj.gguf".into();
        journal.members.swap(0, 1);
        assert!(journal.validate().is_err());
    }

    #[test]
    fn lock_and_journal_paths_are_text_siblings() {
        assert_eq!(
            lock_path(Path::new("/models/model.gguf")),
            PathBuf::from("/models/model.gguf.pair.lock")
        );
        assert_eq!(
            journal_path(Path::new("/models/model.gguf")),
            PathBuf::from("/models/model.gguf.pair.txn.json")
        );
    }

    #[test]
    fn legacy_local_pair_digest_is_enforced_without_remote_provenance() {
        let expected = "a".repeat(64);
        validate_metadata_values(None, None, None, None, Some(&expected), &expected).unwrap();
        assert!(
            validate_metadata_values(None, None, None, None, Some(&expected), &"b".repeat(64),)
                .is_err()
        );
    }

    #[test]
    fn generation_marked_pair_requires_both_matching_members_and_schema() {
        let generation = uuid::Uuid::new_v4().to_string();
        let digest = "c".repeat(64);
        validate_metadata_values(
            Some(&generation),
            Some(&generation),
            Some(PAIR_METADATA_SCHEMA_VERSION),
            Some(PAIR_METADATA_SCHEMA_VERSION),
            Some(&digest),
            &digest,
        )
        .unwrap();
        assert!(validate_metadata_values(
            Some(&generation),
            None,
            Some(PAIR_METADATA_SCHEMA_VERSION),
            None,
            Some(&digest),
            &digest,
        )
        .is_err());
    }

    #[test]
    fn reader_falls_back_to_text_inode_when_sibling_lock_is_unusable() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let text = dir.path().join("model.gguf");
        let projector = dir.path().join("model-mmproj.gguf");
        fs::write(&text, b"text").unwrap();
        fs::write(&projector, b"projector").unwrap();
        symlink(dir.path().join("untrusted-lock-target"), lock_path(&text)).unwrap();

        let guard = PairReadGuard::acquire(&text, &projector).unwrap();

        assert!(matches!(guard._lock, PairReadLock::Text { .. }));
    }

    #[test]
    fn exclusive_writer_text_lock_waits_for_fallback_reader() {
        use std::sync::mpsc;
        use std::time::Duration;

        let dir = tempfile::tempdir().unwrap();
        let text = dir.path().join("model.gguf");
        fs::write(&text, b"text").unwrap();
        let reader = PairTextLock::shared(&text).unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (acquired_tx, acquired_rx) = mpsc::channel();
        let writer_text = text.clone();
        let writer = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            let lock = PairTextLock::exclusive_if_present(&writer_text)
                .unwrap()
                .unwrap();
            acquired_tx.send(()).unwrap();
            drop(lock);
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(acquired_rx
            .recv_timeout(Duration::from_millis(100))
            .is_err());
        drop(reader);
        acquired_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        writer.join().unwrap();
    }

    #[test]
    fn exclusive_candidate_lock_follows_text_inode_through_rename() {
        use std::sync::mpsc;
        use std::time::Duration;

        let dir = tempfile::tempdir().unwrap();
        let staged = dir.path().join("staged.gguf");
        let final_path = dir.path().join("model.gguf");
        fs::write(&staged, b"candidate").unwrap();
        let candidate_lock = PairTextLock::exclusive(&staged).unwrap();
        fs::rename(&staged, &final_path).unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (acquired_tx, acquired_rx) = mpsc::channel();
        let reader_path = final_path.clone();
        let reader = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            let lock = PairTextLock::shared(&reader_path).unwrap();
            acquired_tx.send(()).unwrap();
            drop(lock);
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(acquired_rx
            .recv_timeout(Duration::from_millis(100))
            .is_err());
        drop(candidate_lock);
        acquired_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        reader.join().unwrap();
    }
}
