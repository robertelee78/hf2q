//! Crash-consistency experiment used only by the spike.

mod file;
mod schema;
mod unix;

#[cfg(test)]
mod tests;

use std::fs::File;
use std::path::{Path, PathBuf};

use jiff::Timestamp;
use thiserror::Error;

use crate::candidates::VerifiedMetadataEvidence;
use crate::model::CapturedRole;

use self::schema::{GenerationReceiptV0, SelectorV0};
use self::unix::{Directory, EntryIdentity};

const UPDATE: &str = "update";
const METADATA: &str = "metadata";
const GENERATIONS: &str = "generations";
const CURRENT: &str = "current.json";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Barrier {
    PendingDirectory,
    RootChainFiles,
    RootChainSync,
    TrustedRootBeforeFile,
    TrustedRootFile,
    TimestampFile,
    SnapshotFile,
    TargetsFile,
    ReceiptFile,
    PendingDirectorySync,
    GenerationPublish,
    GenerationsSync,
    SelectorFile,
    MetadataPrecommitSync,
    SelectorCommit,
    SelectorFullSync,
    GenerationPostcommitSync,
    MetadataPostcommitSync,
    UpdatePostcommitSync,
    RootPostcommitSync,
}

pub(crate) const ALL_BARRIERS: &[Barrier] = &[
    Barrier::PendingDirectory,
    Barrier::RootChainFiles,
    Barrier::RootChainSync,
    Barrier::TrustedRootBeforeFile,
    Barrier::TrustedRootFile,
    Barrier::TimestampFile,
    Barrier::SnapshotFile,
    Barrier::TargetsFile,
    Barrier::ReceiptFile,
    Barrier::PendingDirectorySync,
    Barrier::GenerationPublish,
    Barrier::GenerationsSync,
    Barrier::SelectorFile,
    Barrier::MetadataPrecommitSync,
    Barrier::SelectorCommit,
    Barrier::SelectorFullSync,
    Barrier::GenerationPostcommitSync,
    Barrier::MetadataPostcommitSync,
    Barrier::UpdatePostcommitSync,
    Barrier::RootPostcommitSync,
];

impl Barrier {
    fn name(self) -> &'static str {
        match self {
            Self::PendingDirectory => "pending-directory",
            Self::RootChainFiles => "root-chain-files",
            Self::RootChainSync => "root-chain-sync",
            Self::TrustedRootBeforeFile => "trusted-root-before-file",
            Self::TrustedRootFile => "trusted-root-file",
            Self::TimestampFile => "timestamp-file",
            Self::SnapshotFile => "snapshot-file",
            Self::TargetsFile => "targets-file",
            Self::ReceiptFile => "receipt-file",
            Self::PendingDirectorySync => "pending-directory-sync",
            Self::GenerationPublish => "generation-publish",
            Self::GenerationsSync => "generations-sync",
            Self::SelectorFile => "selector-file",
            Self::MetadataPrecommitSync => "metadata-precommit-sync",
            Self::SelectorCommit => "selector-commit",
            Self::SelectorFullSync => "selector-full-sync",
            Self::GenerationPostcommitSync => "generation-postcommit-sync",
            Self::MetadataPostcommitSync => "metadata-postcommit-sync",
            Self::UpdatePostcommitSync => "update-postcommit-sync",
            Self::RootPostcommitSync => "root-postcommit-sync",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        ALL_BARRIERS
            .iter()
            .copied()
            .find(|barrier| barrier.name() == value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FaultAction {
    ReturnError,
    AbortProcess,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HistoryMode {
    Authority,
    LockedRecovery,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FaultPlan {
    pub(crate) barrier: Option<Barrier>,
    pub(crate) action: Option<FaultAction>,
}

impl FaultPlan {
    fn trip(self, barrier: Barrier) -> Result<(), JournalError> {
        if self.barrier != Some(barrier) {
            return Ok(());
        }
        match self.action.unwrap_or(FaultAction::ReturnError) {
            FaultAction::ReturnError => Err(JournalError::Injected(barrier)),
            FaultAction::AbortProcess => std::process::abort(),
        }
    }
}

#[derive(Debug, Error)]
pub(crate) enum JournalError {
    #[error("another update or installation transition is active")]
    Busy,
    #[error("invalid experimental metadata journal: {0}")]
    Invalid(&'static str),
    #[error("required experimental metadata journal entry is missing")]
    Missing,
    #[error("injected failure at {0:?}")]
    Injected(Barrier),
    #[error("journal operation failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("generation {sequence} may be committed but its durability is unknown")]
    CommittedDurabilityUnknown {
        sequence: u64,
        #[source]
        source: Box<JournalError>,
    },
}

impl JournalError {
    fn errno(error: rustix::io::Errno) -> Self {
        Self::Io(std::io::Error::from_raw_os_error(error.raw_os_error()))
    }

    fn after_commit(self, sequence: u64) -> Self {
        Self::CommittedDurabilityUnknown {
            sequence,
            source: Box::new(self),
        }
    }
}

#[derive(Clone, Debug)]
struct CandidateGeneration {
    repository: String,
    channel: String,
    update_start: Timestamp,
    prior_root: CapturedRole,
    root_chain: Vec<CapturedRole>,
    root: CapturedRole,
    timestamp: CapturedRole,
    snapshot: CapturedRole,
    targets: CapturedRole,
}

impl CandidateGeneration {
    fn from_verified(evidence: &VerifiedMetadataEvidence) -> Self {
        Self {
            repository: evidence.repository().to_string(),
            channel: evidence.channel().to_string(),
            update_start: evidence.update_start(),
            prior_root: evidence.prior_root().clone(),
            root_chain: evidence.root_chain().to_vec(),
            root: evidence.root().clone(),
            timestamp: evidence.timestamp().clone(),
            snapshot: evidence.snapshot().clone(),
            targets: evidence.targets().clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CommitOutcome {
    Committed { sequence: u64 },
    AlreadyCommitted { sequence: u64 },
}

struct LockedJournal {
    root_path: PathBuf,
    root: Directory,
    update: Directory,
    metadata: Directory,
    generations: Directory,
    _lock: File,
    lock_identity: EntryIdentity,
}

struct LiveNamespace {
    root: Directory,
    update: Directory,
    metadata: Directory,
    generations: Directory,
}

pub(crate) fn commit_candidate(
    root_path: &Path,
    evidence: &VerifiedMetadataEvidence,
    faults: FaultPlan,
) -> Result<CommitOutcome, JournalError> {
    let journal = LockedJournal::open(root_path)?;
    let candidate = CandidateGeneration::from_verified(evidence);
    journal.commit(&candidate, faults)
}

pub(crate) fn read_selected_sequence(root_path: &Path) -> Result<Option<u64>, JournalError> {
    let root = unix::open_existing_root(root_path)?;
    let update = match unix::open_directory(&root, UPDATE) {
        Ok(directory) => directory,
        Err(JournalError::Missing) => return Ok(None),
        Err(error) => return Err(error),
    };
    let metadata = match unix::open_directory(&update, METADATA) {
        Ok(directory) => directory,
        Err(JournalError::Missing) => return Ok(None),
        Err(error) => return Err(error),
    };
    let generations = unix::open_directory(&metadata, GENERATIONS)?;
    read_selector(&metadata, &generations).map(|selector| selector.map(|item| item.sequence))
}

impl LockedJournal {
    fn open(root_path: &Path) -> Result<Self, JournalError> {
        let root = unix::open_existing_root(root_path)?;
        let update = unix::ensure_private_directory(&root, UPDATE)?;
        let (lock, lock_identity) = unix::acquire_nonblocking_lock(&update)?;
        let metadata = unix::ensure_private_directory(&update, METADATA)?;
        let generations = unix::ensure_private_directory(&metadata, GENERATIONS)?;
        Ok(Self {
            root_path: root_path.to_owned(),
            root,
            update,
            metadata,
            generations,
            _lock: lock,
            lock_identity,
        })
    }

    fn commit(
        &self,
        candidate: &CandidateGeneration,
        faults: FaultPlan,
    ) -> Result<CommitOutcome, JournalError> {
        let prior = read_selector_for_recovery(&self.metadata, &self.generations)?;
        let prior_receipt = prior
            .as_ref()
            .map(|selector| read_generation_receipt(&self.generations, selector))
            .transpose()?;
        let published_successor = prior
            .as_ref()
            .map(|selector| {
                selector
                    .sequence
                    .checked_add(1)
                    .ok_or(JournalError::Invalid("generation sequence overflow"))
            })
            .transpose()?
            .map(|sequence| {
                unix::entry_identity(&self.generations, &format!("{sequence:020}"))
                    .map(|identity| identity.is_some())
            })
            .transpose()?
            .unwrap_or(false);
        if let Some(selector) = &prior {
            // A complete successor is recovery evidence, not permission to
            // reaffirm the older selected generation. Only an exact retry of
            // that successor may reconstruct the missing selector below.
            if !published_successor
                && generation_matches_candidate(&self.generations, selector.sequence, candidate)?
            {
                return self.repair_committed(selector, faults);
            }
        }
        let sequence = prior.as_ref().map_or(Ok(1), |selector| {
            selector
                .sequence
                .checked_add(1)
                .ok_or(JournalError::Invalid("generation sequence overflow"))
        })?;
        let predecessor = prior
            .as_ref()
            .map(|selector| selector.generation_sha256.clone());
        let receipt = GenerationReceiptV0::new(sequence, predecessor, candidate)?;
        if let (Some(selector), Some(prior_receipt)) = (&prior, &prior_receipt) {
            receipt.validate_successor(prior_receipt, &selector.generation_sha256)?;
        }
        let receipt_bytes = receipt.to_bytes()?;
        let generation_sha256 = hex::encode(crate::model::sha256(&receipt_bytes));
        let pending_name = format!(".pending-{sequence:020}");
        let generation_name = format!("{sequence:020}");
        if unix::entry_identity(&self.generations, &generation_name)?.is_none() {
            let pending = unix::ensure_private_directory(&self.generations, &pending_name)?;
            faults.trip(Barrier::PendingDirectory)?;
            write_generation_files(&pending, candidate, &receipt_bytes, faults)?;
            verify_generation(&pending, &receipt, candidate)?;
            unix::sync_directory(&pending)?;
            faults.trip(Barrier::PendingDirectorySync)?;
            unix::rename_noreplace(
                &self.generations,
                &pending_name,
                &self.generations,
                &generation_name,
            )?;
        } else {
            let published = unix::open_directory(&self.generations, &generation_name)?;
            verify_generation(&published, &receipt, candidate)?;
            if unix::entry_identity(&self.generations, &pending_name)?.is_some() {
                return Err(JournalError::Invalid(
                    "matching published generation coexists with pending state",
                ));
            }
        }
        faults.trip(Barrier::GenerationPublish)?;
        unix::sync_directory(&self.generations)?;
        faults.trip(Barrier::GenerationsSync)?;
        let published = unix::open_directory(&self.generations, &generation_name)?;
        verify_generation(&published, &receipt, candidate)?;

        let selector = SelectorV0::new(sequence, generation_sha256);
        let selector_bytes = selector.to_bytes()?;
        let pending_selector = format!(".current-{sequence:020}.json");
        let selector_file =
            file::write_or_resume_private_file(&self.metadata, &pending_selector, &selector_bytes)?;
        unix::full_sync_file(&selector_file)?;
        let selector_identity =
            unix::regular_file_identity(&selector_file, self.metadata.device())?;
        faults.trip(Barrier::SelectorFile)?;
        unix::sync_directory(&self.metadata)?;
        faults.trip(Barrier::MetadataPrecommitSync)?;
        let live = self.reopen_precommit(
            &generation_name,
            &published,
            &pending_selector,
            selector_identity,
            &selector_bytes,
            &receipt,
            candidate,
            prior.as_ref(),
        )?;
        if let Some(expected) = &prior {
            let current = file::read_private_file(&live.metadata, CURRENT, 16 * 1024)?;
            if current.bytes != expected.to_bytes()? {
                return Err(JournalError::Invalid(
                    "current selector changed before commit",
                ));
            }
            unix::rename_replace(&live.metadata, &pending_selector, CURRENT)?;
        } else {
            unix::rename_noreplace(&live.metadata, &pending_selector, &live.metadata, CURRENT)?;
        }
        faults
            .trip(Barrier::SelectorCommit)
            .map_err(|error| error.after_commit(sequence))?;
        self.postcommit(&selector, &published, faults)
    }

    fn repair_committed(
        &self,
        selector: &SelectorV0,
        faults: FaultPlan,
    ) -> Result<CommitOutcome, JournalError> {
        let generation =
            unix::open_directory(&self.generations, &format!("{:020}", selector.sequence))?;
        self.postcommit(selector, &generation, faults)
            .map(|_| CommitOutcome::AlreadyCommitted {
                sequence: selector.sequence,
            })
    }

    fn postcommit(
        &self,
        expected: &SelectorV0,
        expected_generation: &Directory,
        faults: FaultPlan,
    ) -> Result<CommitOutcome, JournalError> {
        let sequence = expected.sequence;
        let repair = || -> Result<(), JournalError> {
            let live = self.reopen_live_namespace()?;
            if read_selector(&live.metadata, &live.generations)?.as_ref() != Some(expected) {
                return Err(JournalError::Invalid(
                    "live selected history does not match committed generation",
                ));
            }
            let selector = file::read_private_file(&live.metadata, CURRENT, 16 * 1024)?;
            if selector.bytes != expected.to_bytes()? {
                return Err(JournalError::Invalid(
                    "live current selector does not match the committed generation",
                ));
            }
            if unix::entry_identity(&live.metadata, &format!(".current-{sequence:020}.json"))?
                .is_some()
            {
                return Err(JournalError::Invalid(
                    "committed selector coexists with pending selector state",
                ));
            }
            let generation = unix::open_directory(&live.generations, &format!("{sequence:020}"))?;
            require_same_directory(&generation, expected_generation)?;
            let receipt = read_generation_receipt(&live.generations, expected)?;
            verify_generation_receipt(&generation, &receipt)?;
            unix::full_sync_file(&selector.file)?;
            faults.trip(Barrier::SelectorFullSync)?;
            unix::sync_directory(&generation)?;
            faults.trip(Barrier::GenerationPostcommitSync)?;
            unix::sync_directory(&live.generations)?;
            unix::sync_directory(&live.metadata)?;
            faults.trip(Barrier::MetadataPostcommitSync)?;
            unix::sync_directory(&live.update)?;
            faults.trip(Barrier::UpdatePostcommitSync)?;
            unix::sync_directory(&live.root)?;
            faults.trip(Barrier::RootPostcommitSync)
        };
        repair()
            .map(|_| CommitOutcome::Committed { sequence })
            .map_err(|error| error.after_commit(sequence))
    }

    fn reopen_live_namespace(&self) -> Result<LiveNamespace, JournalError> {
        let root = unix::open_existing_root(&self.root_path)?;
        require_same_directory(&root, &self.root)?;
        let update = unix::open_directory(&root, UPDATE)?;
        require_same_directory(&update, &self.update)?;
        unix::verify_named_identity(&update, "install.lock", self.lock_identity)?;
        let metadata = unix::open_directory(&update, METADATA)?;
        require_same_directory(&metadata, &self.metadata)?;
        let generations = unix::open_directory(&metadata, GENERATIONS)?;
        require_same_directory(&generations, &self.generations)?;
        Ok(LiveNamespace {
            root,
            update,
            metadata,
            generations,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn reopen_precommit(
        &self,
        generation_name: &str,
        expected_generation: &Directory,
        pending_selector: &str,
        selector_identity: EntryIdentity,
        selector_bytes: &[u8],
        receipt: &GenerationReceiptV0,
        candidate: &CandidateGeneration,
        prior: Option<&SelectorV0>,
    ) -> Result<LiveNamespace, JournalError> {
        let live = self.reopen_live_namespace()?;
        verify_history(
            &live.metadata,
            &live.generations,
            prior,
            HistoryMode::LockedRecovery,
        )?;
        let generation = unix::open_directory(&live.generations, generation_name)?;
        require_same_directory(&generation, expected_generation)?;
        verify_generation(&generation, receipt, candidate)?;
        let staged = file::read_private_file(&live.metadata, pending_selector, 16 * 1024)?;
        unix::verify_named_identity(&live.metadata, pending_selector, selector_identity)?;
        if staged.bytes != selector_bytes
            || unix::regular_file_identity(&staged.file, live.metadata.device())?
                != selector_identity
        {
            return Err(JournalError::Invalid(
                "pending selector changed before commit",
            ));
        }
        Ok(live)
    }
}

fn require_same_directory(actual: &Directory, expected: &Directory) -> Result<(), JournalError> {
    if !actual.same_object(expected) {
        return Err(JournalError::Invalid(
            "journal namespace changed after verification",
        ));
    }
    Ok(())
}

fn write_generation_files(
    directory: &Directory,
    candidate: &CandidateGeneration,
    receipt: &[u8],
    faults: FaultPlan,
) -> Result<(), JournalError> {
    let root_chain = unix::ensure_private_directory(directory, "root-chain")?;
    for root in &candidate.root_chain {
        let name = format!("{:020}.root.json", root.version);
        let file = file::write_or_resume_private_file(&root_chain, &name, &root.raw)?;
        unix::full_sync_file(&file)?;
    }
    faults.trip(Barrier::RootChainFiles)?;
    unix::sync_directory(&root_chain)?;
    faults.trip(Barrier::RootChainSync)?;
    for (name, bytes, barrier) in [
        (
            "trusted-root-before.json",
            candidate.prior_root.raw.as_slice(),
            Barrier::TrustedRootBeforeFile,
        ),
        (
            "trusted-root.json",
            candidate.root.raw.as_slice(),
            Barrier::TrustedRootFile,
        ),
        (
            "timestamp.json",
            candidate.timestamp.raw.as_slice(),
            Barrier::TimestampFile,
        ),
        (
            "snapshot.json",
            candidate.snapshot.raw.as_slice(),
            Barrier::SnapshotFile,
        ),
        (
            "targets.json",
            candidate.targets.raw.as_slice(),
            Barrier::TargetsFile,
        ),
        ("generation.json", receipt, Barrier::ReceiptFile),
    ] {
        let file = file::write_or_resume_private_file(directory, name, bytes)?;
        unix::full_sync_file(&file)?;
        faults.trip(barrier)?;
    }
    Ok(())
}

fn read_selector(
    metadata: &Directory,
    generations: &Directory,
) -> Result<Option<SelectorV0>, JournalError> {
    read_selector_with_mode(metadata, generations, HistoryMode::Authority)
}

fn read_selector_for_recovery(
    metadata: &Directory,
    generations: &Directory,
) -> Result<Option<SelectorV0>, JournalError> {
    read_selector_with_mode(metadata, generations, HistoryMode::LockedRecovery)
}

fn read_selector_with_mode(
    metadata: &Directory,
    generations: &Directory,
    mode: HistoryMode,
) -> Result<Option<SelectorV0>, JournalError> {
    let selector = unix::entry_identity(metadata, CURRENT)?
        .map(|_| {
            let selected = file::read_private_file(metadata, CURRENT, 16 * 1024)?;
            SelectorV0::parse(&selected.bytes)
        })
        .transpose()?;
    verify_history(metadata, generations, selector.as_ref(), mode)?;
    Ok(selector)
}

fn verify_history(
    metadata: &Directory,
    generations: &Directory,
    selected: Option<&SelectorV0>,
    mode: HistoryMode,
) -> Result<(), JournalError> {
    let next = selected.map_or(Ok(1_u64), |selector| {
        selector
            .sequence
            .checked_add(1)
            .ok_or(JournalError::Invalid("generation sequence overflow"))
    })?;
    let pending_generation = format!(".pending-{next:020}");
    let next_generation = format!("{next:020}");
    let pending_selector = format!(".current-{next:020}.json");

    let metadata_names = unix::list_names(metadata)?;
    let mut expected_metadata = std::collections::BTreeSet::from([GENERATIONS.to_string()]);
    if selected.is_some() {
        expected_metadata.insert(CURRENT.to_string());
    }
    let has_pending_selector = metadata_names.contains(&pending_selector);
    if has_pending_selector {
        expected_metadata.insert(pending_selector.clone());
    }
    if metadata_names != expected_metadata {
        return Err(JournalError::Invalid("metadata inventory is not exact"));
    }

    let generation_names = unix::list_names(generations)?;
    let has_pending_generation = generation_names.contains(&pending_generation);
    let has_next_generation = generation_names.contains(&next_generation);
    if has_pending_generation && has_next_generation {
        return Err(JournalError::Invalid(
            "pending and published next generation coexist",
        ));
    }
    if has_pending_selector && !has_next_generation {
        return Err(JournalError::Invalid(
            "pending selector has no complete published generation",
        ));
    }
    if mode == HistoryMode::Authority && has_next_generation && !has_pending_selector {
        return Err(JournalError::Invalid(
            "published next generation is ambiguous without its pending selector",
        ));
    }
    let selected_sequence = selected.map_or(0, |selector| selector.sequence);
    let mut expected_generations = std::collections::BTreeSet::new();
    for sequence in 1..=selected_sequence {
        expected_generations.insert(format!("{sequence:020}"));
    }
    if has_pending_generation {
        expected_generations.insert(pending_generation.clone());
    }
    if has_next_generation {
        expected_generations.insert(next_generation.clone());
    }
    if generation_names != expected_generations {
        return Err(JournalError::Invalid(
            "generation history inventory is not exact",
        ));
    }
    if has_pending_generation {
        let pending = unix::open_directory(generations, &pending_generation)?;
        verify_pending_generation_shape(&pending)?;
    }
    if has_pending_selector {
        let _ = file::read_private_file(metadata, &pending_selector, 16 * 1024)?;
    }

    let final_sequence = selected_sequence
        .checked_add(u64::from(has_next_generation))
        .ok_or(JournalError::Invalid("generation history overflow"))?;
    let mut prior: Option<(GenerationReceiptV0, String)> = None;
    for sequence in 1..=final_sequence {
        let generation = unix::open_directory(generations, &format!("{sequence:020}"))?;
        let stored = file::read_private_file(&generation, "generation.json", 64 * 1024)?;
        let digest = hex::encode(crate::model::sha256(&stored.bytes));
        let receipt = GenerationReceiptV0::parse(&stored.bytes)?;
        if receipt.sequence() != sequence {
            return Err(JournalError::Invalid(
                "generation directory and receipt sequence disagree",
            ));
        }
        verify_generation_receipt(&generation, &receipt)?;
        if let Some((prior_receipt, prior_digest)) = &prior {
            receipt.validate_successor(prior_receipt, prior_digest)?;
        }
        if selected.is_some_and(|selector| selector.sequence == sequence)
            && selected.is_some_and(|selector| selector.generation_sha256 != digest)
        {
            return Err(JournalError::Invalid(
                "selector digest does not match generation receipt",
            ));
        }
        prior = Some((receipt, digest));
    }
    Ok(())
}

fn verify_pending_generation_shape(directory: &Directory) -> Result<(), JournalError> {
    let names = unix::list_names(directory)?;
    let allowed = std::collections::BTreeSet::from([
        "generation.json".to_string(),
        "root-chain".to_string(),
        "snapshot.json".to_string(),
        "targets.json".to_string(),
        "timestamp.json".to_string(),
        "trusted-root-before.json".to_string(),
        "trusted-root.json".to_string(),
    ]);
    if !names.is_subset(&allowed) {
        return Err(JournalError::Invalid(
            "pending generation contains unexpected state",
        ));
    }
    if names.contains("root-chain") {
        let root_chain = unix::open_directory(directory, "root-chain")?;
        let root_names = unix::list_names(&root_chain)?;
        if root_names.len() > 32
            || root_names.iter().any(|name| {
                name.len() != "00000000000000000001.root.json".len()
                    || !name.ends_with(".root.json")
                    || !name[..20].bytes().all(|byte| byte.is_ascii_digit())
            })
        {
            return Err(JournalError::Invalid(
                "pending root chain inventory is invalid",
            ));
        }
    }
    Ok(())
}

fn read_generation_receipt(
    generations: &Directory,
    selector: &SelectorV0,
) -> Result<GenerationReceiptV0, JournalError> {
    let generation = unix::open_directory(generations, &format!("{:020}", selector.sequence))?;
    let receipt = file::read_private_file(&generation, "generation.json", 64 * 1024)?;
    if hex::encode(crate::model::sha256(&receipt.bytes)) != selector.generation_sha256 {
        return Err(JournalError::Invalid(
            "selector digest does not match prior generation receipt",
        ));
    }
    GenerationReceiptV0::parse(&receipt.bytes)
}

fn generation_matches_candidate(
    generations: &Directory,
    sequence: u64,
    candidate: &CandidateGeneration,
) -> Result<bool, JournalError> {
    let generation = unix::open_directory(generations, &format!("{sequence:020}"))?;
    let receipt = file::read_private_file(&generation, "generation.json", 64 * 1024)?;
    let receipt = GenerationReceiptV0::parse(&receipt.bytes)?;
    if !receipt.matches_candidate(candidate) {
        return Ok(false);
    }
    verify_generation(&generation, &receipt, candidate).map(|_| true)
}

fn verify_generation(
    directory: &Directory,
    receipt: &GenerationReceiptV0,
    candidate: &CandidateGeneration,
) -> Result<(), JournalError> {
    verify_generation_receipt(directory, receipt)?;
    receipt.validate_candidate(candidate)
}

fn verify_generation_receipt(
    directory: &Directory,
    receipt: &GenerationReceiptV0,
) -> Result<(), JournalError> {
    let expected = std::collections::BTreeSet::from([
        "generation.json".to_string(),
        "root-chain".to_string(),
        "snapshot.json".to_string(),
        "targets.json".to_string(),
        "timestamp.json".to_string(),
        "trusted-root-before.json".to_string(),
        "trusted-root.json".to_string(),
    ]);
    if unix::list_names(directory)? != expected {
        return Err(JournalError::Invalid("generation inventory is not exact"));
    }
    let stored_receipt = file::read_private_file(directory, "generation.json", 64 * 1024)?;
    if stored_receipt.bytes != receipt.to_bytes()? {
        return Err(JournalError::Invalid(
            "stored generation receipt changed after parsing",
        ));
    }
    let root_chain = unix::open_directory(directory, "root-chain")?;
    let expected_roots = receipt.expected_root_chain_names().into_iter().collect();
    if unix::list_names(&root_chain)? != expected_roots {
        return Err(JournalError::Invalid("root chain inventory is not exact"));
    }
    for name in receipt.expected_root_chain_names() {
        let actual =
            file::read_private_file(&root_chain, &name, receipt.stored_root_limit(&name)?)?;
        receipt.validate_stored_root(&name, &actual.bytes)?;
    }
    for name in [
        "trusted-root-before.json",
        "trusted-root.json",
        "timestamp.json",
        "snapshot.json",
        "targets.json",
    ] {
        let actual = file::read_private_file(directory, name, receipt.stored_role_limit(name)?)?;
        receipt.validate_stored_role(name, &actual.bytes)?;
    }
    Ok(())
}
