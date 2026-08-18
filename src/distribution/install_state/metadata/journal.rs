use sha2::{Digest, Sha256};

use super::schema::{MetadataGenerationReceiptV1, MetadataSelectorV1, MAX_SELECTOR_BYTES};
use super::{
    MetadataCommitOutcome, MetadataJournalError, MetadataStateAuthorization,
    VerifiedMetadataCandidate,
};
use crate::distribution::install_state::file;
use crate::distribution::install_state::locked::LockedInstallation;
use crate::distribution::install_state::unix::{self, Directory, EntryIdentity};

mod cleanup;
mod fault;
mod validation;

use cleanup::{remove_generation, verify_prune_prefix};
#[cfg(not(test))]
use fault::FaultPlan;
use fault::{trip, TestBarrier};
#[cfg(test)]
pub(super) use fault::{Barrier, FaultPlan};
#[cfg(test)]
pub(super) use validation::read_selected;
use validation::{
    generation_matches_candidate, read_receipt, read_receipt_from_directory,
    read_selector_with_mode, require_same_directory, verify_generation, write_generation,
};

const METADATA: &str = "metadata";
const GENERATIONS: &str = "generations";
const CURRENT: &str = "current.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HistoryMode {
    Authority,
    LockedRecovery,
}

pub(super) fn commit_candidate(
    authorization: MetadataStateAuthorization,
    candidate: VerifiedMetadataCandidate,
) -> Result<MetadataCommitOutcome, MetadataJournalError> {
    commit_candidate_with_faults(authorization, candidate, FaultPlan::default())
}

fn commit_candidate_with_faults(
    authorization: MetadataStateAuthorization,
    candidate: VerifiedMetadataCandidate,
    faults: FaultPlan,
) -> Result<MetadataCommitOutcome, MetadataJournalError> {
    if authorization.root.canonical.as_str() != candidate.state_root
        || authorization.installation_id != candidate.installation_id
    {
        return Err(MetadataJournalError::Invalid(
            "candidate identity differs from the explicit metadata-state authorization",
        ));
    }
    let locked = LockedInstallation::acquire(&authorization.root.path)?;
    LockedMetadataJournal::open(
        locked,
        authorization.installation_id,
        candidate.state_root.clone(),
    )?
    .commit(&candidate, faults)
}

struct LockedMetadataJournal {
    locked: LockedInstallation,
    metadata: Directory,
    generations: Directory,
    installation_id: String,
    state_root: String,
}

struct LiveNamespace {
    root: Directory,
    update: Directory,
    metadata: Directory,
    generations: Directory,
}

impl LockedMetadataJournal {
    fn open(
        locked: LockedInstallation,
        installation_id: String,
        state_root: String,
    ) -> Result<Self, MetadataJournalError> {
        let metadata = unix::ensure_private_directory(locked.update(), METADATA)?;
        let generations = unix::ensure_private_directory(&metadata, GENERATIONS)?;
        Ok(Self {
            locked,
            metadata,
            generations,
            installation_id,
            state_root,
        })
    }

    fn commit(
        &self,
        candidate: &VerifiedMetadataCandidate,
        faults: FaultPlan,
    ) -> Result<MetadataCommitOutcome, MetadataJournalError> {
        self.commit_with_precommit_hook(candidate, faults, || {})
    }

    fn commit_with_precommit_hook(
        &self,
        candidate: &VerifiedMetadataCandidate,
        faults: FaultPlan,
        before_precommit_reopen: impl FnOnce(),
    ) -> Result<MetadataCommitOutcome, MetadataJournalError> {
        let prior = read_selector_with_mode(
            &self.metadata,
            &self.generations,
            HistoryMode::LockedRecovery,
        )?;
        let prior_receipt = prior
            .as_ref()
            .map(|selector| read_receipt(&self.generations, selector))
            .transpose()?;
        if let Some(receipt) = &prior_receipt {
            receipt.validate_state_identity(&self.installation_id, &self.state_root)?;
        }

        if let Some(selector) = &prior {
            // A selected generation must be reverified and its commit
            // barriers repaired before deletion authority can touch its
            // predecessor. Otherwise a later power-loss rollback of the
            // selector could point at a predecessor that cleanup already
            // removed. Fault injection for the new transaction must not
            // intercept this independent recovery pass.
            self.repeat_postcommit_barriers(selector, FaultPlan::default())?;
            // A selected generation's predecessor cleanup is independent of
            // the next network candidate and must finish before any successor
            // transaction begins. Otherwise residue N-1 would become
            // unrecognizable after selecting N+1.
            self.finish_predecessor_cleanup(selector, FaultPlan::default())
                .map_err(|error| error.after_commit(selector.sequence()))?;
            let next = selector
                .sequence()
                .checked_add(1)
                .ok_or(MetadataJournalError::Invalid(
                    "metadata generation sequence overflowed",
                ))?;
            let transaction_exists =
                unix::entry_identity(&self.generations, &format!(".pending-{next:020}"))?.is_some()
                    || unix::entry_identity(&self.generations, &format!("{next:020}"))?.is_some()
                    || unix::entry_identity(&self.metadata, &format!(".current-{next:020}.json"))?
                        .is_some();
            if !transaction_exists
                && generation_matches_candidate(&self.generations, selector.sequence(), candidate)?
            {
                return Ok(MetadataCommitOutcome::AlreadyCommitted {
                    sequence: selector.sequence(),
                });
            }
        }

        let sequence = prior.as_ref().map_or(Ok(1), |selector| {
            selector
                .sequence()
                .checked_add(1)
                .ok_or(MetadataJournalError::Invalid(
                    "metadata generation sequence overflowed",
                ))
        })?;
        let predecessor = prior
            .as_ref()
            .map(|selector| selector.generation_sha256().to_owned());
        let receipt = MetadataGenerationReceiptV1::new(sequence, predecessor, candidate)?;
        if let (Some(selector), Some(prior_receipt)) = (&prior, &prior_receipt) {
            receipt.validate_successor(prior_receipt, selector.generation_sha256())?;
        }
        let receipt_bytes = receipt.to_bytes()?;
        let receipt_digest = hex::encode(Sha256::digest(&receipt_bytes));
        let pending_name = format!(".pending-{sequence:020}");
        let generation_name = format!("{sequence:020}");

        let published = if unix::entry_identity(&self.generations, &generation_name)?.is_some() {
            if unix::entry_identity(&self.generations, &pending_name)?.is_some() {
                return Err(MetadataJournalError::Invalid(
                    "published metadata generation coexists with pending state",
                ));
            }
            let directory =
                unix::open_directory_at(&self.generations, &generation_name, Some(0o700), true)?;
            verify_generation(&directory, &receipt, Some(candidate))?;
            directory
        } else {
            let pending = unix::ensure_private_directory(&self.generations, &pending_name)?;
            trip(faults, TestBarrier::PendingDirectory)?;
            write_generation(&pending, candidate, &receipt_bytes)?;
            trip(faults, TestBarrier::GenerationFiles)?;
            verify_generation(&pending, &receipt, Some(candidate))?;
            unix::sync_directory(&pending)?;
            unix::rename_noreplace(
                &self.generations,
                &pending_name,
                &self.generations,
                &generation_name,
            )?;
            trip(faults, TestBarrier::GenerationPublish)?;
            unix::sync_directory(&self.generations)?;
            trip(faults, TestBarrier::GenerationsSync)?;
            let directory =
                unix::open_directory_at(&self.generations, &generation_name, Some(0o700), true)?;
            verify_generation(&directory, &receipt, Some(candidate))?;
            directory
        };

        let selector = MetadataSelectorV1::new(sequence, receipt_digest)?;
        let selector_bytes = selector.to_bytes()?;
        let pending_selector = format!(".current-{sequence:020}.json");
        let selector_file =
            file::write_or_resume_private_file(&self.metadata, &pending_selector, &selector_bytes)?;
        unix::full_sync_file(&selector_file)?;
        let selector_identity =
            unix::regular_file_identity(&selector_file, self.metadata.device())?;
        trip(faults, TestBarrier::SelectorFile)?;
        unix::sync_directory(&self.metadata)?;
        trip(faults, TestBarrier::MetadataPrecommitSync)?;
        before_precommit_reopen();

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
            let (_, bytes, _) =
                file::read_regular_file(&live.metadata, CURRENT, 0o600, MAX_SELECTOR_BYTES)?;
            if bytes != expected.to_bytes()? {
                return Err(MetadataJournalError::Invalid(
                    "current metadata selector changed before commit",
                ));
            }
            unix::rename_replace(&live.metadata, &pending_selector, CURRENT)?;
        } else {
            unix::rename_noreplace(&live.metadata, &pending_selector, &live.metadata, CURRENT)?;
        }
        trip(faults, TestBarrier::SelectorCommit).map_err(|error| error.after_commit(sequence))?;
        self.repeat_postcommit_barriers(&selector, faults)?;
        self.finish_predecessor_cleanup(&selector, faults)
            .map_err(|error| error.after_commit(sequence))?;
        Ok(MetadataCommitOutcome::Committed { sequence })
    }

    #[allow(clippy::too_many_arguments)]
    fn reopen_precommit(
        &self,
        generation_name: &str,
        expected_generation: &Directory,
        pending_selector: &str,
        selector_identity: EntryIdentity,
        selector_bytes: &[u8],
        receipt: &MetadataGenerationReceiptV1,
        candidate: &VerifiedMetadataCandidate,
        prior: Option<&MetadataSelectorV1>,
    ) -> Result<LiveNamespace, MetadataJournalError> {
        let live = self.reopen_namespace()?;
        let _ = read_selector_with_mode(
            &live.metadata,
            &live.generations,
            HistoryMode::LockedRecovery,
        )?;
        if prior.is_some() != unix::entry_identity(&live.metadata, CURRENT)?.is_some() {
            return Err(MetadataJournalError::Invalid(
                "metadata selection changed before commit",
            ));
        }
        let generation =
            unix::open_directory_at(&live.generations, generation_name, Some(0o700), true)?;
        require_same_directory(&generation, expected_generation)?;
        verify_generation(&generation, receipt, Some(candidate))?;
        let (staged_file, staged_bytes, staged_identity) =
            file::read_regular_file(&live.metadata, pending_selector, 0o600, MAX_SELECTOR_BYTES)?;
        unix::verify_named_identity(&live.metadata, pending_selector, selector_identity)?;
        if staged_bytes != selector_bytes
            || staged_identity != selector_identity
            || unix::regular_file_identity(&staged_file, live.metadata.device())?
                != selector_identity
        {
            return Err(MetadataJournalError::Invalid(
                "pending metadata selector changed before commit",
            ));
        }
        Ok(live)
    }

    fn reopen_namespace(&self) -> Result<LiveNamespace, MetadataJournalError> {
        let locked = self.locked.reopen()?;
        let metadata = unix::open_directory_at(&locked.update, METADATA, Some(0o700), true)?;
        require_same_directory(&metadata, &self.metadata)?;
        let generations = unix::open_directory_at(&metadata, GENERATIONS, Some(0o700), true)?;
        require_same_directory(&generations, &self.generations)?;
        Ok(LiveNamespace {
            root: locked.root,
            update: locked.update,
            metadata,
            generations,
        })
    }

    fn repeat_postcommit_barriers(
        &self,
        expected: &MetadataSelectorV1,
        faults: FaultPlan,
    ) -> Result<(), MetadataJournalError> {
        let sequence = expected.sequence();
        let repair = || -> Result<(), MetadataJournalError> {
            let live = self.reopen_namespace()?;
            let selected = read_selector_with_mode(
                &live.metadata,
                &live.generations,
                HistoryMode::LockedRecovery,
            )?
            .ok_or(MetadataJournalError::Invalid(
                "committed metadata selector disappeared",
            ))?;
            if &selected != expected {
                return Err(MetadataJournalError::Invalid(
                    "live metadata selector differs from the committed generation",
                ));
            }
            let (selector_file, selector_bytes, _) =
                file::read_regular_file(&live.metadata, CURRENT, 0o600, MAX_SELECTOR_BYTES)?;
            if selector_bytes != expected.to_bytes()?
                || unix::entry_identity(&live.metadata, &format!(".current-{sequence:020}.json"))?
                    .is_some()
            {
                return Err(MetadataJournalError::Invalid(
                    "committed metadata selector has transaction cruft",
                ));
            }
            let generation = unix::open_directory_at(
                &live.generations,
                &format!("{sequence:020}"),
                Some(0o700),
                true,
            )?;
            let receipt = read_receipt(&live.generations, expected)?;
            receipt.validate_state_identity(&self.installation_id, &self.state_root)?;
            verify_generation(&generation, &receipt, None)?;
            unix::sync_directory(&generation)?;
            trip(faults, TestBarrier::GenerationPostcommitSync)?;
            unix::sync_directory(&live.generations)?;
            unix::sync_directory(&live.metadata)?;
            trip(faults, TestBarrier::MetadataPostcommitSync)?;
            unix::sync_directory(&live.update)?;
            trip(faults, TestBarrier::UpdatePostcommitSync)?;
            unix::sync_directory(&live.root)?;
            trip(faults, TestBarrier::RootPostcommitSync)?;
            // The final FULLFSYNC is the media-flush endpoint for every
            // preceding file and directory barrier on this state-root device.
            unix::full_sync_file(&selector_file)?;
            trip(faults, TestBarrier::SelectorFullSync)
        };
        repair().map_err(|error| error.after_commit(sequence))
    }

    fn finish_predecessor_cleanup(
        &self,
        selected: &MetadataSelectorV1,
        faults: FaultPlan,
    ) -> Result<(), MetadataJournalError> {
        if selected.sequence() == 1 {
            return Ok(());
        }
        // Deletion authority is never exercised through the descriptors that
        // happened to be opened when the lock was acquired. Reopen and bind
        // the full authorized namespace immediately before inspecting or
        // mutating cleanup state.
        let live = self.reopen_namespace()?;
        let live_selected = read_selector_with_mode(
            &live.metadata,
            &live.generations,
            HistoryMode::LockedRecovery,
        )?
        .ok_or(MetadataJournalError::Invalid(
            "selected metadata generation disappeared before cleanup",
        ))?;
        if &live_selected != selected {
            return Err(MetadataJournalError::Invalid(
                "selected metadata generation changed before cleanup",
            ));
        }
        let selected_receipt = read_receipt(&live.generations, selected)?;
        let predecessor_sequence = selected.sequence() - 1;
        let predecessor_name = format!("{predecessor_sequence:020}");
        let prune_name = format!(".prune-{predecessor_sequence:020}");
        let normal_exists = unix::entry_identity(&live.generations, &predecessor_name)?.is_some();
        let prune_exists = unix::entry_identity(&live.generations, &prune_name)?.is_some();
        if normal_exists && prune_exists {
            return Err(MetadataJournalError::Invalid(
                "metadata predecessor and prune state coexist",
            ));
        }
        if !normal_exists && !prune_exists {
            return Ok(());
        }
        let expected_digest =
            selected_receipt
                .predecessor_digest()
                .ok_or(MetadataJournalError::Invalid(
                    "selected generation lacks predecessor binding",
                ))?;
        if normal_exists {
            let predecessor =
                unix::open_directory_at(&live.generations, &predecessor_name, Some(0o700), true)?;
            let receipt = read_receipt_from_directory(&predecessor)?;
            if receipt.digest()? != expected_digest {
                return Err(MetadataJournalError::Invalid(
                    "metadata predecessor digest does not match selected generation",
                ));
            }
            verify_generation(&predecessor, &receipt, None)?;
            unix::rename_noreplace(
                &live.generations,
                &predecessor_name,
                &live.generations,
                &prune_name,
            )?;
            unix::sync_directory(&live.generations)?;
            trip(faults, TestBarrier::PredecessorPruneRename)?;
        }
        let prune = unix::open_directory_at(&live.generations, &prune_name, Some(0o700), true)?;
        let receipt = verify_prune_prefix(&prune, expected_digest)?;
        remove_generation(
            &live.generations,
            &prune_name,
            &prune,
            receipt.as_ref(),
            expected_digest,
            faults,
        )?;
        unix::sync_directory(&live.generations)?;
        trip(faults, TestBarrier::PredecessorPruneRemoval)?;
        unix::sync_directory(&live.metadata)?;
        unix::sync_directory(&live.update)?;
        unix::sync_directory(&live.root)?;
        let (selector_file, selector_bytes, _) =
            file::read_regular_file(&live.metadata, CURRENT, 0o600, MAX_SELECTOR_BYTES)?;
        if selector_bytes != selected.to_bytes()? {
            return Err(MetadataJournalError::Invalid(
                "selected metadata generation changed during cleanup",
            ));
        }
        // This endpoint makes all predecessor deletion barriers durable
        // before a later selector is allowed to commit.
        unix::full_sync_file(&selector_file)?;
        trip(faults, TestBarrier::PredecessorPruneFullSync)
    }
}

#[cfg(test)]
pub(super) fn commit_candidate_for_test(
    authorization: MetadataStateAuthorization,
    candidate: VerifiedMetadataCandidate,
    faults: FaultPlan,
) -> Result<MetadataCommitOutcome, MetadataJournalError> {
    commit_candidate_with_faults(authorization, candidate, faults)
}

#[cfg(test)]
pub(super) fn commit_candidate_with_hook_for_test(
    authorization: MetadataStateAuthorization,
    candidate: VerifiedMetadataCandidate,
    hook: impl FnOnce(),
) -> Result<MetadataCommitOutcome, MetadataJournalError> {
    if authorization.root.canonical.as_str() != candidate.state_root
        || authorization.installation_id != candidate.installation_id
    {
        return Err(MetadataJournalError::Invalid(
            "candidate identity differs from the explicit metadata-state authorization",
        ));
    }
    let locked = LockedInstallation::acquire(&authorization.root.path)?;
    let journal = LockedMetadataJournal::open(
        locked,
        authorization.installation_id,
        candidate.state_root.clone(),
    )?;
    hook();
    journal.commit(&candidate, FaultPlan::default())
}

#[cfg(test)]
pub(super) fn commit_candidate_with_precommit_hook_for_test(
    authorization: MetadataStateAuthorization,
    candidate: VerifiedMetadataCandidate,
    hook: impl FnOnce(),
) -> Result<MetadataCommitOutcome, MetadataJournalError> {
    if authorization.root.canonical.as_str() != candidate.state_root
        || authorization.installation_id != candidate.installation_id
    {
        return Err(MetadataJournalError::Invalid(
            "candidate identity differs from the explicit metadata-state authorization",
        ));
    }
    let locked = LockedInstallation::acquire(&authorization.root.path)?;
    LockedMetadataJournal::open(
        locked,
        authorization.installation_id,
        candidate.state_root.clone(),
    )?
    .commit_with_precommit_hook(&candidate, FaultPlan::default(), hook)
}

#[cfg(test)]
pub(super) fn cleanup_selected_with_hook_for_test(
    authorization: MetadataStateAuthorization,
    hook: impl FnOnce(),
) -> Result<(), MetadataJournalError> {
    let locked = LockedInstallation::acquire(&authorization.root.path)?;
    let journal = LockedMetadataJournal::open(
        locked,
        authorization.installation_id,
        authorization.root.canonical.as_str().to_owned(),
    )?;
    let selected = read_selector_with_mode(
        &journal.metadata,
        &journal.generations,
        HistoryMode::LockedRecovery,
    )?
    .ok_or(MetadataJournalError::Invalid(
        "selected metadata generation disappeared before cleanup",
    ))?;
    hook();
    journal.finish_predecessor_cleanup(&selected, FaultPlan::default())
}

#[cfg(test)]
pub(super) fn hold_metadata_lock_for_test(root: &std::path::Path, ready: &std::path::Path) {
    let locked = LockedInstallation::acquire(root).expect("acquire metadata installation lock");
    let _journal = LockedMetadataJournal::open(
        locked,
        "7c907c7a-3125-4a40-a8b3-1c125080e46a".to_owned(),
        root.to_str().expect("UTF-8 root").to_owned(),
    )
    .expect("open metadata journal");
    std::fs::write(ready, b"ready").expect("signal metadata lock");
    std::thread::sleep(std::time::Duration::from_secs(60));
}
