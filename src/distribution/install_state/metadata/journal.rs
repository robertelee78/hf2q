use sha2::{Digest, Sha256};

use super::schema::{MetadataGenerationReceiptV2, MetadataSelectorV2, MAX_SELECTOR_BYTES};
use super::{
    MetadataCommitOutcome, MetadataJournalError, MetadataRestartCleanup,
    MetadataStateAuthorization, VerifiedMetadataCandidate,
};
use crate::distribution::install_state::file;
use crate::distribution::install_state::identity::LockedInstallationIdentity;
use crate::distribution::install_state::unix::{self, Directory};
use crate::distribution::update_auth::AdvancingCommitGuard;

mod cleanup;
mod durability;
mod fault;
mod restart;
#[cfg(test)]
mod test_support;
mod validation;

#[cfg(not(test))]
use fault::FaultPlan;
use fault::{trip, TestBarrier};
#[cfg(test)]
pub(in crate::distribution) use fault::{Barrier, FaultPlan};
#[cfg(test)]
pub(super) use test_support::{
    cleanup_selected_with_hook_for_test, commit_candidate_with_hook_for_test,
    commit_candidate_with_precommit_hook_for_test, discard_unselected_with_hook_for_test,
    hold_metadata_lock_for_test,
};
#[cfg(test)]
pub(in crate::distribution) use test_support::{
    commit_candidate_for_test, discard_unselected_for_test,
};
pub(in crate::distribution) use validation::read_selected;
use validation::{
    generation_matches_candidate, read_receipt, read_selector_with_mode, verify_generation,
    write_generation,
};

const METADATA: &str = "metadata";
const GENERATIONS: &str = "generations";
const CURRENT: &str = "current.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HistoryMode {
    Authority,
    LockedRecovery,
}

/// One installation-scoped metadata transaction session.
///
/// Holding this value keeps the same descriptor-backed installation lock
/// across live-floor revalidation, journal commit, and committed-byte replay.
/// It grants no target lookup, activation, overwrite, or deletion authority
/// outside the metadata journal.
pub(in crate::distribution) struct LockedMetadataState {
    journal: LockedMetadataJournal,
}

pub(in crate::distribution) fn lock_metadata_state(
    authorization: &MetadataStateAuthorization,
) -> Result<LockedMetadataState, MetadataJournalError> {
    let locked = authorization.identity.lock()?;
    Ok(LockedMetadataState {
        journal: LockedMetadataJournal::open(locked)?,
    })
}

impl LockedMetadataState {
    /// Read selected bytes under the shared installation lock with ordinary
    /// authority semantics. Any unselected transaction residue fails closed.
    pub(in crate::distribution) fn read_selected_for_authority(
        &self,
    ) -> Result<Option<super::StoredMetadataGeneration>, MetadataJournalError> {
        self.journal.read_selected_for_authority()
    }

    pub(in crate::distribution) fn create_ephemeral_artifact_stage(
        &self,
        authorization: crate::distribution::update_auth::ArchiveStageAuthorization,
    ) -> Result<super::super::EphemeralArtifactStage, super::super::ArtifactStageError> {
        super::super::artifact::create_ephemeral_artifact_stage_under_lock(
            &self.journal.locked,
            authorization.expected_length(),
        )
    }

    pub(in crate::distribution) fn open_release_extraction<'lock>(
        &'lock self,
        authorization: crate::distribution::update_auth::ExtractionStageAuthorization,
        exact_manifest: &[u8],
        manifest: &crate::distribution::schema::ReleaseManifestV1,
    ) -> Result<super::super::ReleaseExtractionStage<'lock>, super::super::ExtractionError> {
        super::super::extraction::open_release_extraction(
            &self.journal.locked,
            authorization,
            exact_manifest,
            manifest,
        )
    }

    #[cfg(target_os = "macos")]
    pub(in crate::distribution) fn with_extracted_executable<R, E>(
        &self,
        tree: &super::super::ExtractedReleaseTree,
        exact_manifest: &[u8],
        manifest: &crate::distribution::schema::ReleaseManifestV1,
        operation: impl FnOnce(
            &std::path::Path,
            &std::fs::File,
            super::super::ExecutableReleaseBinding,
        ) -> Result<R, E>,
    ) -> Result<R, E>
    where
        E: From<super::super::ExtractionError>,
    {
        super::super::extraction::with_extracted_executable(
            &self.journal.locked,
            tree,
            exact_manifest,
            manifest,
            operation,
        )
    }

    pub(in crate::distribution) fn normalize_extracted_release(
        &self,
        developer_id: crate::distribution::prepared_release::DeveloperIdVerification,
        tree: super::super::ExtractedReleaseTree,
        exact_manifest: &[u8],
        manifest: &crate::distribution::schema::ReleaseManifestV1,
    ) -> Result<super::super::NormalizedExtractedReleaseTree, super::super::ExtractionError> {
        super::super::extraction::normalize_developer_id_verified_release(
            &self.journal.locked,
            developer_id,
            tree,
            exact_manifest,
            manifest,
        )
    }

    #[cfg(target_os = "macos")]
    pub(in crate::distribution) fn with_normalized_executable<R, E>(
        &self,
        tree: &super::super::NormalizedExtractedReleaseTree,
        exact_manifest: &[u8],
        manifest: &crate::distribution::schema::ReleaseManifestV1,
        operation: impl FnOnce(
            &std::path::Path,
            &std::fs::File,
            super::super::ExecutableReleaseBinding,
        ) -> Result<R, E>,
    ) -> Result<R, E>
    where
        E: From<super::super::ExtractionError>,
    {
        super::super::extraction::with_normalized_executable(
            &self.journal.locked,
            tree,
            exact_manifest,
            manifest,
            operation,
        )
    }

    pub(in crate::distribution) fn verify_normalized_release_tree(
        &self,
        tree: &super::super::NormalizedExtractedReleaseTree,
        exact_manifest: &[u8],
        manifest: &crate::distribution::schema::ReleaseManifestV1,
    ) -> Result<(), super::super::ExtractionError> {
        super::super::extraction::verify_normalized_release_tree(
            &self.journal.locked,
            tree,
            exact_manifest,
            manifest,
        )
    }

    /// Read structurally complete selected bytes while allowing only the
    /// bounded transaction residue that this held lock is authorized to
    /// recover. The signed-update verifier must authenticate the result.
    pub(in crate::distribution) fn read_selected_for_recovery(
        &self,
    ) -> Result<Option<super::StoredMetadataGeneration>, MetadataJournalError> {
        self.journal.read_selected_for_recovery()
    }

    pub(in crate::distribution) fn repair_selected(
        &self,
        candidate: &VerifiedMetadataCandidate,
    ) -> Result<MetadataCommitOutcome, MetadataJournalError> {
        if self.journal.locked.state_root().as_str() != candidate.state_root()
            || self.journal.locked.installation_id().as_str() != candidate.installation_id()
        {
            return Err(MetadataJournalError::Invalid(
                "candidate identity differs from the locked metadata state",
            ));
        }
        self.journal.repair_selected(candidate)
    }

    /// Commit only through the sealed TUF coordinator's freshness capability.
    pub(in crate::distribution) fn commit_advancing(
        &self,
        guard: &mut AdvancingCommitGuard<'_>,
    ) -> Result<MetadataCommitOutcome, MetadataJournalError> {
        let candidate = guard.candidate();
        if self.journal.locked.state_root().as_str() != candidate.state_root()
            || self.journal.locked.installation_id().as_str() != candidate.installation_id()
        {
            return Err(MetadataJournalError::Invalid(
                "candidate identity differs from the locked metadata state",
            ));
        }
        self.journal.commit_with_precommit_hooks(
            candidate,
            FaultPlan::default(),
            || Ok(()),
            || guard.check_at_selector_boundary(),
        )
    }

    /// Remove only the bounded successor transaction that never became
    /// selected. Stored bytes are never converted into TUF authority.
    pub(in crate::distribution) fn discard_unselected_transaction(
        &self,
    ) -> Result<MetadataRestartCleanup, MetadataJournalError> {
        self.journal
            .discard_unselected_transaction(FaultPlan::default())
    }
}

struct LockedMetadataJournal {
    locked: LockedInstallationIdentity,
    metadata: Directory,
    generations: Directory,
}

struct LiveNamespace {
    root: Directory,
    update: Directory,
    metadata: Directory,
    generations: Directory,
}

impl LockedMetadataJournal {
    fn open(locked: LockedInstallationIdentity) -> Result<Self, MetadataJournalError> {
        let metadata = unix::ensure_private_directory(locked.update(), METADATA)?;
        let generations = unix::ensure_private_directory(&metadata, GENERATIONS)?;
        Ok(Self {
            locked,
            metadata,
            generations,
        })
    }

    fn repair_selected(
        &self,
        candidate: &VerifiedMetadataCandidate,
    ) -> Result<MetadataCommitOutcome, MetadataJournalError> {
        let live = self.reopen_namespace()?;
        let selector = read_selector_with_mode(
            &live.metadata,
            &live.generations,
            HistoryMode::LockedRecovery,
        )?
        .ok_or(MetadataJournalError::Invalid(
            "selected metadata generation is absent",
        ))?;
        let receipt = read_receipt(&live.generations, &selector)?;
        receipt.validate_state_identity(
            self.locked.installation_id().as_str(),
            self.locked.state_root().as_str(),
        )?;
        let next = selector
            .sequence()
            .checked_add(1)
            .ok_or(MetadataJournalError::Invalid(
                "metadata generation sequence overflowed",
            ))?;
        if unix::entry_identity(&live.generations, &format!(".pending-{next:020}"))?.is_some()
            || unix::entry_identity(&live.generations, &format!("{next:020}"))?.is_some()
            || unix::entry_identity(&live.metadata, &format!(".current-{next:020}.json"))?.is_some()
            || !generation_matches_candidate(&live.generations, selector.sequence(), candidate)?
        {
            return Err(MetadataJournalError::Invalid(
                "selected metadata does not exactly match the repair candidate",
            ));
        }
        self.repeat_postcommit_barriers(&selector, FaultPlan::default())?;
        self.finish_predecessor_cleanup(&selector, FaultPlan::default())
            .map_err(|error| error.after_commit(selector.sequence()))?;
        Ok(MetadataCommitOutcome::AlreadyCommitted {
            sequence: selector.sequence(),
        })
    }

    #[cfg(test)]
    fn commit(
        &self,
        candidate: &VerifiedMetadataCandidate,
        faults: FaultPlan,
    ) -> Result<MetadataCommitOutcome, MetadataJournalError> {
        self.commit_with_precommit_hooks(candidate, faults, || Ok(()), || Ok(()))
    }

    fn read_selected_for_recovery(
        &self,
    ) -> Result<Option<super::StoredMetadataGeneration>, MetadataJournalError> {
        let live = self.reopen_namespace()?;
        let selected = validation::read_selected_with_mode(
            &live.metadata,
            &live.generations,
            HistoryMode::LockedRecovery,
        )?;
        if let Some(stored) = &selected {
            MetadataGenerationReceiptV2::parse(&stored.generation_receipt)?
                .validate_state_identity(
                    self.locked.installation_id().as_str(),
                    self.locked.state_root().as_str(),
                )?;
        }
        Ok(selected)
    }

    fn read_selected_for_authority(
        &self,
    ) -> Result<Option<super::StoredMetadataGeneration>, MetadataJournalError> {
        let live = self.reopen_namespace()?;
        let selected = validation::read_selected_with_mode(
            &live.metadata,
            &live.generations,
            HistoryMode::Authority,
        )?;
        if let Some(stored) = &selected {
            MetadataGenerationReceiptV2::parse(&stored.generation_receipt)?
                .validate_state_identity(
                    self.locked.installation_id().as_str(),
                    self.locked.state_root().as_str(),
                )?;
        }
        Ok(selected)
    }

    fn commit_with_precommit_hooks(
        &self,
        candidate: &VerifiedMetadataCandidate,
        faults: FaultPlan,
        before_precommit_reopen: impl FnOnce() -> Result<(), MetadataJournalError>,
        final_precommit_guard: impl FnOnce() -> Result<(), MetadataJournalError>,
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
            receipt.validate_state_identity(
                self.locked.installation_id().as_str(),
                self.locked.state_root().as_str(),
            )?;
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
        let receipt = MetadataGenerationReceiptV2::new(sequence, predecessor, candidate)?;
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

        let selector = MetadataSelectorV2::new(sequence, receipt_digest)?;
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
        before_precommit_reopen()?;

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
        }
        final_precommit_guard()?;
        if prior.is_some() {
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
}
