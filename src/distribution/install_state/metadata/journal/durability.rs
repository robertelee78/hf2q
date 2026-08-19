use super::super::schema::{MetadataGenerationReceiptV2, MetadataSelectorV2, MAX_SELECTOR_BYTES};
use super::super::{MetadataJournalError, VerifiedMetadataCandidate};
use super::cleanup::{remove_generation, verify_prune_prefix};
use super::fault::{trip, FaultPlan, TestBarrier};
use super::validation::{
    read_receipt, read_receipt_from_directory, read_selector_with_mode, require_same_directory,
    verify_generation,
};
use super::{HistoryMode, LiveNamespace, LockedMetadataJournal, CURRENT, GENERATIONS, METADATA};
use crate::distribution::install_state::file;
use crate::distribution::install_state::unix::{self, Directory, EntryIdentity};

impl LockedMetadataJournal {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn reopen_precommit(
        &self,
        generation_name: &str,
        expected_generation: &Directory,
        pending_selector: &str,
        selector_identity: EntryIdentity,
        selector_bytes: &[u8],
        receipt: &MetadataGenerationReceiptV2,
        candidate: &VerifiedMetadataCandidate,
        prior: Option<&MetadataSelectorV2>,
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

    pub(super) fn reopen_namespace(&self) -> Result<LiveNamespace, MetadataJournalError> {
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

    pub(super) fn repeat_postcommit_barriers(
        &self,
        expected: &MetadataSelectorV2,
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
            receipt.validate_state_identity(
                self.locked.installation_id().as_str(),
                self.locked.state_root().as_str(),
            )?;
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

    pub(super) fn finish_predecessor_cleanup(
        &self,
        selected: &MetadataSelectorV2,
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
