use super::super::schema::{MetadataSelectorV1, MAX_SELECTOR_BYTES};
use super::super::{MetadataJournalError, MetadataRestartCleanup};
use super::cleanup::discard_pending_generation;
use super::fault::{trip, FaultPlan, TestBarrier};
use super::validation::{
    read_receipt, read_receipt_from_directory, read_selector_with_mode, require_same_directory,
};
use super::{HistoryMode, LockedMetadataJournal, CURRENT};
use crate::distribution::install_state::file;
use crate::distribution::install_state::unix;

impl LockedMetadataJournal {
    pub(super) fn discard_unselected_transaction(
        &self,
        faults: FaultPlan,
    ) -> Result<MetadataRestartCleanup, MetadataJournalError> {
        let live = self.reopen_namespace()?;
        let selected = read_selector_with_mode(
            &live.metadata,
            &live.generations,
            HistoryMode::LockedRecovery,
        )?;
        if let Some(selector) = &selected {
            let receipt = read_receipt(&live.generations, selector)?;
            receipt.validate_state_identity(&self.installation_id, &self.state_root)?;
            self.repeat_postcommit_barriers(selector, FaultPlan::default())?;
            self.finish_predecessor_cleanup(selector, FaultPlan::default())
                .map_err(|error| error.after_commit(selector.sequence()))?;
        }

        let live = self.reopen_namespace()?;
        let selected = read_selector_with_mode(
            &live.metadata,
            &live.generations,
            HistoryMode::LockedRecovery,
        )?;
        let sequence = selected.as_ref().map_or(Ok(1), |selector| {
            selector
                .sequence()
                .checked_add(1)
                .ok_or(MetadataJournalError::Invalid(
                    "metadata generation sequence overflowed",
                ))
        })?;
        let pending_name = format!(".pending-{sequence:020}");
        let generation_name = format!("{sequence:020}");
        let pending_selector = format!(".current-{sequence:020}.json");
        let has_pending = unix::entry_identity(&live.generations, &pending_name)?.is_some();
        let has_published = unix::entry_identity(&live.generations, &generation_name)?.is_some();
        let has_staged_selector =
            unix::entry_identity(&live.metadata, &pending_selector)?.is_some();

        if !has_pending && !has_published && !has_staged_selector {
            self.finish_discard_barriers(selected.as_ref(), faults)?;
            return Ok(MetadataRestartCleanup::Clean);
        }

        if has_staged_selector {
            if !has_published || has_pending {
                return Err(MetadataJournalError::Invalid(
                    "staged selector is not paired with one published successor",
                ));
            }
            let successor =
                unix::open_directory_at(&live.generations, &generation_name, Some(0o700), true)?;
            let receipt = read_receipt_from_directory(&successor)?;
            let expected = MetadataSelectorV1::new(sequence, receipt.digest()?)?.to_bytes()?;
            let (_, actual, identity) = file::read_regular_file(
                &live.metadata,
                &pending_selector,
                0o600,
                MAX_SELECTOR_BYTES,
            )?;
            if actual != expected {
                return Err(MetadataJournalError::Invalid(
                    "staged selector does not bind the published successor",
                ));
            }
            unix::remove_named_regular_file(&live.metadata, &pending_selector, identity)?;
            unix::sync_directory(&live.metadata)?;
            trip(faults, TestBarrier::SuccessorDiscardSelector)?;
        }

        if has_published {
            let published =
                unix::open_directory_at(&live.generations, &generation_name, Some(0o700), true)?;
            unix::rename_noreplace(
                &live.generations,
                &generation_name,
                &live.generations,
                &pending_name,
            )?;
            unix::sync_directory(&live.generations)?;
            let renamed =
                unix::open_directory_at(&live.generations, &pending_name, Some(0o700), true)?;
            require_same_directory(&renamed, &published)?;
            trip(faults, TestBarrier::SuccessorDiscardRename)?;
        }

        let live = self.reopen_namespace()?;
        let _ = read_selector_with_mode(
            &live.metadata,
            &live.generations,
            HistoryMode::LockedRecovery,
        )?;
        let pending = unix::open_directory_at(&live.generations, &pending_name, Some(0o700), true)?;
        discard_pending_generation(&live.generations, &pending_name, &pending, faults)?;
        self.finish_discard_barriers(selected.as_ref(), faults)?;
        Ok(MetadataRestartCleanup::DiscardedUnselected { sequence })
    }

    fn finish_discard_barriers(
        &self,
        expected_selected: Option<&MetadataSelectorV1>,
        faults: FaultPlan,
    ) -> Result<(), MetadataJournalError> {
        let live = self.reopen_namespace()?;
        let selected =
            read_selector_with_mode(&live.metadata, &live.generations, HistoryMode::Authority)?;
        if selected.as_ref() != expected_selected {
            return Err(MetadataJournalError::Invalid(
                "metadata selection changed during successor discard",
            ));
        }
        unix::sync_directory(&live.generations)?;
        trip(faults, TestBarrier::SuccessorDiscardGenerationsSync)?;
        unix::sync_directory(&live.metadata)?;
        trip(faults, TestBarrier::SuccessorDiscardMetadataSync)?;
        unix::sync_directory(&live.update)?;
        trip(faults, TestBarrier::SuccessorDiscardUpdateSync)?;
        unix::sync_directory(&live.root)?;
        trip(faults, TestBarrier::SuccessorDiscardRootSync)?;
        if let Some(selector) = &selected {
            let (file, bytes, _) =
                file::read_regular_file(&live.metadata, CURRENT, 0o600, MAX_SELECTOR_BYTES)?;
            if bytes != selector.to_bytes()? {
                return Err(MetadataJournalError::Invalid(
                    "selected metadata changed during successor discard",
                ));
            }
            unix::full_sync_file(&file)?;
        } else {
            self.locked.full_sync_endpoint()?;
        }
        trip(faults, TestBarrier::SuccessorDiscardFullSync)
    }
}
