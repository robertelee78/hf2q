use super::super::{
    MetadataCommitOutcome, MetadataJournalError, MetadataRestartCleanup,
    MetadataStateAuthorization, VerifiedMetadataCandidate,
};
use super::fault::FaultPlan;
use super::validation::read_selector_with_mode;
use super::{HistoryMode, LockedMetadataJournal};
use crate::distribution::install_state::locked::LockedInstallation;

#[cfg(test)]
fn commit_candidate_with_faults(
    authorization: &MetadataStateAuthorization,
    candidate: VerifiedMetadataCandidate,
    faults: FaultPlan,
) -> Result<MetadataCommitOutcome, MetadataJournalError> {
    if authorization.root.canonical.as_str() != candidate.state_root()
        || authorization.installation_id != candidate.installation_id()
    {
        return Err(MetadataJournalError::Invalid(
            "candidate identity differs from the explicit metadata-state authorization",
        ));
    }
    let locked = LockedInstallation::acquire(&authorization.root.path)?;
    LockedMetadataJournal::open(
        locked,
        authorization.installation_id.clone(),
        candidate.state_root().to_owned(),
    )?
    .commit(&candidate, faults)
}

#[cfg(test)]
pub(in crate::distribution) fn commit_candidate_for_test(
    authorization: MetadataStateAuthorization,
    candidate: VerifiedMetadataCandidate,
    faults: FaultPlan,
) -> Result<MetadataCommitOutcome, MetadataJournalError> {
    commit_candidate_with_faults(&authorization, candidate, faults)
}

#[cfg(test)]
pub(in crate::distribution::install_state::metadata) fn commit_candidate_with_hook_for_test(
    authorization: MetadataStateAuthorization,
    candidate: VerifiedMetadataCandidate,
    hook: impl FnOnce(),
) -> Result<MetadataCommitOutcome, MetadataJournalError> {
    if authorization.root.canonical.as_str() != candidate.state_root()
        || authorization.installation_id != candidate.installation_id()
    {
        return Err(MetadataJournalError::Invalid(
            "candidate identity differs from the explicit metadata-state authorization",
        ));
    }
    let locked = LockedInstallation::acquire(&authorization.root.path)?;
    let journal = LockedMetadataJournal::open(
        locked,
        authorization.installation_id,
        candidate.state_root().to_owned(),
    )?;
    hook();
    journal.commit(&candidate, FaultPlan::default())
}

#[cfg(test)]
pub(in crate::distribution::install_state::metadata) fn commit_candidate_with_precommit_hook_for_test(
    authorization: MetadataStateAuthorization,
    candidate: VerifiedMetadataCandidate,
    hook: impl FnOnce(),
) -> Result<MetadataCommitOutcome, MetadataJournalError> {
    if authorization.root.canonical.as_str() != candidate.state_root()
        || authorization.installation_id != candidate.installation_id()
    {
        return Err(MetadataJournalError::Invalid(
            "candidate identity differs from the explicit metadata-state authorization",
        ));
    }
    let locked = LockedInstallation::acquire(&authorization.root.path)?;
    LockedMetadataJournal::open(
        locked,
        authorization.installation_id,
        candidate.state_root().to_owned(),
    )?
    .commit_with_precommit_hooks(
        &candidate,
        FaultPlan::default(),
        || {
            hook();
            Ok(())
        },
        || Ok(()),
    )
}

#[cfg(test)]
pub(in crate::distribution) fn discard_unselected_for_test(
    authorization: MetadataStateAuthorization,
    faults: FaultPlan,
) -> Result<MetadataRestartCleanup, MetadataJournalError> {
    let locked = LockedInstallation::acquire(&authorization.root.path)?;
    LockedMetadataJournal::open(
        locked,
        authorization.installation_id,
        authorization.root.canonical.as_str().to_owned(),
    )?
    .discard_unselected_transaction(faults)
}

#[cfg(test)]
pub(in crate::distribution::install_state::metadata) fn discard_unselected_with_hook_for_test(
    authorization: MetadataStateAuthorization,
    hook: impl FnOnce(),
) -> Result<MetadataRestartCleanup, MetadataJournalError> {
    let locked = LockedInstallation::acquire(&authorization.root.path)?;
    let journal = LockedMetadataJournal::open(
        locked,
        authorization.installation_id,
        authorization.root.canonical.as_str().to_owned(),
    )?;
    hook();
    journal.discard_unselected_transaction(FaultPlan::default())
}

#[cfg(test)]
pub(in crate::distribution::install_state::metadata) fn cleanup_selected_with_hook_for_test(
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
pub(in crate::distribution::install_state::metadata) fn hold_metadata_lock_for_test(
    root: &std::path::Path,
    ready: &std::path::Path,
) {
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
