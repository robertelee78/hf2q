use sha2::{Digest, Sha256};

use super::model::{
    EmbeddedTrustRoot, MetadataResponse, VerificationStep, VerifiedMetadataCandidate,
};
use super::replay::{begin_from_selected_with_clock, replay_selected};
use super::verifier::{begin_from_anchor_with_clock, ClockSource};
use super::TufVerifierError;
use crate::distribution::install_state::metadata::schema::MetadataGenerationReceiptV2;
use crate::distribution::install_state::metadata::{
    lock_metadata_state, read_selected, MetadataCommitOutcome, MetadataRestartCleanup,
    MetadataStateAuthorization, StoredMetadataGeneration,
};

/// Proof that the exact authenticated candidate was durably selected and then
/// reread through the ordinary fail-closed journal path.
///
/// This intentionally exposes neither targets bytes nor target lookup. A
/// later transport/application slice must start from a fresh selected-journal
/// replay rather than treating this value as install authority.
#[derive(Debug)]
pub(super) struct DurableMetadataBaseline {
    sequence: u64,
    generation_sha256: [u8; 32],
}

#[derive(Debug)]
pub(super) struct RestartRecovery {
    cleanup: MetadataRestartCleanup,
    selected: Option<DurableMetadataBaseline>,
}

impl RestartRecovery {
    #[cfg(test)]
    pub(super) fn cleanup(&self) -> MetadataRestartCleanup {
        self.cleanup
    }

    #[cfg(test)]
    pub(super) fn selected(&self) -> Option<&DurableMetadataBaseline> {
        self.selected.as_ref()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidateDisposition {
    AlreadySelected,
    Advancing,
}

/// Non-forgeable, single-use authority for one advancing selector commit.
///
/// Construction is private to the TUF coordinator after lock-held replay.
/// The metadata journal can invoke the final clock check but no other
/// distribution sibling can fabricate this capability or bypass freshness.
pub(in crate::distribution) struct AdvancingCommitGuard<'a> {
    candidate: &'a VerifiedMetadataCandidate,
    clock: ClockSource,
    last_sample: jiff::Timestamp,
    expiries: [jiff::Timestamp; 4],
}

impl<'a> AdvancingCommitGuard<'a> {
    fn new(
        candidate: &'a VerifiedMetadataCandidate,
        mut clock: ClockSource,
    ) -> Result<Self, TufVerifierError> {
        let expiries = candidate_expiries(candidate)?;
        let last_sample = clock.sample()?;
        require_candidate_fresh_at(candidate, &expiries, last_sample)?;
        Ok(Self {
            candidate,
            clock,
            last_sample,
            expiries,
        })
    }

    pub(in crate::distribution) fn candidate(&self) -> &'a VerifiedMetadataCandidate {
        self.candidate
    }

    pub(in crate::distribution) fn check_at_selector_boundary(
        &mut self,
    ) -> Result<(), crate::distribution::install_state::metadata::MetadataJournalError> {
        let sample = self.clock.sample().map_err(|_| {
            crate::distribution::install_state::metadata::MetadataJournalError::PrecommitRejected
        })?;
        if sample < self.last_sample {
            return Err(
                crate::distribution::install_state::metadata::MetadataJournalError::PrecommitRejected,
            );
        }
        require_candidate_fresh_at(self.candidate, &self.expiries, sample).map_err(|_| {
            crate::distribution::install_state::metadata::MetadataJournalError::PrecommitRejected
        })?;
        self.last_sample = sample;
        Ok(())
    }
}

impl DurableMetadataBaseline {
    #[cfg(test)]
    pub(super) fn sequence(&self) -> u64 {
        self.sequence
    }

    #[cfg(test)]
    pub(super) fn generation_sha256(&self) -> [u8; 32] {
        self.generation_sha256
    }
}

pub(super) fn commit_and_reopen(
    authorization: &MetadataStateAuthorization,
    anchor: &EmbeddedTrustRoot,
    candidate: VerifiedMetadataCandidate,
) -> Result<(MetadataCommitOutcome, DurableMetadataBaseline), TufVerifierError> {
    commit_and_reopen_with_clock(authorization, anchor, candidate, ClockSource::System)
}

fn commit_and_reopen_with_clock(
    authorization: &MetadataStateAuthorization,
    anchor: &EmbeddedTrustRoot,
    candidate: VerifiedMetadataCandidate,
    commit_clock: ClockSource,
) -> Result<(MetadataCommitOutcome, DurableMetadataBaseline), TufVerifierError> {
    let locked = lock_metadata_state(authorization)?;
    let selected = locked.read_selected_for_recovery()?;
    let disposition = reauthenticate_candidate(authorization, anchor, selected, &candidate)?;

    let outcome = match disposition {
        CandidateDisposition::AlreadySelected => locked.repair_selected(&candidate)?,
        CandidateDisposition::Advancing => {
            let mut guard = AdvancingCommitGuard::new(&candidate, commit_clock)?;
            locked.commit_advancing(&mut guard)?
        }
    };
    let locked_selected = locked
        .read_selected_for_recovery()?
        .ok_or(TufVerifierError::DurableCommitMismatch)?;
    let locked_proof = authenticate_exact_selection(anchor, locked_selected, &candidate)?;
    drop(locked);

    let reopened = read_selected(authorization)?.ok_or(TufVerifierError::DurableCommitMismatch)?;
    let reopened_proof = authenticate_exact_selection(anchor, reopened, &candidate)?;
    if locked_proof.sequence != reopened_proof.sequence
        || locked_proof.generation_sha256 != reopened_proof.generation_sha256
    {
        return Err(TufVerifierError::DurableCommitMismatch);
    }
    Ok((outcome, reopened_proof))
}

/// Recover crash durability without converting unselected disk residue into
/// TUF authority. A selected generation is historically authenticated as a
/// rollback floor; every never-selected transaction is discarded under the
/// held lock and must be replaced by a completely fresh network transcript.
pub(super) fn recover_after_process_restart(
    authorization: &MetadataStateAuthorization,
    anchor: &EmbeddedTrustRoot,
) -> Result<RestartRecovery, TufVerifierError> {
    let locked = lock_metadata_state(authorization)?;
    let before = locked
        .read_selected_for_recovery()?
        .map(|stored| authenticate_stored_selection(anchor, stored))
        .transpose()?;
    let cleanup = locked.discard_unselected_transaction()?;
    let locked_selected = locked
        .read_selected_for_recovery()?
        .map(|stored| authenticate_stored_selection(anchor, stored))
        .transpose()?;
    require_same_optional_proof(before.as_ref(), locked_selected.as_ref())?;
    drop(locked);

    let reopened = read_selected(authorization)?
        .map(|stored| authenticate_stored_selection(anchor, stored))
        .transpose()?;
    require_same_optional_proof(locked_selected.as_ref(), reopened.as_ref())?;
    Ok(RestartRecovery {
        cleanup,
        selected: reopened,
    })
}

fn reauthenticate_candidate(
    authorization: &MetadataStateAuthorization,
    anchor: &EmbeddedTrustRoot,
    selected: Option<StoredMetadataGeneration>,
    candidate: &VerifiedMetadataCandidate,
) -> Result<CandidateDisposition, TufVerifierError> {
    if let Some(stored) = selected.as_ref() {
        if selection_matches_candidate(stored, candidate)? {
            let _ = authenticate_exact_selection(
                anchor,
                selected.ok_or(TufVerifierError::DurableCommitMismatch)?,
                candidate,
            )?;
            return Ok(CandidateDisposition::AlreadySelected);
        }
    }
    let selected_root_count = selected
        .as_ref()
        .map_or(0, |stored| stored.root_chain().len());
    let prior_targets = selected.as_ref().map(|stored| stored.targets().to_vec());
    if candidate.root_chain().len() < selected_root_count {
        return Err(TufVerifierError::RollbackOrEquivocation);
    }
    let clock = ClockSource::fixed(
        candidate.verification_started_at(),
        candidate.verification_completed_at(),
    );
    let mut step = match selected {
        Some(stored) => begin_from_selected_with_clock(authorization, anchor, stored, clock)?,
        None => begin_from_anchor_with_clock(authorization, anchor, clock)?,
    };

    for root in &candidate.root_chain()[selected_root_count..] {
        step = respond_exact(step, root.request_name(), root.bytes())?;
    }
    step = respond_not_found(step)?;
    step = respond_exact(
        step,
        candidate.timestamp().request_name(),
        candidate.timestamp().bytes(),
    )?;
    step = respond_exact(
        step,
        candidate.snapshot().request_name(),
        candidate.snapshot().bytes(),
    )?;
    step = respond_exact(
        step,
        candidate.targets().request_name(),
        candidate.targets().bytes(),
    )?;
    match step {
        VerificationStep::Candidate(reverified) if reverified.exactly_matches(candidate) => {
            if let Some(prior_targets) = prior_targets {
                super::target_set::require_retained_release_floor(
                    &prior_targets,
                    reverified.targets().bytes(),
                )?;
            }
            Ok(CandidateDisposition::Advancing)
        }
        VerificationStep::Candidate(_) | VerificationStep::Request(_) => {
            Err(TufVerifierError::AuthenticationFailed)
        }
    }
}

fn require_candidate_fresh_at(
    candidate: &VerifiedMetadataCandidate,
    expiries: &[jiff::Timestamp; 4],
    reference: jiff::Timestamp,
) -> Result<(), TufVerifierError> {
    if reference < candidate.verification_completed_at() {
        return Err(TufVerifierError::ClockRollback);
    }
    if expiries.iter().any(|expiry| *expiry <= reference) {
        return Err(TufVerifierError::ExpiredMetadata);
    }
    Ok(())
}

fn candidate_expiries(
    candidate: &VerifiedMetadataCandidate,
) -> Result<[jiff::Timestamp; 4], TufVerifierError> {
    let root = super::profile::root(candidate.trusted_root().bytes())?;
    let timestamp = super::profile::timestamp(candidate.timestamp().bytes())?;
    let snapshot = super::profile::snapshot(candidate.snapshot().bytes())?;
    let targets = super::profile::targets(candidate.targets().bytes())?;
    Ok([
        super::profile::expiry(&root.signed)?,
        super::profile::expiry(&timestamp.signed)?,
        super::profile::expiry(&snapshot.signed)?,
        super::profile::expiry(&targets.signed)?,
    ])
}

fn respond_exact(
    step: VerificationStep,
    expected_name: &str,
    bytes: &[u8],
) -> Result<VerificationStep, TufVerifierError> {
    let VerificationStep::Request(request) = step else {
        return Err(TufVerifierError::UnexpectedResponse);
    };
    if request.spec().relative_name() != expected_name {
        return Err(TufVerifierError::UnexpectedResponse);
    }
    request.respond(MetadataResponse::Found(bytes.into()))
}

fn respond_not_found(step: VerificationStep) -> Result<VerificationStep, TufVerifierError> {
    let VerificationStep::Request(request) = step else {
        return Err(TufVerifierError::UnexpectedResponse);
    };
    request.respond(MetadataResponse::ConfirmedNotFound)
}

fn authenticate_exact_selection(
    anchor: &EmbeddedTrustRoot,
    stored: StoredMetadataGeneration,
    candidate: &VerifiedMetadataCandidate,
) -> Result<DurableMetadataBaseline, TufVerifierError> {
    if !selection_matches_candidate(&stored, candidate)? {
        return Err(TufVerifierError::DurableCommitMismatch);
    }
    authenticate_stored_selection(anchor, stored)
}

fn authenticate_stored_selection(
    anchor: &EmbeddedTrustRoot,
    stored: StoredMetadataGeneration,
) -> Result<DurableMetadataBaseline, TufVerifierError> {
    let receipt = MetadataGenerationReceiptV2::parse(stored.generation_receipt())?;
    let sequence = stored.sequence();
    let generation_sha256 = Sha256::digest(stored.generation_receipt()).into();
    let completed = receipt.verification_completed_at()?;
    let _ = replay_selected(anchor, &receipt, stored, completed)?;
    Ok(DurableMetadataBaseline {
        sequence,
        generation_sha256,
    })
}

fn require_same_optional_proof(
    left: Option<&DurableMetadataBaseline>,
    right: Option<&DurableMetadataBaseline>,
) -> Result<(), TufVerifierError> {
    match (left, right) {
        (None, None) => Ok(()),
        (Some(left), Some(right))
            if left.sequence == right.sequence
                && left.generation_sha256 == right.generation_sha256 =>
        {
            Ok(())
        }
        _ => Err(TufVerifierError::DurableCommitMismatch),
    }
}

#[cfg(test)]
pub(super) fn commit_and_reopen_for_test(
    authorization: &MetadataStateAuthorization,
    anchor: &EmbeddedTrustRoot,
    candidate: VerifiedMetadataCandidate,
    samples: impl IntoIterator<Item = jiff::Timestamp>,
) -> Result<(MetadataCommitOutcome, DurableMetadataBaseline), TufVerifierError> {
    commit_and_reopen_with_clock(
        authorization,
        anchor,
        candidate,
        ClockSource::scripted(samples),
    )
}

fn selection_matches_candidate(
    stored: &StoredMetadataGeneration,
    candidate: &VerifiedMetadataCandidate,
) -> Result<bool, TufVerifierError> {
    let receipt = MetadataGenerationReceiptV2::parse(stored.generation_receipt())?;
    Ok(receipt.matches_candidate(candidate)
        && stored.anchor_root() == candidate.anchor_root().bytes()
        && stored.trusted_root() == candidate.trusted_root().bytes()
        && stored.timestamp() == candidate.timestamp().bytes()
        && stored.snapshot() == candidate.snapshot().bytes()
        && stored.targets() == candidate.targets().bytes()
        && stored.root_chain().len() == candidate.root_chain().len()
        && stored
            .root_chain()
            .iter()
            .zip(candidate.root_chain())
            .all(|(stored, candidate)| stored.as_ref() == candidate.bytes()))
}
