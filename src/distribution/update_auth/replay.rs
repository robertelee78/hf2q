use jiff::Timestamp;
use sigstore_tuf::trusted::TrustedMetadataSet;

use super::model::{EmbeddedTrustRoot, ExactMetadataRole, VerificationStep};
use super::verifier::{
    metadata_name, request_root, ClockSource, Phase, RoleFloor, VerificationState,
};
use super::{profile, TufVerifierError};
use crate::distribution::install_state::metadata::schema::MetadataGenerationReceiptV1;
use crate::distribution::install_state::metadata::{
    MetadataStateAuthorization, StoredMetadataGeneration,
};

/// Begin from structurally selected journal bytes only after replaying every
/// exact role through the compiled anchor and transport-free TUF engine.
pub(super) fn begin_from_selected(
    authorization: &MetadataStateAuthorization,
    anchor: &EmbeddedTrustRoot,
    stored: StoredMetadataGeneration,
) -> Result<VerificationStep, TufVerifierError> {
    begin_from_selected_with_clock(authorization, anchor, stored, ClockSource::System)
}

pub(super) fn begin_from_selected_with_clock(
    authorization: &MetadataStateAuthorization,
    anchor: &EmbeddedTrustRoot,
    stored: StoredMetadataGeneration,
    mut clock: ClockSource,
) -> Result<VerificationStep, TufVerifierError> {
    let receipt = MetadataGenerationReceiptV1::parse(stored.generation_receipt())?;
    if receipt.sequence() != stored.sequence() {
        return Err(TufVerifierError::AuthenticationFailed);
    }
    receipt.validate_state_identity(authorization.installation_id(), authorization.state_root())?;
    let completed_floor = receipt.verification_completed_at()?;
    let mut baseline = replay_selected(anchor, &receipt, stored, completed_floor)?;
    let started_at = clock.sample()?;
    if started_at < completed_floor {
        return Err(TufVerifierError::ClockRollback);
    }
    baseline.installation_id = authorization.installation_id().to_owned();
    baseline.state_root = authorization.state_root().to_owned();
    baseline.clock = clock;
    baseline.started_at = started_at;
    request_root(baseline)
}

pub(super) fn replay_selected(
    anchor: &EmbeddedTrustRoot,
    receipt: &MetadataGenerationReceiptV1,
    stored: StoredMetadataGeneration,
    historical_time: Timestamp,
) -> Result<VerificationState, TufVerifierError> {
    if receipt.sequence() != stored.sequence() {
        return Err(TufVerifierError::AuthenticationFailed);
    }
    if stored.anchor_root() != anchor.bytes() {
        return Err(TufVerifierError::AnchorMismatch);
    }
    let (anchor_root_bytes, stored_roots, trusted_root, timestamp, snapshot, targets) =
        stored.into_authenticated_bytes();
    let parsed_anchor = profile::root(anchor.bytes())?;
    receipt.validate_authenticated_role(
        "anchor-root.json",
        &format!("{}.root.json", parsed_anchor.signed.version),
        parsed_anchor.signed.version,
        anchor.bytes(),
    )?;
    let mut trusted = TrustedMetadataSet::from_root(anchor.bytes())
        .map_err(|_| TufVerifierError::AuthenticationFailed)?;
    let reset_from_version = receipt.timestamp_snapshot_floor_reset_from_root_version();
    let mut root_chain = Vec::with_capacity(stored_roots.len());
    for (index, bytes) in stored_roots.into_iter().enumerate() {
        let parsed = profile::root(&bytes)?;
        let expected = trusted
            .root()
            .version
            .checked_add(1)
            .ok_or(TufVerifierError::RootVersionExhausted)?;
        if parsed.signed.version != expected {
            return Err(TufVerifierError::AuthenticationFailed);
        }
        let request_name = format!("{expected}.root.json");
        receipt.validate_authenticated_root(index, &request_name, expected, &bytes)?;
        trusted
            .update_root(&bytes)
            .map_err(|_| TufVerifierError::AuthenticationFailed)?;
        root_chain.push(ExactMetadataRole::new(request_name, expected, bytes));
    }
    if trusted.root_bytes() != trusted_root.as_ref() {
        return Err(TufVerifierError::AuthenticationFailed);
    }
    let trusted_root_version = trusted.root().version;
    receipt.validate_authenticated_role(
        "trusted-root.json",
        &format!("{trusted_root_version}.root.json"),
        trusted_root_version,
        &trusted_root,
    )?;
    let reset_binding_change_observed = if let Some(version) = reset_from_version {
        let base_bytes = if parsed_anchor.signed.version == version {
            anchor_root_bytes.as_ref()
        } else {
            root_chain
                .iter()
                .find(|root| root.version() == version)
                .map(ExactMetadataRole::bytes)
                .ok_or(TufVerifierError::AuthenticationFailed)?
        };
        let base = profile::root(base_bytes)?;
        super::verifier::online_role_binding_invalidated(&base.signed, trusted.root())?
    } else {
        false
    };
    receipt.validate_timestamp_snapshot_floor_reset(
        &anchor_root_bytes,
        &root_chain,
        &trusted_root,
        reset_binding_change_observed,
    )?;
    profile::require_fresh(trusted.root(), historical_time)?;

    let parsed_timestamp = profile::timestamp(&timestamp)?;
    receipt.validate_authenticated_role(
        "timestamp.json",
        "timestamp.json",
        parsed_timestamp.signed.version,
        &timestamp,
    )?;
    profile::require_fresh(&parsed_timestamp.signed, historical_time)?;
    trusted
        .update_timestamp(&timestamp, historical_time)
        .map_err(|_| TufVerifierError::AuthenticationFailed)?;

    let parsed_snapshot = profile::snapshot(&snapshot)?;
    let snapshot_name = metadata_name(
        trusted.root().consistent_snapshot,
        parsed_snapshot.signed.version,
        "snapshot",
    );
    receipt.validate_authenticated_role(
        "snapshot.json",
        &snapshot_name,
        parsed_snapshot.signed.version,
        &snapshot,
    )?;
    profile::require_fresh(&parsed_snapshot.signed, historical_time)?;
    trusted
        .update_snapshot(&snapshot, historical_time)
        .map_err(|_| TufVerifierError::AuthenticationFailed)?;

    let parsed_targets = profile::targets(&targets)?;
    let targets_name = metadata_name(
        trusted.root().consistent_snapshot,
        parsed_targets.signed.version,
        "targets",
    );
    receipt.validate_authenticated_role(
        "targets.json",
        &targets_name,
        parsed_targets.signed.version,
        &targets,
    )?;
    profile::require_fresh(&parsed_targets.signed, historical_time)?;
    trusted
        .update_targets(&targets, historical_time)
        .map_err(|_| TufVerifierError::AuthenticationFailed)?;
    let floor_base_root = trusted.root().clone();

    Ok(VerificationState {
        installation_id: String::new(),
        state_root: String::new(),
        clock: ClockSource::System,
        started_at: historical_time,
        trusted,
        anchor_root: ExactMetadataRole::new(
            format!("{}.root.json", parsed_anchor.signed.version),
            parsed_anchor.signed.version,
            anchor_root_bytes,
        ),
        root_chain,
        timestamp_snapshot_floor_base_root: Some(floor_base_root),
        timestamp_snapshot_floor_reset_from_root_version: None,
        timestamp_floor: Some(RoleFloor::new(parsed_timestamp.signed.version, &timestamp)),
        snapshot_floor: Some(RoleFloor::new(parsed_snapshot.signed.version, &snapshot)),
        targets_floor: Some(RoleFloor::new(parsed_targets.signed.version, &targets)),
        timestamp: None,
        snapshot: None,
        targets: None,
        phase: Phase::Root,
    })
}

#[cfg(test)]
pub(super) fn begin_from_selected_for_test(
    authorization: &MetadataStateAuthorization,
    anchor: &EmbeddedTrustRoot,
    stored: StoredMetadataGeneration,
    samples: impl IntoIterator<Item = Timestamp>,
) -> Result<VerificationStep, TufVerifierError> {
    begin_from_selected_with_clock(
        authorization,
        anchor,
        stored,
        ClockSource::scripted(samples),
    )
}
