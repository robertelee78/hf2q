use std::collections::BTreeSet;
#[cfg(test)]
use std::collections::VecDeque;

use jiff::Timestamp;
use sha2::{Digest, Sha256};
use sigstore_tuf::trusted::TrustedMetadataSet;

use super::model::{
    EmbeddedTrustRoot, ExactMetadataRole, MetadataResponse, MetadataRoleKind,
    PendingMetadataRequest, RequestSpec, VerificationStep, VerifiedMetadataCandidate,
    MAX_ROOT_ROTATIONS,
};
use super::{profile, TufVerifierError};
use crate::distribution::install_state::metadata::MetadataStateAuthorization;

#[derive(Debug)]
pub(super) struct RoleFloor {
    version: u64,
    sha256: [u8; 32],
}

impl RoleFloor {
    pub(super) fn new(version: u64, bytes: &[u8]) -> Self {
        Self {
            version,
            sha256: Sha256::digest(bytes).into(),
        }
    }

    pub(super) fn require(&self, version: u64, bytes: &[u8]) -> Result<(), TufVerifierError> {
        let digest: [u8; 32] = Sha256::digest(bytes).into();
        if version < self.version || (version == self.version && digest != self.sha256) {
            return Err(TufVerifierError::RollbackOrEquivocation);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(super) enum ClockSource {
    System,
    Recorded {
        samples: [Timestamp; 2],
        next: usize,
    },
    #[cfg(test)]
    Scripted(VecDeque<Timestamp>),
}

impl ClockSource {
    pub(super) fn sample(&mut self) -> Result<Timestamp, TufVerifierError> {
        match self {
            Self::System => Ok(Timestamp::now()),
            Self::Recorded { samples, next } => {
                let sample = samples
                    .get(*next)
                    .copied()
                    .ok_or(TufVerifierError::ClockRollback)?;
                *next += 1;
                Ok(sample)
            }
            #[cfg(test)]
            Self::Scripted(samples) => samples.pop_front().ok_or(TufVerifierError::ClockRollback),
        }
    }

    #[cfg(test)]
    pub(super) fn scripted(samples: impl IntoIterator<Item = Timestamp>) -> Self {
        Self::Scripted(samples.into_iter().collect())
    }

    pub(super) fn fixed(started: Timestamp, completed: Timestamp) -> Self {
        Self::Recorded {
            samples: [started, completed],
            next: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Phase {
    Root,
    Timestamp,
    Snapshot,
    Targets,
}

#[derive(Debug)]
pub(super) struct VerificationState {
    pub(super) installation_id: String,
    pub(super) state_root: String,
    pub(super) clock: ClockSource,
    pub(super) started_at: Timestamp,
    pub(super) trusted: TrustedMetadataSet,
    pub(super) anchor_root: ExactMetadataRole,
    pub(super) root_chain: Vec<ExactMetadataRole>,
    pub(super) timestamp_snapshot_floor_base_root: Option<sigstore_tuf::metadata::Root>,
    pub(super) timestamp_snapshot_floor_reset_from_root_version: Option<u64>,
    pub(super) timestamp_floor: Option<RoleFloor>,
    pub(super) snapshot_floor: Option<RoleFloor>,
    pub(super) targets_floor: Option<RoleFloor>,
    pub(super) timestamp: Option<ExactMetadataRole>,
    pub(super) snapshot: Option<ExactMetadataRole>,
    pub(super) targets: Option<ExactMetadataRole>,
    pub(super) phase: Phase,
}

/// Initial bootstrap is deliberately private to this bounded context. A
/// production caller will exist only after ADR-045 lands real compiled root
/// bytes and the offline-key runbook.
pub(super) fn begin_from_anchor_with_clock(
    authorization: &MetadataStateAuthorization,
    anchor: &EmbeddedTrustRoot,
    mut clock: ClockSource,
) -> Result<VerificationStep, TufVerifierError> {
    let parsed = profile::root(anchor.bytes())?;
    let trusted = TrustedMetadataSet::from_root(anchor.bytes())
        .map_err(|_| TufVerifierError::AuthenticationFailed)?;
    if trusted.root().version != parsed.signed.version || trusted.root_bytes() != anchor.bytes() {
        return Err(TufVerifierError::AuthenticationFailed);
    }
    let started_at = clock.sample()?;
    let state = VerificationState {
        installation_id: authorization.installation_id().to_owned(),
        state_root: authorization.state_root().to_owned(),
        clock,
        started_at,
        trusted,
        anchor_root: exact_root(anchor.bytes(), parsed.signed.version),
        root_chain: Vec::new(),
        timestamp_snapshot_floor_base_root: None,
        timestamp_snapshot_floor_reset_from_root_version: None,
        timestamp_floor: None,
        snapshot_floor: None,
        targets_floor: None,
        timestamp: None,
        snapshot: None,
        targets: None,
        phase: Phase::Root,
    };
    Ok(request_root(state)?)
}

pub(super) fn request_root(state: VerificationState) -> Result<VerificationStep, TufVerifierError> {
    let next = state
        .trusted
        .root()
        .version
        .checked_add(1)
        .ok_or(TufVerifierError::RootVersionExhausted)?;
    Ok(request(
        state,
        MetadataRoleKind::Root,
        format!("{next}.root.json"),
    ))
}

fn request(state: VerificationState, kind: MetadataRoleKind, name: String) -> VerificationStep {
    VerificationStep::Request(PendingMetadataRequest {
        state,
        spec: RequestSpec::new(kind, name),
    })
}

pub(super) fn respond(
    pending: PendingMetadataRequest,
    response: MetadataResponse,
) -> Result<VerificationStep, TufVerifierError> {
    let PendingMetadataRequest { mut state, spec } = pending;
    if spec.role()
        != match state.phase {
            Phase::Root => MetadataRoleKind::Root,
            Phase::Timestamp => MetadataRoleKind::Timestamp,
            Phase::Snapshot => MetadataRoleKind::Snapshot,
            Phase::Targets => MetadataRoleKind::Targets,
        }
    {
        return Err(TufVerifierError::UnexpectedResponse);
    }
    match (state.phase, response) {
        (Phase::Root, MetadataResponse::ConfirmedNotFound) => {
            profile::require_fresh(state.trusted.root(), state.started_at)?;
            apply_online_role_recovery(&mut state)?;
            state.phase = Phase::Timestamp;
            Ok(request(
                state,
                MetadataRoleKind::Timestamp,
                "timestamp.json".to_owned(),
            ))
        }
        (Phase::Root, MetadataResponse::Found(bytes)) => {
            if state.root_chain.len() >= MAX_ROOT_ROTATIONS {
                return Err(TufVerifierError::RootRotationLimit);
            }
            require_response_bound(&spec, &bytes)?;
            let parsed = profile::root(&bytes)?;
            let expected = state
                .trusted
                .root()
                .version
                .checked_add(1)
                .ok_or(TufVerifierError::RootVersionExhausted)?;
            if spec.relative_name() != format!("{expected}.root.json")
                || parsed.signed.version != expected
            {
                return Err(TufVerifierError::UnexpectedResponse);
            }
            state
                .trusted
                .update_root(&bytes)
                .map_err(|_| TufVerifierError::AuthenticationFailed)?;
            state.root_chain.push(ExactMetadataRole::new(
                spec.relative_name().to_owned(),
                expected,
                bytes,
            ));
            request_root(state)
        }
        (Phase::Timestamp, MetadataResponse::Found(bytes)) => {
            require_response_bound(&spec, &bytes)?;
            if spec.relative_name() != "timestamp.json" {
                return Err(TufVerifierError::UnexpectedResponse);
            }
            let parsed = profile::timestamp(&bytes)?;
            if let Some(floor) = &state.timestamp_floor {
                floor.require(parsed.signed.version, &bytes)?;
            }
            let result = state.trusted.update_timestamp(&bytes, state.started_at);
            match result {
                Ok(()) => {}
                Err(sigstore_tuf::Error::EqualVersion { .. })
                    if state.timestamp_floor.as_ref().is_some_and(|floor| {
                        floor.require(parsed.signed.version, &bytes).is_ok()
                    }) => {}
                Err(_) => return Err(TufVerifierError::AuthenticationFailed),
            }
            profile::require_fresh(&parsed.signed, state.started_at)?;
            let version = parsed
                .signed
                .snapshot_meta()
                .ok_or(TufVerifierError::MalformedMetadata)?
                .version;
            state.timestamp = Some(ExactMetadataRole::new(
                spec.relative_name().to_owned(),
                parsed.signed.version,
                bytes,
            ));
            state.phase = Phase::Snapshot;
            let name = metadata_name(
                state.trusted.root().consistent_snapshot,
                version,
                "snapshot",
            );
            Ok(request(state, MetadataRoleKind::Snapshot, name))
        }
        (Phase::Snapshot, MetadataResponse::Found(bytes)) => {
            require_response_bound(&spec, &bytes)?;
            let parsed = profile::snapshot(&bytes)?;
            let expected_version = state
                .trusted
                .timestamp()
                .and_then(|timestamp| timestamp.snapshot_meta())
                .ok_or(TufVerifierError::IncompleteTranscript)?
                .version;
            let expected_name = metadata_name(
                state.trusted.root().consistent_snapshot,
                expected_version,
                "snapshot",
            );
            if spec.relative_name() != expected_name || parsed.signed.version != expected_version {
                return Err(TufVerifierError::UnexpectedResponse);
            }
            if let Some(floor) = &state.snapshot_floor {
                floor.require(parsed.signed.version, &bytes)?;
            }
            state
                .trusted
                .update_snapshot(&bytes, state.started_at)
                .map_err(|_| TufVerifierError::AuthenticationFailed)?;
            profile::require_fresh(&parsed.signed, state.started_at)?;
            let version = parsed
                .signed
                .meta
                .get("targets.json")
                .ok_or(TufVerifierError::MalformedMetadata)?
                .version;
            state.snapshot = Some(ExactMetadataRole::new(
                spec.relative_name().to_owned(),
                parsed.signed.version,
                bytes,
            ));
            state.phase = Phase::Targets;
            let name = metadata_name(state.trusted.root().consistent_snapshot, version, "targets");
            Ok(request(state, MetadataRoleKind::Targets, name))
        }
        (Phase::Targets, MetadataResponse::Found(bytes)) => {
            require_response_bound(&spec, &bytes)?;
            let parsed = profile::targets(&bytes)?;
            let expected_version = state
                .trusted
                .snapshot()
                .and_then(|snapshot| snapshot.meta.get("targets.json"))
                .ok_or(TufVerifierError::IncompleteTranscript)?
                .version;
            let expected_name = metadata_name(
                state.trusted.root().consistent_snapshot,
                expected_version,
                "targets",
            );
            if spec.relative_name() != expected_name || parsed.signed.version != expected_version {
                return Err(TufVerifierError::UnexpectedResponse);
            }
            if let Some(floor) = &state.targets_floor {
                floor.require(parsed.signed.version, &bytes)?;
            }
            state
                .trusted
                .update_targets(&bytes, state.started_at)
                .map_err(|_| TufVerifierError::AuthenticationFailed)?;
            profile::require_fresh(&parsed.signed, state.started_at)?;
            state.targets = Some(ExactMetadataRole::new(
                spec.relative_name().to_owned(),
                parsed.signed.version,
                bytes,
            ));
            finish(state)
        }
        (_, MetadataResponse::ConfirmedNotFound) => Err(TufVerifierError::RequiredMetadataMissing),
    }
}

fn finish(mut state: VerificationState) -> Result<VerificationStep, TufVerifierError> {
    let completed_at = state.clock.sample()?;
    if completed_at < state.started_at {
        return Err(TufVerifierError::ClockRollback);
    }
    profile::require_fresh(state.trusted.root(), completed_at)?;
    profile::require_fresh(
        state
            .trusted
            .timestamp()
            .ok_or(TufVerifierError::IncompleteTranscript)?,
        completed_at,
    )?;
    profile::require_fresh(
        state
            .trusted
            .snapshot()
            .ok_or(TufVerifierError::IncompleteTranscript)?,
        completed_at,
    )?;
    profile::require_fresh(
        state
            .trusted
            .targets()
            .ok_or(TufVerifierError::IncompleteTranscript)?,
        completed_at,
    )?;
    let trusted_root = ExactMetadataRole::new(
        format!("{}.root.json", state.trusted.root().version),
        state.trusted.root().version,
        state.trusted.root_bytes().to_vec().into_boxed_slice(),
    );
    Ok(VerificationStep::Candidate(VerifiedMetadataCandidate::new(
        state.installation_id,
        state.state_root,
        state.started_at,
        completed_at,
        state.anchor_root,
        state.root_chain,
        trusted_root,
        state.timestamp_snapshot_floor_reset_from_root_version,
        state
            .timestamp
            .take()
            .ok_or(TufVerifierError::IncompleteTranscript)?,
        state
            .snapshot
            .take()
            .ok_or(TufVerifierError::IncompleteTranscript)?,
        state
            .targets
            .take()
            .ok_or(TufVerifierError::IncompleteTranscript)?,
    )))
}

fn apply_online_role_recovery(state: &mut VerificationState) -> Result<(), TufVerifierError> {
    if state.timestamp_floor.is_none() && state.snapshot_floor.is_none() {
        return Ok(());
    }
    let base = state
        .timestamp_snapshot_floor_base_root
        .as_ref()
        .ok_or(TufVerifierError::AuthenticationFailed)?;
    // TUF 1.0.36 root-update step 11 permits recovery from an online
    // fast-forward compromise after root authority revokes the old online
    // quorum. Compare the selected predecessor directly with the final root:
    // an additive or transient rotation cannot erase rollback floors.
    if online_role_binding_invalidated(base, state.trusted.root())? {
        state.timestamp_snapshot_floor_reset_from_root_version = Some(base.version);
        state.timestamp_floor = None;
        state.snapshot_floor = None;
    }
    Ok(())
}

pub(super) fn online_role_binding_invalidated(
    old: &sigstore_tuf::metadata::Root,
    new: &sigstore_tuf::metadata::Root,
) -> Result<bool, TufVerifierError> {
    Ok(role_binding_invalidated(old, new, "timestamp")?
        || role_binding_invalidated(old, new, "snapshot")?)
}

fn role_binding_invalidated(
    old: &sigstore_tuf::metadata::Root,
    new: &sigstore_tuf::metadata::Root,
    role: &str,
) -> Result<bool, TufVerifierError> {
    let old_binding = old.role(role).ok_or(TufVerifierError::MalformedMetadata)?;
    let new_binding = new.role(role).ok_or(TufVerifierError::MalformedMetadata)?;
    let old_ids: BTreeSet<_> = old_binding.keyids.iter().collect();
    let surviving = new_binding
        .keyids
        .iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|key_id| old_ids.contains(key_id) && old.keys.get(*key_id) == new.keys.get(*key_id))
        .count();
    Ok(surviving < new_binding.threshold)
}

fn require_response_bound(spec: &RequestSpec, bytes: &[u8]) -> Result<(), TufVerifierError> {
    if bytes.is_empty() || bytes.len() > spec.maximum_bytes() {
        return Err(TufVerifierError::MetadataSize);
    }
    Ok(())
}

pub(super) fn metadata_name(consistent: bool, version: u64, role: &str) -> String {
    if consistent {
        format!("{version}.{role}.json")
    } else {
        format!("{role}.json")
    }
}

fn exact_root(bytes: &[u8], version: u64) -> ExactMetadataRole {
    ExactMetadataRole::new(
        format!("{version}.root.json"),
        version,
        bytes.to_vec().into_boxed_slice(),
    )
}

#[cfg(test)]
pub(super) fn begin_from_anchor_for_test(
    authorization: &MetadataStateAuthorization,
    anchor: &EmbeddedTrustRoot,
    samples: impl IntoIterator<Item = Timestamp>,
) -> Result<VerificationStep, TufVerifierError> {
    begin_from_anchor_with_clock(authorization, anchor, ClockSource::scripted(samples))
}
