use super::commit::{commit_and_reopen, recover_after_process_restart};
use super::model::{
    EmbeddedTrustRoot, MetadataResponse, MetadataRoleKind, PendingMetadataRequest, VerificationStep,
};
use super::replay::begin_from_selected;
use super::verifier::{begin_from_anchor_with_clock, ClockSource};
use super::TufVerifierError;
use crate::distribution::install_state::metadata::{
    read_selected, MetadataCommitOutcome, MetadataStateAuthorization,
};

/// Closed role identity for one verifier-issued metadata request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::distribution) enum MetadataRequestKind {
    Root,
    Timestamp,
    Snapshot,
    Targets,
}

/// Borrowed, non-constructible view of the one outstanding verifier request.
///
/// The relative name and byte cap are derived by authenticated parent
/// metadata (or by the gapless next-root rule), never by the transport.
#[derive(Clone, Copy)]
pub(in crate::distribution) struct MetadataRequestView<'a> {
    pending: &'a PendingMetadataRequest,
}

impl std::fmt::Debug for MetadataRequestView<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MetadataRequestView")
            .field("kind", &self.kind())
            .field("relative_name", &self.relative_name())
            .field("maximum_bytes", &self.maximum_bytes())
            .finish()
    }
}

impl MetadataRequestView<'_> {
    pub(in crate::distribution) fn kind(&self) -> MetadataRequestKind {
        match self.pending.spec().role() {
            MetadataRoleKind::Root => MetadataRequestKind::Root,
            MetadataRoleKind::Timestamp => MetadataRequestKind::Timestamp,
            MetadataRoleKind::Snapshot => MetadataRequestKind::Snapshot,
            MetadataRoleKind::Targets => MetadataRequestKind::Targets,
        }
    }

    pub(in crate::distribution) fn relative_name(&self) -> &str {
        self.pending.spec().relative_name()
    }

    pub(in crate::distribution) fn maximum_bytes(&self) -> usize {
        self.pending.spec().maximum_bytes()
    }

    pub(in crate::distribution) fn accepts_confirmed_not_found(&self) -> bool {
        self.kind() == MetadataRequestKind::Root
    }
}

/// One exact transport response. Only a next-root request may accept the
/// explicit absence proof; lower roles are always required.
#[derive(Debug)]
pub(in crate::distribution) enum MetadataFetchResponse {
    Found(Box<[u8]>),
    ConfirmedNotFound,
}

/// Progress from consuming one outstanding request.
#[derive(Debug)]
pub(in crate::distribution) enum MetadataSessionProgress<'a> {
    Request(MetadataUpdateSession<'a>),
    Complete(MetadataCommitOutcome),
}

/// Single-use production metadata-update transcript.
///
/// The session owns the verifier state hidden inside `PendingMetadataRequest`.
/// It cannot be cloned, serialized, reordered, or reused after a response.
pub(in crate::distribution) struct MetadataUpdateSession<'a> {
    authorization: &'a MetadataStateAuthorization,
    anchor: &'a EmbeddedTrustRoot,
    pending: PendingMetadataRequest,
}

impl std::fmt::Debug for MetadataUpdateSession<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MetadataUpdateSession")
            .field("request", &self.request())
            .finish_non_exhaustive()
    }
}

impl<'a> MetadataUpdateSession<'a> {
    pub(in crate::distribution) fn request(&self) -> MetadataRequestView<'_> {
        MetadataRequestView {
            pending: &self.pending,
        }
    }

    pub(in crate::distribution) fn respond(
        self,
        response: MetadataFetchResponse,
    ) -> Result<MetadataSessionProgress<'a>, TufVerifierError> {
        if matches!(response, MetadataFetchResponse::ConfirmedNotFound)
            && !self.request().accepts_confirmed_not_found()
        {
            return Err(TufVerifierError::RequiredMetadataMissing);
        }
        let response = match response {
            MetadataFetchResponse::Found(bytes) => MetadataResponse::Found(bytes),
            MetadataFetchResponse::ConfirmedNotFound => MetadataResponse::ConfirmedNotFound,
        };
        let next = self.pending.respond(response)?;
        progress(self.authorization, self.anchor, next)
    }
}

/// Start a fresh network transcript from a durably authenticated floor.
///
/// Restart cleanup is deliberately part of this boundary: a fresh process
/// repairs the selected floor and discards only the exact never-selected
/// transaction before it begins a wholly fresh transcript.
pub(in crate::distribution) fn begin_metadata_update<'a>(
    authorization: &'a MetadataStateAuthorization,
    anchor: &'a EmbeddedTrustRoot,
) -> Result<MetadataUpdateSession<'a>, TufVerifierError> {
    let _ = recover_after_process_restart(authorization, anchor)?;
    let step = match read_selected(authorization)? {
        Some(selected) => begin_from_selected(authorization, anchor, selected)?,
        None => begin_from_anchor_with_clock(authorization, anchor, ClockSource::System)?,
    };
    match progress(authorization, anchor, step)? {
        MetadataSessionProgress::Request(session) => Ok(session),
        MetadataSessionProgress::Complete(_) => Err(TufVerifierError::IncompleteTranscript),
    }
}

fn progress<'a>(
    authorization: &'a MetadataStateAuthorization,
    anchor: &'a EmbeddedTrustRoot,
    step: VerificationStep,
) -> Result<MetadataSessionProgress<'a>, TufVerifierError> {
    match step {
        VerificationStep::Request(pending) => {
            Ok(MetadataSessionProgress::Request(MetadataUpdateSession {
                authorization,
                anchor,
                pending,
            }))
        }
        VerificationStep::Candidate(candidate) => {
            let (outcome, _durable) = commit_and_reopen(authorization, anchor, candidate)?;
            Ok(MetadataSessionProgress::Complete(outcome))
        }
    }
}
