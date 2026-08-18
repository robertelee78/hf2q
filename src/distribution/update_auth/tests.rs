use jiff::Timestamp;
use sha2::{Digest, Sha256};

use super::commit::{commit_and_reopen, commit_and_reopen_for_test, recover_after_process_restart};
use super::model::{
    EmbeddedTrustRoot, MetadataResponse, PendingMetadataRequest, VerificationStep,
    MAX_TIMESTAMP_BYTES,
};
use super::profile;
use super::replay::begin_from_selected_for_test;
use super::strict_json;
use super::verifier::begin_from_anchor_for_test;
use super::TufVerifierError;
use crate::distribution::install_state::metadata::{
    commit_candidate_for_test, read_selected, Barrier, FaultPlan, MetadataCommitOutcome,
    MetadataJournalError, MetadataRestartCleanup, MetadataStateAuthorization,
};

mod adversarial;
#[path = "tests/commit_recovery.rs"]
mod commit_recovery;
#[path = "tests/corpus_interop.rs"]
mod corpus_interop;
#[path = "tests/protocol_profile.rs"]
mod protocol_profile;

const ROOT: &[u8] = include_bytes!("testdata/tuf-v1/root-v1.json");
const TIMESTAMP: &[u8] = include_bytes!("testdata/tuf-v1/timestamp-v2-normalized.json");
const SNAPSHOT: &[u8] = include_bytes!("testdata/tuf-v1/snapshot-v2.json");
const TARGETS: &[u8] = include_bytes!("testdata/tuf-v1/targets-v2.json");
const PYTHON_ROOT_V1: &[u8] = include_bytes!("testdata/python-tuf-v1/1.root.json");
const PYTHON_ROOT_V2: &[u8] = include_bytes!("testdata/python-tuf-v1/2.root.json");
const PYTHON_TIMESTAMP_V2: &[u8] = include_bytes!("testdata/python-tuf-v1/timestamp.json");
const PYTHON_SNAPSHOT_V2: &[u8] = include_bytes!("testdata/python-tuf-v1/2.snapshot.json");
const PYTHON_TARGETS_V2: &[u8] = include_bytes!("testdata/python-tuf-v1/2.targets.json");
const INSTALLATION_ID: &str = "7c907c7a-3125-4a40-a8b3-1c125080e46a";

fn instant(value: &str) -> Timestamp {
    value.parse().expect("fixed canonical timestamp")
}

fn authorization() -> (tempfile::TempDir, MetadataStateAuthorization) {
    let temp = tempfile::tempdir().expect("temporary root parent");
    let root = temp
        .path()
        .canonicalize()
        .expect("canonical temporary root")
        .join("state");
    let authorization = MetadataStateAuthorization::for_test_path(&root, INSTALLATION_ID);
    (temp, authorization)
}

fn request(step: VerificationStep, expected: &str) -> PendingMetadataRequest {
    match step {
        VerificationStep::Request(request) => {
            assert_eq!(request.spec().relative_name(), expected);
            request
        }
        VerificationStep::Candidate(_) => panic!("expected another metadata request"),
    }
}

fn complete_static_transcript(step: VerificationStep) -> super::VerifiedMetadataCandidate {
    let root = request(step, "2.root.json");
    let timestamp = request(
        root.respond(MetadataResponse::ConfirmedNotFound)
            .expect("explicit root-chain termination"),
        "timestamp.json",
    );
    let snapshot = request(
        timestamp
            .respond(MetadataResponse::Found(TIMESTAMP.into()))
            .expect("timestamp authenticates"),
        "snapshot.json",
    );
    let targets = request(
        snapshot
            .respond(MetadataResponse::Found(SNAPSHOT.into()))
            .expect("snapshot authenticates"),
        "targets.json",
    );
    match targets
        .respond(MetadataResponse::Found(TARGETS.into()))
        .expect("targets authenticates")
    {
        VerificationStep::Candidate(candidate) => candidate,
        VerificationStep::Request(_) => panic!("complete transcript requested extra metadata"),
    }
}

fn commit_at_recorded_completion(
    authorization: &MetadataStateAuthorization,
    anchor: &EmbeddedTrustRoot,
    candidate: super::VerifiedMetadataCandidate,
) -> Result<
    (
        MetadataCommitOutcome,
        super::commit::DurableMetadataBaseline,
    ),
    TufVerifierError,
> {
    let completed = candidate.verification_completed_at();
    commit_and_reopen_for_test(authorization, anchor, candidate, [completed, completed])
}
