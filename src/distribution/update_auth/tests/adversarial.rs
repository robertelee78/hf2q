#[path = "floors.rs"]
mod floors;
#[path = "hostile_profile.rs"]
mod hostile_profile;
#[path = "rotation.rs"]
mod rotation;

use super::{
    authorization, commit_at_recorded_completion, complete_static_transcript, instant, request,
    ROOT,
};
use crate::distribution::install_state::metadata::read_selected;
use crate::distribution::update_auth::model::{
    EmbeddedTrustRoot, MetadataResponse, VerificationStep, MAX_ROOT_ROTATIONS,
};
use crate::distribution::update_auth::replay::begin_from_selected_for_test;
use crate::distribution::update_auth::test_repository::{
    anchor_at_version, multi_rotation, same_key_chain, same_key_chain_with_expiry,
    same_key_lower_roles, static_lower_roles, successive_threshold_rotations, threshold_rotation,
    RepositoryFixture, RotationSignatures,
};
use crate::distribution::update_auth::verifier::begin_from_anchor_for_test;
use crate::distribution::update_auth::verifier::RoleFloor;
use crate::distribution::update_auth::{profile, strict_json, TufVerifierError};

fn leaked_anchor(bytes: &[u8]) -> EmbeddedTrustRoot {
    EmbeddedTrustRoot::from_compiled(Box::leak(bytes.to_vec().into_boxed_slice()))
}

fn complete_fixture(
    mut step: VerificationStep,
    fixture: &RepositoryFixture,
    already_trusted_roots: usize,
) -> crate::distribution::update_auth::VerifiedMetadataCandidate {
    for (index, root) in fixture.roots.iter().enumerate().skip(already_trusted_roots) {
        let expected = format!("{}.root.json", index + 2);
        step = request(step, &expected)
            .respond(MetadataResponse::Found(root.clone().into_boxed_slice()))
            .expect("gapless root authenticates");
    }
    let terminal = format!("{}.root.json", fixture.roots.len() + 2);
    step = request(step, &terminal)
        .respond(MetadataResponse::ConfirmedNotFound)
        .expect("only explicit root not-found terminates the chain");
    step = request(step, "timestamp.json")
        .respond(MetadataResponse::Found(
            fixture.timestamp.clone().into_boxed_slice(),
        ))
        .expect("timestamp authenticates");
    let snapshot_name = if fixture.consistent_snapshot {
        format!("{}.snapshot.json", fixture.metadata_version)
    } else {
        "snapshot.json".to_owned()
    };
    step = request(step, &snapshot_name)
        .respond(MetadataResponse::Found(
            fixture.snapshot.clone().into_boxed_slice(),
        ))
        .expect("snapshot authenticates");
    let targets_name = if fixture.consistent_snapshot {
        format!("{}.targets.json", fixture.metadata_version)
    } else {
        "targets.json".to_owned()
    };
    step = request(step, &targets_name)
        .respond(MetadataResponse::Found(
            fixture.targets.clone().into_boxed_slice(),
        ))
        .expect("targets authenticate");
    match step {
        VerificationStep::Candidate(candidate) => candidate,
        VerificationStep::Request(_) => panic!("complete repository requested another role"),
    }
}
