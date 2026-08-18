use std::collections::BTreeMap;

use jiff::Timestamp;
use sigstore_tuf::metadata::TargetFile;

use super::*;
use crate::distribution::install_state::metadata::{
    MetadataCommitOutcome, MetadataStateAuthorization,
};
use crate::distribution::update_auth::artifact_authorization::begin_artifact_fetch_for_test;
use crate::distribution::update_auth::commit::commit_and_reopen_for_test;
use crate::distribution::update_auth::model::{
    MetadataResponse, PendingMetadataRequest, VerificationStep,
};
use crate::distribution::update_auth::replay::begin_from_selected_for_test;
use crate::distribution::update_auth::test_repository::{
    stable_release_repository, stable_release_repository_with_expiry,
    stable_release_repository_with_mismatched_pointer, stable_release_successor_pair,
    RepositoryFixture, RetainedReleaseMutation,
};
use crate::distribution::update_auth::verifier::begin_from_anchor_for_test;
use crate::distribution::update_auth::ArtifactFetchAuthorizationError;

const INSTALLATION_ID: &str = "7c907c7a-3125-4a40-a8b3-1c125080e46a";

fn instant(value: &str) -> Timestamp {
    value.parse().expect("fixed timestamp")
}

fn make_authorization() -> (tempfile::TempDir, MetadataStateAuthorization) {
    let temp = tempfile::tempdir().expect("temporary state parent");
    let root = temp
        .path()
        .canonicalize()
        .expect("canonical temporary parent")
        .join("state");
    let authorization = MetadataStateAuthorization::for_test_path(&root, INSTALLATION_ID);
    (temp, authorization)
}

fn leaked_anchor(bytes: &[u8]) -> EmbeddedTrustRoot {
    EmbeddedTrustRoot::from_compiled(Box::leak(bytes.to_vec().into_boxed_slice()))
}

fn request(step: VerificationStep, expected: &str) -> PendingMetadataRequest {
    match step {
        VerificationStep::Request(request) => {
            assert_eq!(request.spec().relative_name(), expected);
            request
        }
        VerificationStep::Candidate(_) => panic!("expected metadata request"),
    }
}

fn candidate(
    authorization: &MetadataStateAuthorization,
    anchor: &EmbeddedTrustRoot,
    fixture: &RepositoryFixture,
) -> super::super::VerifiedMetadataCandidate {
    let step = begin_from_anchor_for_test(
        authorization,
        anchor,
        [
            instant("2026-08-18T09:00:00Z"),
            instant("2026-08-18T09:00:01Z"),
        ],
    )
    .expect("compiled anchor authenticates");
    complete_repository(step, fixture)
}

fn successor_candidate(
    authorization: &MetadataStateAuthorization,
    anchor: &EmbeddedTrustRoot,
    fixture: &RepositoryFixture,
) -> super::super::VerifiedMetadataCandidate {
    let selected = crate::distribution::install_state::metadata::read_selected(authorization)
        .expect("selected generation reads")
        .expect("selected generation exists");
    let step = begin_from_selected_for_test(
        authorization,
        anchor,
        selected,
        [
            instant("2026-08-18T09:00:02Z"),
            instant("2026-08-18T09:00:03Z"),
        ],
    )
    .expect("selected generation replays");
    complete_repository(step, fixture)
}

fn complete_repository(
    step: VerificationStep,
    fixture: &RepositoryFixture,
) -> super::super::VerifiedMetadataCandidate {
    let mut step = request(step, "2.root.json")
        .respond(MetadataResponse::ConfirmedNotFound)
        .expect("root chain terminates");
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
        VerificationStep::Request(_) => panic!("complete transcript requested another role"),
    }
}

fn commit_fixture(
    authorization: &MetadataStateAuthorization,
    anchor: &EmbeddedTrustRoot,
    fixture: &RepositoryFixture,
) {
    let candidate = candidate(authorization, anchor, fixture);
    let completed = candidate.verification_completed_at();
    let (outcome, _) =
        commit_and_reopen_for_test(authorization, anchor, candidate, [completed, completed])
            .expect("metadata commits");
    assert_eq!(outcome, MetadataCommitOutcome::Committed { sequence: 1 });
}

#[test]
fn current_time_replay_mints_only_a_generation_bound_release_binding() {
    let fixture = stable_release_repository(true);
    let (_temp, authorization) = make_authorization();
    let anchor = leaked_anchor(&fixture.repository.anchor);
    commit_fixture(&authorization, &anchor, &fixture.repository);

    let targets = authenticate_selected_targets_for_test(
        &authorization,
        &anchor,
        [
            instant("2026-08-18T09:01:00Z"),
            instant("2026-08-18T09:01:01Z"),
        ],
    )
    .expect("selected targets replay fresh");
    let release = targets
        .bind_channel_pointer(&fixture.pointer)
        .expect("pointer cross-binds");
    assert_eq!(release.selected_sequence(), 1);
    assert_eq!(release.version().as_str(), "0.2.0");
    assert_eq!(release.target().as_str(), "aarch64-apple-darwin");
    assert_eq!(release.metadata_versions(), [1, 2, 2, 2]);
    assert_eq!(release.installation_id(), INSTALLATION_ID);
    assert_eq!(release.state_root(), authorization.state_root());
    assert_eq!(release.authenticated_at(), instant("2026-08-18T09:01:01Z"));
    assert_eq!(release.earliest_expiry(), instant("2999-01-01T00:00:00Z"));
    assert_eq!(
        release.pointer().physical_name().as_str(),
        format!(
            "channels/stable/{}.aarch64-apple-darwin.json",
            release.pointer().sha256().as_str()
        )
    );
    assert_eq!(
        release.manifest().physical_name().as_str(),
        format!(
            "releases/v0.2.0/aarch64-apple-darwin/{}.release-manifest.json",
            release.manifest().sha256().as_str()
        )
    );
    assert_eq!(
        release.archive().physical_name().as_str(),
        format!(
            "releases/v0.2.0/aarch64-apple-darwin/{}.hf2q-v0.2.0-aarch64-apple-darwin.zip",
            release.archive().sha256().as_str()
        )
    );
    assert_ne!(release.selected_generation_sha256(), [0; 32]);
}

#[test]
fn artifact_fetch_reauthenticates_under_lock_before_and_after_archive_io() {
    let fixture = stable_release_repository(true);
    let (_temp, authorization) = make_authorization();
    let anchor = leaked_anchor(&fixture.repository.anchor);
    commit_fixture(&authorization, &anchor, &fixture.repository);

    let fetch = begin_artifact_fetch_for_test(
        &authorization,
        &anchor,
        [
            instant("2026-08-18T09:01:00Z"),
            instant("2026-08-18T09:01:01Z"),
        ],
    )
    .expect("initial fetch authority");
    let mut bound = fetch
        .bind_pointer(&fixture.pointer)
        .expect("pointer-bound authority");
    let stage = bound
        .create_archive_stage_for_test([
            instant("2026-08-18T09:01:02Z"),
            instant("2026-08-18T09:01:03Z"),
        ])
        .expect("lock-held pre-archive proof");
    drop(stage);
    assert!(matches!(
        bound.create_archive_stage_for_test([
            instant("2026-08-18T09:01:03Z"),
            instant("2026-08-18T09:01:04Z"),
        ]),
        Err(ArtifactFetchAuthorizationError::ArchiveStageAlreadyCreated)
    ));
    let finalized = bound
        .finalize_for_test([
            instant("2026-08-18T09:01:05Z"),
            instant("2026-08-18T09:01:06Z"),
        ])
        .expect("lock-held post-I/O proof");
    assert_eq!(
        finalized.authenticated_at(),
        instant("2026-08-18T09:01:06Z")
    );
    assert_eq!(finalized.targets().selected_sequence(), 1);

    let fetch = begin_artifact_fetch_for_test(
        &authorization,
        &anchor,
        [
            instant("2026-08-18T09:02:00Z"),
            instant("2026-08-18T09:02:01Z"),
        ],
    )
    .expect("second fetch authority");
    let bound = fetch
        .bind_pointer(&fixture.pointer)
        .expect("second pointer-bound authority");
    assert!(matches!(
        bound.finalize_for_test([
            instant("2026-08-18T09:02:02Z"),
            instant("2026-08-18T09:02:03Z"),
        ]),
        Err(ArtifactFetchAuthorizationError::ArchiveStageMissing)
    ));
}

#[test]
fn artifact_fetch_rejects_cross_phase_clock_rollback_and_generation_drift() {
    let (initial, successor) = stable_release_successor_pair(RetainedReleaseMutation::AppendOnly);
    let (_temp, authorization) = make_authorization();
    let anchor = leaked_anchor(&initial.repository.anchor);
    commit_fixture(&authorization, &anchor, &initial.repository);

    let fetch = begin_artifact_fetch_for_test(
        &authorization,
        &anchor,
        [
            instant("2026-08-18T09:01:00Z"),
            instant("2026-08-18T09:01:01Z"),
        ],
    )
    .expect("fetch authority");
    let mut bound = fetch
        .bind_pointer(&initial.pointer)
        .expect("pointer-bound authority");
    assert!(matches!(
        bound.create_archive_stage_for_test([
            instant("2026-08-18T09:00:58Z"),
            instant("2026-08-18T09:00:59Z")
        ]),
        Err(
            crate::distribution::update_auth::ArtifactFetchAuthorizationError::Authentication(
                TufVerifierError::ClockRollback
            )
        )
    ));

    let fetch = begin_artifact_fetch_for_test(
        &authorization,
        &anchor,
        [
            instant("2026-08-18T09:02:00Z"),
            instant("2026-08-18T09:02:01Z"),
        ],
    )
    .expect("fresh authority");
    let mut bound = fetch
        .bind_pointer(&initial.pointer)
        .expect("bound authority");
    let candidate = successor_candidate(&authorization, &anchor, &successor.repository);
    let completed = candidate.verification_completed_at();
    commit_and_reopen_for_test(&authorization, &anchor, candidate, [completed, completed])
        .expect("successor commits");
    let error = bound
        .create_archive_stage_for_test([
            instant("2026-08-18T09:03:00Z"),
            instant("2026-08-18T09:03:01Z"),
        ])
        .expect_err("generation drift must fail");
    assert!(matches!(
        error,
        crate::distribution::update_auth::ArtifactFetchAuthorizationError::Authentication(
            TufVerifierError::DurableCommitMismatch | TufVerifierError::TargetBinding
        )
    ));
}

#[test]
fn signed_pointer_can_select_only_its_named_older_retained_pair() {
    let (initial, successor) =
        stable_release_successor_pair(RetainedReleaseMutation::AppendOnlySelectOld);
    let (_temp, authorization) = make_authorization();
    let anchor = leaked_anchor(&initial.repository.anchor);
    commit_fixture(&authorization, &anchor, &initial.repository);

    let candidate = successor_candidate(&authorization, &anchor, &successor.repository);
    let completed = candidate.verification_completed_at();
    let (outcome, _) =
        commit_and_reopen_for_test(&authorization, &anchor, candidate, [completed, completed])
            .expect("append-only retained inventory commits");
    assert_eq!(outcome, MetadataCommitOutcome::Committed { sequence: 2 });

    let targets = authenticate_selected_targets_for_test(
        &authorization,
        &anchor,
        [
            instant("2026-08-18T09:01:00Z"),
            instant("2026-08-18T09:01:01Z"),
        ],
    )
    .expect("retained target set authenticates");
    let release = targets
        .bind_channel_pointer(&successor.pointer)
        .expect("pointer selects its exact older retained pair");
    assert_eq!(release.version().as_str(), "0.1.0");
    assert!(release.manifest().logical_name().contains("/v0.1.0/"));
    assert!(release.archive().logical_name().contains("/v0.1.0/"));
}

#[test]
fn no_selection_false_consistent_snapshot_and_clock_faults_fail_closed() {
    let (_temp, empty) = make_authorization();
    let fixture = stable_release_repository(true);
    let anchor = leaked_anchor(&fixture.repository.anchor);
    assert!(matches!(
        authenticate_selected_targets_for_test(
            &empty,
            &anchor,
            [
                instant("2026-08-18T09:01:00Z"),
                instant("2026-08-18T09:01:01Z")
            ]
        ),
        Err(TufVerifierError::NoSelectedMetadata)
    ));

    let false_fixture = stable_release_repository(false);
    let (_temp, authorization) = make_authorization();
    let false_anchor = leaked_anchor(&false_fixture.repository.anchor);
    commit_fixture(&authorization, &false_anchor, &false_fixture.repository);
    assert!(matches!(
        authenticate_selected_targets_for_test(
            &authorization,
            &false_anchor,
            [
                instant("2026-08-18T09:01:00Z"),
                instant("2026-08-18T09:01:01Z")
            ]
        ),
        Err(TufVerifierError::UnsupportedTargetProfile)
    ));

    let fixture = stable_release_repository(true);
    let (_temp, authorization) = make_authorization();
    let anchor = leaked_anchor(&fixture.repository.anchor);
    commit_fixture(&authorization, &anchor, &fixture.repository);
    assert!(matches!(
        authenticate_selected_targets_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T09:02:00Z"),
                instant("2026-08-18T09:01:59Z")
            ]
        ),
        Err(TufVerifierError::ClockRollback)
    ));
}

#[test]
fn expiry_at_the_second_bracket_sample_is_rejected() {
    let fixture = stable_release_repository_with_expiry(true, "2026-08-18T09:02:00Z");
    let (_temp, authorization) = make_authorization();
    let anchor = leaked_anchor(&fixture.repository.anchor);
    commit_fixture(&authorization, &anchor, &fixture.repository);
    assert!(matches!(
        authenticate_selected_targets_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T09:01:59Z"),
                instant("2026-08-18T09:02:00Z")
            ]
        ),
        Err(TufVerifierError::ExpiredMetadata)
    ));
}

#[test]
fn pointer_bytes_and_repeated_descriptors_must_match_exactly() {
    let fixture = stable_release_repository(true);
    let (_temp, authorization) = make_authorization();
    let anchor = leaked_anchor(&fixture.repository.anchor);
    commit_fixture(&authorization, &anchor, &fixture.repository);
    let targets = authenticate_selected_targets_for_test(
        &authorization,
        &anchor,
        [
            instant("2026-08-18T09:01:00Z"),
            instant("2026-08-18T09:01:01Z"),
        ],
    )
    .expect("target set");
    let mut changed = fixture.pointer.clone();
    changed.push(b' ');
    assert!(matches!(
        targets.bind_channel_pointer(&changed),
        Err(TufVerifierError::TargetBinding)
    ));

    let mismatch = stable_release_repository_with_mismatched_pointer();
    let (_temp, authorization) = make_authorization();
    let anchor = leaked_anchor(&mismatch.repository.anchor);
    commit_fixture(&authorization, &anchor, &mismatch.repository);
    let targets = authenticate_selected_targets_for_test(
        &authorization,
        &anchor,
        [
            instant("2026-08-18T09:01:00Z"),
            instant("2026-08-18T09:01:01Z"),
        ],
    )
    .expect("mismatched pointer target still authenticates at TUF layer");
    assert!(matches!(
        targets.bind_channel_pointer(&mismatch.pointer),
        Err(TufVerifierError::TargetBinding)
    ));
}

fn target(length: u64, digest: char) -> TargetFile {
    TargetFile {
        length,
        hashes: BTreeMap::from([("sha256".to_owned(), digest.to_string().repeat(64))]),
        custom: None,
        extra: BTreeMap::new(),
    }
}

fn inventory() -> BTreeMap<String, TargetFile> {
    BTreeMap::from([
        (
            "channels/stable/aarch64-apple-darwin.json".to_owned(),
            target(512, 'a'),
        ),
        (
            "releases/v0.2.0/aarch64-apple-darwin/release-manifest.json".to_owned(),
            target(1024, 'b'),
        ),
        (
            "releases/v0.2.0/aarch64-apple-darwin/hf2q-v0.2.0-aarch64-apple-darwin.zip".to_owned(),
            target(2048, 'c'),
        ),
    ])
}

#[test]
fn bounded_canonical_history_is_allowed_but_orphans_and_unrelated_targets_are_not() {
    let mut retained = inventory();
    retained.insert(
        "releases/v0.1.0/aarch64-apple-darwin/release-manifest.json".to_owned(),
        target(1024, 'd'),
    );
    retained.insert(
        "releases/v0.1.0/aarch64-apple-darwin/hf2q-v0.1.0-aarch64-apple-darwin.zip".to_owned(),
        target(2048, 'e'),
    );
    let (_, releases) = validate_inventory(&retained).expect("complete retained history");
    assert_eq!(releases.len(), 2);
    assert_eq!(releases[0].version.as_str(), "0.1.0");
    assert_eq!(releases[1].version.as_str(), "0.2.0");

    let mut orphan = retained.clone();
    orphan.remove("releases/v0.1.0/aarch64-apple-darwin/hf2q-v0.1.0-aarch64-apple-darwin.zip");
    assert!(validate_inventory(&orphan).is_err());

    let mut unrelated = inventory();
    unrelated.insert("notes.txt".to_owned(), target(1, 'f'));
    assert!(validate_inventory(&unrelated).is_err());
}

#[test]
fn inventory_enforces_role_caps_digests_and_absent_custom_metadata() {
    let mut missing_pointer = inventory();
    missing_pointer.remove("channels/stable/aarch64-apple-darwin.json");
    assert!(validate_inventory(&missing_pointer).is_err());

    let mut zero = inventory();
    zero.get_mut("channels/stable/aarch64-apple-darwin.json")
        .expect("pointer")
        .length = 0;
    assert!(validate_inventory(&zero).is_err());

    let mut archive = inventory();
    archive
        .get_mut("releases/v0.2.0/aarch64-apple-darwin/hf2q-v0.2.0-aarch64-apple-darwin.zip")
        .expect("archive")
        .length = MAX_RELEASE_ARCHIVE_BYTES + 1;
    assert!(validate_inventory(&archive).is_err());

    let mut custom = inventory();
    custom
        .get_mut("channels/stable/aarch64-apple-darwin.json")
        .expect("pointer")
        .custom = Some(serde_json::json!({}));
    assert!(validate_inventory(&custom).is_err());

    let mut uppercase = inventory();
    uppercase
        .get_mut("channels/stable/aarch64-apple-darwin.json")
        .expect("pointer")
        .hashes = BTreeMap::from([("sha256".to_owned(), "A".repeat(64))]);
    assert!(validate_inventory(&uppercase).is_err());

    let mut multiple_hashes = inventory();
    multiple_hashes
        .get_mut("channels/stable/aarch64-apple-darwin.json")
        .expect("pointer")
        .hashes
        .insert("sha512".to_owned(), "b".repeat(128));
    assert!(validate_inventory(&multiple_hashes).is_err());

    let mut missing_sha256 = inventory();
    let pointer = missing_sha256
        .get_mut("channels/stable/aarch64-apple-darwin.json")
        .expect("pointer");
    pointer.hashes.clear();
    pointer.hashes.insert("sha512".to_owned(), "b".repeat(128));
    assert!(validate_inventory(&missing_sha256).is_err());
}

#[test]
fn every_application_target_accepts_its_cap_and_rejects_zero_or_cap_plus_one() {
    let cases = [
        (
            "channels/stable/aarch64-apple-darwin.json",
            MAX_CHANNEL_POINTER_BYTES as u64,
        ),
        (
            "releases/v0.2.0/aarch64-apple-darwin/release-manifest.json",
            MAX_RELEASE_MANIFEST_BYTES as u64,
        ),
        (
            "releases/v0.2.0/aarch64-apple-darwin/hf2q-v0.2.0-aarch64-apple-darwin.zip",
            MAX_RELEASE_ARCHIVE_BYTES,
        ),
    ];

    for (name, cap) in cases {
        let mut at_cap = inventory();
        at_cap.get_mut(name).expect("fixture target").length = cap;
        validate_inventory(&at_cap).expect("exact application target cap is valid");

        let mut zero = inventory();
        zero.get_mut(name).expect("fixture target").length = 0;
        assert!(validate_inventory(&zero).is_err(), "zero length: {name}");

        let mut over_cap = inventory();
        over_cap.get_mut(name).expect("fixture target").length = cap + 1;
        assert!(
            validate_inventory(&over_cap).is_err(),
            "cap plus one: {name}"
        );
    }
}
