use std::os::unix::process::ExitStatusExt;

use super::*;
use crate::distribution::install_state::{
    abort_after_prepared_barrier, fail_after_prepared_barrier, observed_prepared_barriers,
    reset_observed_prepared_barriers, set_prepared_precommit_hook, PreparedVersionError,
};
use crate::distribution::prepared_release::{
    prepare_release_for_test, prepare_release_for_test_with_clocks, PreparedReleaseOutcome,
};
#[cfg(target_os = "macos")]
#[test]
fn prepared_version_publication_is_exact_and_idempotently_recoverable() {
    let (manifest, manifest_bytes) = manifest();
    let archive_bytes = archive(&manifest_bytes, CompressionMethod::Deflated);
    let fixture = stable_release_repository_for_artifacts(&manifest_bytes, &archive_bytes);
    let (temp, authorization) = make_authorization();
    let root = temp
        .path()
        .canonicalize()
        .expect("canonical parent")
        .join("state");
    let anchor = leaked_anchor(&fixture.repository.anchor);
    commit_fixture(&authorization, &anchor, &fixture.repository);

    let release = bundle(
        &authorization,
        &anchor,
        &fixture.pointer,
        manifest.clone(),
        manifest_bytes.clone(),
        &archive_bytes,
    );
    let outcome = prepare_release_for_test(release).expect("first prepared publication");
    assert!(matches!(outcome, PreparedReleaseOutcome::Prepared(_)));
    let version = root.join("versions/0.2.0");
    assert_eq!(
        std::fs::read(version.join("release-manifest.json")).expect("published manifest"),
        manifest_bytes
    );
    assert!(version.join("version-installation.json").is_file());
    assert!(!root.join("activations").exists());
    assert!(!root.join("current").exists());

    let retry = bundle(
        &authorization,
        &anchor,
        &fixture.pointer,
        manifest,
        std::fs::read(version.join("release-manifest.json")).expect("retry manifest"),
        &archive_bytes,
    );
    let recovered = prepare_release_for_test(retry).expect("exact published recovery");
    assert!(matches!(
        recovered,
        PreparedReleaseOutcome::AlreadyPrepared(_)
    ));
    assert!(!root.join("activations").exists());
    assert!(!root.join("current").exists());
}

#[cfg(target_os = "macos")]
#[test]
fn authenticated_prepared_version_feeds_the_existing_first_activation_boundary() {
    let (manifest, manifest_bytes) = manifest();
    let archive_bytes = archive(&manifest_bytes, CompressionMethod::Deflated);
    let fixture = stable_release_repository_for_artifacts(&manifest_bytes, &archive_bytes);
    let (temp, authorization) = make_authorization();
    let root = temp
        .path()
        .canonicalize()
        .expect("canonical parent")
        .join("state");
    let anchor = leaked_anchor(&fixture.repository.anchor);
    commit_fixture(&authorization, &anchor, &fixture.repository);
    let release = bundle(
        &authorization,
        &anchor,
        &fixture.pointer,
        manifest,
        manifest_bytes,
        &archive_bytes,
    );
    let PreparedReleaseOutcome::Prepared(prepared) =
        prepare_release_for_test(release).expect("authenticated version publication")
    else {
        panic!("new version must not be reported as already prepared")
    };
    let explicit = crate::distribution::install_state::ExplicitRootAuthorization::new(&root)
        .expect("explicit activation root");
    let identity =
        crate::distribution::install_state::open_existing_installation_identity(explicit)
            .expect("open prepared installation identity")
            .expect("prepared installation identity exists");
    let preparation =
        crate::distribution::install_state::prepare_first_activation(identity, prepared)
            .expect("prepared capability enters activation boundary");
    let crate::distribution::install_state::FirstActivationPreparation::Ready(preparation) =
        preparation
    else {
        panic!("first activation cannot already be committed")
    };
    assert_eq!(
        preparation
            .commit()
            .expect("first activation commit")
            .sequence,
        1
    );
    assert_eq!(
        std::fs::read_link(root.join("current")).expect("current link"),
        std::path::PathBuf::from("activations/00000000000000000001")
    );
}

#[cfg(target_os = "macos")]
fn is_prepared_durability_unknown(error: &PreparedReleaseError) -> bool {
    match error {
        PreparedReleaseError::PreparedVersionDurabilityUnknown { .. }
        | PreparedReleaseError::Publication(PreparedVersionError::PublishedDurabilityUnknown {
            ..
        })
        | PreparedReleaseError::PreparedCommit(
            crate::distribution::update_auth::PreparedVersionCommitError::Publication(
                PreparedVersionError::PublishedDurabilityUnknown { .. },
            ),
        ) => true,
        _ => false,
    }
}

#[path = "prepared_publication_cases/fault_cases.rs"]
mod fault_cases;

#[cfg(target_os = "macos")]
#[test]
fn prepared_intent_and_unactivated_version_both_block_metadata_advancement() {
    let (manifest, manifest_bytes) = manifest();
    let archive_bytes = archive(&manifest_bytes, CompressionMethod::Stored);
    let initial = stable_release_repository_for_artifacts(&manifest_bytes, &archive_bytes);
    let successor = stable_release_successor_for_artifacts(&manifest_bytes, &archive_bytes);
    let (temp, authorization) = make_authorization();
    let root = temp
        .path()
        .canonicalize()
        .expect("canonical parent")
        .join("state");
    let anchor = leaked_anchor(&initial.repository.anchor);
    commit_fixture(&authorization, &anchor, &initial.repository);
    let selected = root.join("update/metadata/current.json");
    let selected_before = std::fs::read(&selected).expect("selected metadata selector");
    let metadata_names_before: std::collections::BTreeSet<_> =
        std::fs::read_dir(root.join("update/metadata"))
            .expect("metadata inventory")
            .map(|entry| entry.unwrap().file_name())
            .collect();

    reset_observed_prepared_barriers();
    fail_after_prepared_barrier(1);
    let release = bundle(
        &authorization,
        &anchor,
        &initial.pointer,
        manifest.clone(),
        manifest_bytes.clone(),
        &archive_bytes,
    );
    assert!(prepare_release_for_test(release).is_err());
    assert!(
        std::fs::read_dir(root.join("update/prepared"))
            .expect("prepared intent inventory")
            .next()
            .is_some(),
        "the first publication barrier leaves a durable marker intent"
    );

    reset_observed_prepared_barriers();
    let candidate = successor_candidate(&authorization, &anchor, &successor.repository);
    let completed = candidate.verification_completed_at();
    assert!(matches!(
        commit_and_reopen_for_test(&authorization, &anchor, candidate, [completed, completed]),
        Err(TufVerifierError::Journal(
            MetadataJournalError::InstallState(
                crate::distribution::install_state::InstallStateError::InvalidLayout(
                    "prepared-version intent blocks metadata advancement"
                )
            )
        ))
    ));
    assert_eq!(
        std::fs::read(&selected).expect("selector after rejected advancement"),
        selected_before
    );
    let metadata_names_after: std::collections::BTreeSet<_> =
        std::fs::read_dir(root.join("update/metadata"))
            .expect("metadata inventory after rejection")
            .map(|entry| entry.unwrap().file_name())
            .collect();
    assert_eq!(metadata_names_after, metadata_names_before);

    let retry = bundle(
        &authorization,
        &anchor,
        &initial.pointer,
        manifest,
        manifest_bytes,
        &archive_bytes,
    );
    drop(prepare_release_for_test(retry).expect("prepared intent exact retry"));
    assert!(root.join("versions/0.2.0").is_dir());
    let candidate = successor_candidate(&authorization, &anchor, &successor.repository);
    let completed = candidate.verification_completed_at();
    assert!(matches!(
        commit_and_reopen_for_test(&authorization, &anchor, candidate, [completed, completed]),
        Err(TufVerifierError::Journal(
            MetadataJournalError::InstallState(
                crate::distribution::install_state::InstallStateError::InvalidLayout(
                    "unactivated prepared version blocks metadata advancement"
                )
            )
        ))
    ));
    assert_eq!(
        std::fs::read(selected).expect("selector after final rejected advancement"),
        selected_before
    );
    assert!(!root.join("activations").exists());
    assert!(!root.join("current").exists());
}

#[cfg(target_os = "macos")]
#[test]
fn prepared_publication_clock_boundaries_fail_closed() {
    #[derive(Clone, Copy, Debug)]
    enum Case {
        CommitRollback,
        CommitExpiry,
        DelayedFresh,
        PostcommitRollback,
        PostcommitExpiry,
        FinalizationRollback,
        FinalizationExpiry,
    }

    for case in [
        Case::CommitRollback,
        Case::CommitExpiry,
        Case::DelayedFresh,
        Case::PostcommitRollback,
        Case::PostcommitExpiry,
        Case::FinalizationRollback,
        Case::FinalizationExpiry,
    ] {
        let (manifest, manifest_bytes) = manifest();
        let archive_bytes = archive(&manifest_bytes, CompressionMethod::Stored);
        let fixture = stable_release_repository_for_artifacts_with_expiry(
            &manifest_bytes,
            &archive_bytes,
            "2099-01-01T00:00:00Z",
        );
        let (temp, authorization) = make_authorization();
        let root = temp
            .path()
            .canonicalize()
            .expect("canonical parent")
            .join("state");
        let anchor = leaked_anchor(&fixture.repository.anchor);
        commit_fixture(&authorization, &anchor, &fixture.repository);
        let release = bundle(
            &authorization,
            &anchor,
            &fixture.pointer,
            manifest,
            manifest_bytes,
            &archive_bytes,
        );
        let before = vec![
            instant("2098-01-01T00:00:00Z"),
            instant("2098-01-01T00:00:01Z"),
        ];
        let (selector, after) = match case {
            Case::CommitRollback => (
                vec![instant("2098-01-01T00:00:00Z")],
                vec![
                    instant("2098-01-01T00:00:03Z"),
                    instant("2098-01-01T00:00:04Z"),
                    instant("2098-01-01T00:00:05Z"),
                ],
            ),
            Case::CommitExpiry => (
                vec![instant("2099-01-01T00:00:00Z")],
                vec![
                    instant("2099-01-01T00:00:01Z"),
                    instant("2099-01-01T00:00:02Z"),
                    instant("2099-01-01T00:00:03Z"),
                ],
            ),
            Case::DelayedFresh => (
                vec![instant("2098-01-01T00:00:02Z")],
                vec![
                    instant("2098-01-01T00:00:03Z"),
                    instant("2098-01-01T00:00:04Z"),
                    instant("2098-01-01T00:00:05Z"),
                ],
            ),
            Case::PostcommitRollback => (
                vec![instant("2098-01-01T00:00:02Z")],
                vec![
                    instant("2098-01-01T00:00:01Z"),
                    instant("2098-01-01T00:00:03Z"),
                    instant("2098-01-01T00:00:04Z"),
                ],
            ),
            Case::PostcommitExpiry => (
                vec![instant("2098-01-01T00:00:02Z")],
                vec![
                    instant("2098-12-31T23:59:59Z"),
                    instant("2099-01-01T00:00:00Z"),
                    instant("2099-01-01T00:00:01Z"),
                ],
            ),
            Case::FinalizationRollback => (
                vec![instant("2098-01-01T00:00:02Z")],
                vec![
                    instant("2098-01-01T00:00:03Z"),
                    instant("2098-01-01T00:00:04Z"),
                    instant("2098-01-01T00:00:03Z"),
                ],
            ),
            Case::FinalizationExpiry => (
                vec![instant("2098-01-01T00:00:02Z")],
                vec![
                    instant("2098-12-31T23:59:57Z"),
                    instant("2098-12-31T23:59:58Z"),
                    instant("2099-01-01T00:00:00Z"),
                ],
            ),
        };
        let result = prepare_release_for_test_with_clocks(release, before, selector, after);
        match case {
            Case::DelayedFresh => {
                assert!(matches!(
                    result.expect("delayed but fresh publication"),
                    PreparedReleaseOutcome::Prepared(_)
                ));
                assert!(root.join("versions/0.2.0").is_dir());
            }
            Case::CommitRollback | Case::CommitExpiry => {
                let error = result.expect_err("selector-boundary clock must fail");
                assert!(matches!(
                    error,
                    PreparedReleaseError::PreparedCommit(
                        crate::distribution::update_auth::PreparedVersionCommitError::Authentication(
                            ArtifactFetchAuthorizationError::Authentication(
                                TufVerifierError::ClockRollback | TufVerifierError::ExpiredMetadata
                            )
                        )
                    )
                ));
                assert!(!root.join("versions/0.2.0").exists());
                assert!(!is_prepared_durability_unknown(&error));
            }
            Case::PostcommitRollback
            | Case::PostcommitExpiry
            | Case::FinalizationRollback
            | Case::FinalizationExpiry => {
                let error = match result {
                    Err(error) => error,
                    Ok(outcome) => {
                        panic!("postcommit clock {case:?} must fail, got {outcome:?}")
                    }
                };
                assert!(is_prepared_durability_unknown(&error));
                assert!(root.join("versions/0.2.0").is_dir());
            }
        }
        assert!(!root.join("activations").exists());
        assert!(!root.join("current").exists());
    }
}

#[path = "prepared_publication_cases/hostile_cases.rs"]
mod hostile_cases;
