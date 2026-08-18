use super::adversarial::complete_fixture;
use super::*;
use crate::distribution::update_auth::test_repository::{
    stable_release_successor_pair, RetainedReleaseMutation,
};

#[test]
fn candidate_is_reauthenticated_under_lock_and_after_reopen() {
    let (_temp, authorization) = authorization();
    let anchor = EmbeddedTrustRoot::from_compiled(ROOT);
    let first = complete_static_transcript(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T08:30:00.123456789Z"),
                instant("2026-08-18T08:30:00.223456789Z"),
            ],
        )
        .expect("initial verifier starts"),
    );
    let (outcome, durable) =
        commit_and_reopen(&authorization, &anchor, first).expect("first commit is durable");
    assert_eq!(outcome, MetadataCommitOutcome::Committed { sequence: 1 });
    assert_eq!(durable.sequence(), 1);
    assert_ne!(durable.generation_sha256(), [0; 32]);

    let selected = read_selected(&authorization)
        .expect("selected journal is structurally readable")
        .expect("first generation is selected");
    let second = complete_static_transcript(
        begin_from_selected_for_test(
            &authorization,
            &anchor,
            selected,
            [
                instant("2026-08-18T08:31:00.123456789Z"),
                instant("2026-08-18T08:31:00.223456789Z"),
            ],
        )
        .expect("selected bytes replay from the compiled anchor"),
    );
    let (outcome, durable) =
        commit_and_reopen(&authorization, &anchor, second).expect("successor commit is durable");
    assert_eq!(outcome, MetadataCommitOutcome::Committed { sequence: 2 });
    assert_eq!(durable.sequence(), 2);
    assert_ne!(durable.generation_sha256(), [0; 32]);
}

#[test]
fn retained_release_pairs_are_append_only_across_lock_held_signed_successors() {
    for mutation in [
        RetainedReleaseMutation::RebindManifestDigest,
        RetainedReleaseMutation::RebindArchiveLength,
        RetainedReleaseMutation::RemoveManifest,
        RetainedReleaseMutation::RemovePair,
    ] {
        let (initial, successor) = stable_release_successor_pair(mutation);
        let (_temp, authorization) = authorization();
        let anchor = EmbeddedTrustRoot::from_compiled(Box::leak(
            initial.repository.anchor.clone().into_boxed_slice(),
        ));
        let first = complete_fixture(
            begin_from_anchor_for_test(
                &authorization,
                &anchor,
                [
                    instant("2026-08-18T10:00:00Z"),
                    instant("2026-08-18T10:00:01Z"),
                ],
            )
            .expect("initial stable inventory authenticates"),
            &initial.repository,
            0,
        );
        commit_at_recorded_completion(&authorization, &anchor, first)
            .expect("initial stable inventory commits");
        let selected = read_selected(&authorization)
            .expect("selected stable floor reads")
            .expect("selected stable floor exists");
        let next = complete_fixture(
            begin_from_selected_for_test(
                &authorization,
                &anchor,
                selected,
                [
                    instant("2026-08-18T10:01:00Z"),
                    instant("2026-08-18T10:01:01Z"),
                ],
            )
            .expect("signed successor authenticates at TUF layer"),
            &successor.repository,
            0,
        );
        assert!(matches!(
            commit_at_recorded_completion(&authorization, &anchor, next),
            Err(TufVerifierError::RetainedReleaseMutation)
        ));
        assert_eq!(
            read_selected(&authorization)
                .expect("rejected successor leaves stable floor readable")
                .expect("stable floor remains selected")
                .sequence(),
            1,
            "mutation {mutation:?}"
        );
    }

    let (initial, successor) = stable_release_successor_pair(RetainedReleaseMutation::AppendOnly);
    let (_temp, authorization) = authorization();
    let anchor = EmbeddedTrustRoot::from_compiled(Box::leak(
        initial.repository.anchor.clone().into_boxed_slice(),
    ));
    let first = complete_fixture(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T10:02:00Z"),
                instant("2026-08-18T10:02:01Z"),
            ],
        )
        .expect("initial stable inventory authenticates"),
        &initial.repository,
        0,
    );
    commit_at_recorded_completion(&authorization, &anchor, first)
        .expect("initial stable inventory commits");
    let selected = read_selected(&authorization)
        .expect("selected stable floor reads")
        .expect("selected stable floor exists");
    let next = complete_fixture(
        begin_from_selected_for_test(
            &authorization,
            &anchor,
            selected,
            [
                instant("2026-08-18T10:03:00Z"),
                instant("2026-08-18T10:03:01Z"),
            ],
        )
        .expect("append-only successor replays from stable floor"),
        &successor.repository,
        0,
    );
    let (outcome, _) = commit_at_recorded_completion(&authorization, &anchor, next)
        .expect("byte-identical retained pair plus new pair commits");
    assert_eq!(outcome, MetadataCommitOutcome::Committed { sequence: 2 });
}

#[test]
fn exact_selected_retry_repairs_and_returns_already_committed() {
    let (_temp, authorization) = authorization();
    let anchor = EmbeddedTrustRoot::from_compiled(ROOT);
    let samples = [
        instant("2026-08-18T08:40:00.123456789Z"),
        instant("2026-08-18T08:40:00.223456789Z"),
    ];
    let first = complete_static_transcript(
        begin_from_anchor_for_test(&authorization, &anchor, samples)
            .expect("first exact candidate authenticates"),
    );
    commit_and_reopen(&authorization, &anchor, first).expect("first candidate commits");

    let exact_retry = complete_static_transcript(
        begin_from_anchor_for_test(&authorization, &anchor, samples)
            .expect("same exact candidate can be reconstructed"),
    );
    let (outcome, durable) = commit_and_reopen(&authorization, &anchor, exact_retry)
        .expect("selected exact retry repairs and reopens");
    assert_eq!(
        outcome,
        MetadataCommitOutcome::AlreadyCommitted { sequence: 1 }
    );
    assert_eq!(durable.sequence(), 1);
}

#[test]
fn advancing_commit_rechecks_freshness_under_the_lock_and_at_precommit() {
    let (_temp, authorization) = authorization();
    let anchor = EmbeddedTrustRoot::from_compiled(ROOT);
    let candidate = complete_static_transcript(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T08:45:00Z"),
                instant("2026-08-18T08:45:01Z"),
            ],
        )
        .expect("candidate authenticates before expiry"),
    );
    assert!(matches!(
        commit_and_reopen_for_test(
            &authorization,
            &anchor,
            candidate,
            [instant("2999-01-01T00:00:00Z")],
        ),
        Err(TufVerifierError::ExpiredMetadata)
    ));
    assert!(read_selected(&authorization)
        .expect("expired pre-stage attempt leaves a readable empty journal")
        .is_none());

    let candidate = complete_static_transcript(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T08:46:00Z"),
                instant("2026-08-18T08:46:01Z"),
            ],
        )
        .expect("second candidate authenticates"),
    );
    assert!(matches!(
        commit_and_reopen_for_test(
            &authorization,
            &anchor,
            candidate,
            [
                instant("2998-12-31T23:59:59.999999999Z"),
                instant("2999-01-01T00:00:00Z"),
            ],
        ),
        Err(TufVerifierError::Journal(
            MetadataJournalError::PrecommitRejected
        ))
    ));
    assert!(read_selected(&authorization).is_err());
    let recovered = recover_after_process_restart(&authorization, &anchor)
        .expect("restart discards the never-selected expired transaction");
    assert_eq!(
        recovered.cleanup(),
        MetadataRestartCleanup::DiscardedUnselected { sequence: 1 }
    );
    assert!(recovered.selected().is_none());
    assert!(read_selected(&authorization)
        .expect("discard restores the authoritative reader")
        .is_none());
}

#[test]
fn advancing_commit_rejects_a_backward_final_clock_sample() {
    let (_temp, authorization) = authorization();
    let anchor = EmbeddedTrustRoot::from_compiled(ROOT);
    let candidate = complete_static_transcript(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T08:46:00Z"),
                instant("2026-08-18T08:46:01Z"),
            ],
        )
        .expect("candidate authenticates"),
    );
    assert!(matches!(
        commit_and_reopen_for_test(
            &authorization,
            &anchor,
            candidate,
            [
                instant("2026-08-18T08:46:10Z"),
                instant("2026-08-18T08:46:09Z"),
            ],
        ),
        Err(TufVerifierError::Journal(
            MetadataJournalError::PrecommitRejected
        ))
    ));
    assert!(read_selected(&authorization).is_err());
    let recovered = recover_after_process_restart(&authorization, &anchor)
        .expect("restart discards the never-selected rollback-clock transaction");
    assert_eq!(
        recovered.cleanup(),
        MetadataRestartCleanup::DiscardedUnselected { sequence: 1 }
    );
    assert!(recovered.selected().is_none());
}

#[test]
fn exact_selected_retry_repairs_as_a_floor_even_after_expiry() {
    let (_temp, authorization) = authorization();
    let anchor = EmbeddedTrustRoot::from_compiled(ROOT);
    let samples = [
        instant("2026-08-18T08:47:00Z"),
        instant("2026-08-18T08:47:01Z"),
    ];
    let candidate = complete_static_transcript(
        begin_from_anchor_for_test(&authorization, &anchor, samples)
            .expect("candidate authenticates"),
    );
    commit_and_reopen_for_test(
        &authorization,
        &anchor,
        candidate,
        [
            instant("2026-08-18T08:47:02Z"),
            instant("2026-08-18T08:47:03Z"),
        ],
    )
    .expect("candidate commits while fresh");

    let retry = complete_static_transcript(
        begin_from_anchor_for_test(&authorization, &anchor, samples)
            .expect("exact receipt candidate reconstructs"),
    );
    let (outcome, floor) =
        commit_and_reopen_for_test(&authorization, &anchor, retry, std::iter::empty())
            .expect("selected historical repair does not require current freshness");
    assert_eq!(
        outcome,
        MetadataCommitOutcome::AlreadyCommitted { sequence: 1 }
    );
    assert_eq!(floor.sequence(), 1);
}

#[test]
fn fresh_process_discards_every_unselected_signed_transaction_phase() {
    for barrier in [
        Barrier::PendingDirectory,
        Barrier::GenerationFiles,
        Barrier::GenerationPublish,
        Barrier::GenerationsSync,
        Barrier::SelectorFile,
        Barrier::MetadataPrecommitSync,
    ] {
        let (_temp, authorization) = authorization();
        let root = std::path::PathBuf::from(authorization.state_root());
        let anchor = EmbeddedTrustRoot::from_compiled(ROOT);
        let candidate = complete_static_transcript(
            begin_from_anchor_for_test(
                &authorization,
                &anchor,
                [
                    instant("2026-08-18T08:48:00Z"),
                    instant("2026-08-18T08:48:01Z"),
                ],
            )
            .expect("signed candidate authenticates"),
        );
        assert!(commit_candidate_for_test(
            authorization,
            candidate,
            FaultPlan {
                barrier: Some(barrier),
            },
        )
        .is_err());

        let authorization = MetadataStateAuthorization::for_test_path(&root, INSTALLATION_ID);
        assert!(read_selected(&authorization).is_err());
        let recovered = recover_after_process_restart(&authorization, &anchor)
            .expect("restart discards unselected state without its candidate");
        assert_eq!(
            recovered.cleanup(),
            MetadataRestartCleanup::DiscardedUnselected { sequence: 1 },
            "barrier {}",
            barrier.name(),
        );
        assert!(recovered.selected().is_none());
        assert!(read_selected(&authorization)
            .expect("ordinary reader is restored after discard")
            .is_none());
    }
}

#[test]
fn restart_repairs_a_selector_commit_without_requiring_the_candidate() {
    let (_temp, authorization) = authorization();
    let root = std::path::PathBuf::from(authorization.state_root());
    let anchor = EmbeddedTrustRoot::from_compiled(ROOT);
    let candidate = complete_static_transcript(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T08:49:00Z"),
                instant("2026-08-18T08:49:01Z"),
            ],
        )
        .expect("signed candidate authenticates"),
    );
    assert!(matches!(
        commit_candidate_for_test(
            authorization,
            candidate,
            FaultPlan {
                barrier: Some(Barrier::SelectorCommit),
            },
        ),
        Err(MetadataJournalError::CommittedDurabilityUnknown { sequence: 1, .. })
    ));

    let authorization = MetadataStateAuthorization::for_test_path(&root, INSTALLATION_ID);
    let recovered = recover_after_process_restart(&authorization, &anchor)
        .expect("selected bytes authenticate and durability is repaired");
    assert_eq!(recovered.cleanup(), MetadataRestartCleanup::Clean);
    assert_eq!(recovered.selected().map(|proof| proof.sequence()), Some(1));
}

#[test]
fn restart_discards_an_unselected_successor_and_preserves_the_authenticated_floor() {
    let (_temp, authorization) = authorization();
    let root = std::path::PathBuf::from(authorization.state_root());
    let anchor = EmbeddedTrustRoot::from_compiled(ROOT);
    let first = complete_static_transcript(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T08:50:00Z"),
                instant("2026-08-18T08:50:01Z"),
            ],
        )
        .expect("selected predecessor authenticates"),
    );
    let (_, first_floor) = commit_at_recorded_completion(&authorization, &anchor, first)
        .expect("selected predecessor commits");

    let selected = read_selected(&authorization)
        .expect("selected predecessor reads")
        .expect("selected predecessor exists");
    let successor = complete_static_transcript(
        begin_from_selected_for_test(
            &authorization,
            &anchor,
            selected,
            [
                instant("2026-08-18T08:51:00Z"),
                instant("2026-08-18T08:51:01Z"),
            ],
        )
        .expect("successor authenticates from the durable floor"),
    );
    assert!(commit_candidate_for_test(
        authorization,
        successor,
        FaultPlan {
            barrier: Some(Barrier::MetadataPrecommitSync),
        },
    )
    .is_err());

    let authorization = MetadataStateAuthorization::for_test_path(&root, INSTALLATION_ID);
    assert!(read_selected(&authorization).is_err());
    let recovered = recover_after_process_restart(&authorization, &anchor)
        .expect("production restart path authenticates the selected floor and discards N+1");
    assert_eq!(
        recovered.cleanup(),
        MetadataRestartCleanup::DiscardedUnselected { sequence: 2 }
    );
    let recovered_floor = recovered.selected().expect("selected floor survives");
    assert_eq!(recovered_floor.sequence(), 1);
    assert_eq!(
        recovered_floor.generation_sha256(),
        first_floor.generation_sha256()
    );

    let selected = read_selected(&authorization)
        .expect("ordinary reader reopens the selected floor")
        .expect("selected floor remains");
    let fresh_successor = complete_static_transcript(
        begin_from_selected_for_test(
            &authorization,
            &anchor,
            selected,
            [
                instant("2026-08-18T08:52:00Z"),
                instant("2026-08-18T08:52:01Z"),
            ],
        )
        .expect("fresh successor reauthenticates"),
    );
    let (outcome, durable) =
        commit_at_recorded_completion(&authorization, &anchor, fresh_successor)
            .expect("fresh successor reuses sequence two after discard");
    assert_eq!(outcome, MetadataCommitOutcome::Committed { sequence: 2 });
    assert_eq!(durable.sequence(), 2);
}
