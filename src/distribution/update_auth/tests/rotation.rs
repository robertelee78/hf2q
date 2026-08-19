use serde_json::json;

use crate::distribution::install_state::metadata::MetadataCommitOutcome;

#[test]
fn root_rotation_enforces_both_old_and_new_two_of_two_thresholds() {
    let valid = threshold_rotation(RotationSignatures::Complete);
    let (_temp, authorization) = authorization();
    let anchor = leaked_anchor(&valid.anchor);
    let candidate = complete_fixture(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T09:00:00Z"),
                instant("2026-08-18T09:00:01Z"),
            ],
        )
        .expect("threshold anchor authenticates"),
        &valid,
        0,
    );
    assert_eq!(candidate.root_chain().len(), 1);

    for invalid in [
        threshold_rotation(RotationSignatures::MissingOld),
        threshold_rotation(RotationSignatures::MissingNew),
    ] {
        let (_temp, authorization) = super::authorization();
        let anchor = leaked_anchor(&invalid.anchor);
        let root = request(
            begin_from_anchor_for_test(
                &authorization,
                &anchor,
                [
                    instant("2026-08-18T09:00:00Z"),
                    instant("2026-08-18T09:00:01Z"),
                ],
            )
            .expect("threshold anchor authenticates"),
            "2.root.json",
        );
        assert!(matches!(
            root.respond(MetadataResponse::Found(
                invalid.roots[0].clone().into_boxed_slice()
            )),
            Err(TufVerifierError::AuthenticationFailed)
        ));
    }
}

#[test]
fn root_requests_reject_skips_zero_and_version_exhaustion() {
    let skipped = same_key_chain(2, false);
    let (_temp, authorization) = authorization();
    let anchor = leaked_anchor(&skipped.anchor);
    let root = request(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T09:05:00Z"),
                instant("2026-08-18T09:05:01Z"),
            ],
        )
        .expect("edge anchor starts"),
        "2.root.json",
    );
    assert!(matches!(
        root.respond(MetadataResponse::Found(
            skipped.roots[1].clone().into_boxed_slice()
        )),
        Err(TufVerifierError::UnexpectedResponse)
    ));

    let mut zero: serde_json::Value = serde_json::from_slice(&skipped.anchor).expect("root JSON");
    zero["signed"]["version"] = json!(0);
    assert!(matches!(
        profile::root(&serde_json::to_vec(&zero).expect("zero root JSON")),
        Err(TufVerifierError::MalformedMetadata)
    ));

    let exhausted = anchor_at_version(u64::MAX);
    let (_temp, authorization) = super::authorization();
    let anchor = leaked_anchor(&exhausted);
    assert!(matches!(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T09:05:00Z"),
                instant("2026-08-18T09:05:01Z")
            ]
        ),
        Err(TufVerifierError::RootVersionExhausted)
    ));
}

#[test]
fn multi_root_history_commits_and_restart_uses_the_actual_root_floor() {
    let fixture = multi_rotation();
    let (_temp, authorization) = super::authorization();
    let anchor = leaked_anchor(&fixture.anchor);
    let first = complete_fixture(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T09:10:00Z"),
                instant("2026-08-18T09:10:01Z"),
            ],
        )
        .expect("compiled anchor starts"),
        &fixture,
        0,
    );
    assert_eq!(first.root_chain().len(), 2);
    let (_, durable) = commit_at_recorded_completion(&authorization, &anchor, first)
        .expect("rotated history commits");
    assert_eq!(durable.sequence(), 1);

    let selected = read_selected(&authorization)
        .expect("journal reads")
        .expect("rotated generation is selected");
    let second = complete_fixture(
        begin_from_selected_for_test(
            &authorization,
            &anchor,
            selected,
            [
                instant("2026-08-18T09:11:00Z"),
                instant("2026-08-18T09:11:01Z"),
            ],
        )
        .expect("restart replays the full root history"),
        &fixture,
        fixture.roots.len(),
    );
    assert_eq!(second.root_chain().len(), 2);
    let (_, durable) = commit_at_recorded_completion(&authorization, &anchor, second)
        .expect("restart candidate commits from root three");
    assert_eq!(durable.sequence(), 2);
}

#[test]
fn selected_floor_accepts_a_new_threshold_rotation_and_rejects_its_rollback() {
    let (first_repository, second_repository, rollback_repository) =
        successive_threshold_rotations();
    let (_temp, authorization) = super::authorization();
    let anchor = leaked_anchor(&first_repository.anchor);
    let first = complete_fixture(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T09:12:00Z"),
                instant("2026-08-18T09:12:01Z"),
            ],
        )
        .expect("first threshold repository starts"),
        &first_repository,
        0,
    );
    assert_eq!(first.root_chain().len(), 1);
    let (outcome, _) = commit_at_recorded_completion(&authorization, &anchor, first)
        .expect("first threshold rotation commits");
    assert_eq!(outcome, MetadataCommitOutcome::Committed { sequence: 1 });

    let selected = read_selected(&authorization)
        .expect("first threshold floor reads")
        .expect("first threshold floor is selected");
    let second = complete_fixture(
        begin_from_selected_for_test(
            &authorization,
            &anchor,
            selected,
            [
                instant("2026-08-18T09:13:00Z"),
                instant("2026-08-18T09:13:01Z"),
            ],
        )
        .expect("selected threshold floor replays"),
        &second_repository,
        1,
    );
    assert_eq!(second.root_chain().len(), 2);
    let (outcome, durable) = commit_at_recorded_completion(&authorization, &anchor, second)
        .expect("new threshold rotation commits from the selected floor");
    assert_eq!(outcome, MetadataCommitOutcome::Committed { sequence: 2 });
    assert_eq!(durable.sequence(), 2);

    let selected = read_selected(&authorization)
        .expect("second threshold floor reads")
        .expect("second threshold floor is selected");
    let terminal = request(
        begin_from_selected_for_test(
            &authorization,
            &anchor,
            selected,
            [
                instant("2026-08-18T09:14:00Z"),
                instant("2026-08-18T09:14:01Z"),
            ],
        )
        .expect("second threshold floor replays"),
        "4.root.json",
    )
    .respond(MetadataResponse::ConfirmedNotFound)
    .expect("root chain terminates at the durable version-three root");
    let timestamp = request(terminal, "timestamp.json");
    assert!(matches!(
        timestamp.respond(MetadataResponse::Found(
            rollback_repository.timestamp.into_boxed_slice()
        )),
        Err(TufVerifierError::RollbackOrEquivocation)
    ));
}

#[test]
fn online_key_rotation_resets_only_timestamp_and_snapshot_floors() {
    let (baseline_repository, recovered_repository, rollback_repository) =
        online_key_rotation_recovery();
    let (_temp, authorization) = authorization();
    let anchor = leaked_anchor(&baseline_repository.anchor);
    let baseline = complete_fixture(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T09:15:00Z"),
                instant("2026-08-18T09:15:01Z"),
            ],
        )
        .expect("high online-role baseline starts"),
        &baseline_repository,
        0,
    );
    commit_at_recorded_completion(&authorization, &anchor, baseline)
        .expect("high online-role floors commit");

    let mut step = begin_from_selected_for_test(
        &authorization,
        &anchor,
        read_selected(&authorization)
            .expect("baseline reads")
            .expect("baseline is selected"),
        [
            instant("2026-08-18T09:16:00Z"),
            instant("2026-08-18T09:16:01Z"),
        ],
    )
    .expect("selected high floors replay");
    step = request(step, "2.root.json")
        .respond(MetadataResponse::Found(
            rollback_repository.roots[0].clone().into_boxed_slice(),
        ))
        .expect("unrelated bridge root authenticates");
    step = request(step, "3.root.json")
        .respond(MetadataResponse::Found(
            rollback_repository.roots[1].clone().into_boxed_slice(),
        ))
        .expect("offline-authorized online-key rotation authenticates");
    step = request(step, "4.root.json")
        .respond(MetadataResponse::ConfirmedNotFound)
        .expect("rotated root chain terminates");
    step = request(step, "timestamp.json")
        .respond(MetadataResponse::Found(
            rollback_repository.timestamp.clone().into_boxed_slice(),
        ))
        .expect("rotated timestamp key may establish a lower recovery floor");
    step = request(step, "snapshot.json")
        .respond(MetadataResponse::Found(
            rollback_repository.snapshot.clone().into_boxed_slice(),
        ))
        .expect("rotated snapshot key may establish a lower recovery floor");
    assert!(matches!(
        request(step, "targets.json").respond(MetadataResponse::Found(
            rollback_repository.targets.clone().into_boxed_slice()
        )),
        Err(TufVerifierError::RollbackOrEquivocation)
    ));

    let recovered = complete_fixture(
        begin_from_selected_for_test(
            &authorization,
            &anchor,
            read_selected(&authorization)
                .expect("baseline rereads")
                .expect("failed recovery did not change selection"),
            [
                instant("2026-08-18T09:17:00Z"),
                instant("2026-08-18T09:17:01Z"),
            ],
        )
        .expect("recovery transcript starts"),
        &recovered_repository,
        0,
    );
    assert_eq!(
        recovered
            .timestamp_snapshot_floor_reset_from_root()
            .expect("candidate carries sealed reset evidence")
            .version(),
        1,
        "the reset binds the selected root, not the intermediate bridge root"
    );
    let (outcome, durable) = commit_at_recorded_completion(&authorization, &anchor, recovered)
        .expect("root-authorized online-role recovery commits and reopens");
    assert_eq!(outcome, MetadataCommitOutcome::Committed { sequence: 2 });
    assert_eq!(durable.sequence(), 2);

    let step = begin_from_selected_for_test(
        &authorization,
        &anchor,
        read_selected(&authorization)
            .expect("recovered floor reads")
            .expect("recovered floor is selected"),
        [
            instant("2026-08-18T09:18:00Z"),
            instant("2026-08-18T09:18:01Z"),
        ],
    )
    .expect("recovered generation replays from the compiled anchor");
    let timestamp = request(
        request(step, "4.root.json")
            .respond(MetadataResponse::ConfirmedNotFound)
            .expect("no further root rotation"),
        "timestamp.json",
    );
    assert!(matches!(
        timestamp.respond(MetadataResponse::Found(
            rollback_repository.timestamp.into_boxed_slice()
        )),
        Err(TufVerifierError::RollbackOrEquivocation)
    ));
}

#[test]
fn unrelated_root_or_targets_rotation_does_not_reset_online_role_floors() {
    for (label, (baseline_repository, rollback_repository)) in [
        ("root", unrelated_root_rotation_with_lower_rollback()),
        ("targets", targets_key_rotation_with_lower_rollback()),
        ("transient", transient_online_rotation_with_lower_rollback()),
    ] {
        let (_temp, authorization) = authorization();
        let anchor = leaked_anchor(&baseline_repository.anchor);
        let baseline = complete_fixture(
            begin_from_anchor_for_test(
                &authorization,
                &anchor,
                [
                    instant("2026-08-18T09:19:00Z"),
                    instant("2026-08-18T09:19:01Z"),
                ],
            )
            .expect("unrelated-role baseline starts"),
            &baseline_repository,
            0,
        );
        commit_at_recorded_completion(&authorization, &anchor, baseline)
            .expect("unrelated-role baseline commits");

        let mut step = begin_from_selected_for_test(
            &authorization,
            &anchor,
            read_selected(&authorization)
                .expect("baseline reads")
                .expect("baseline is selected"),
            [
                instant("2026-08-18T09:19:10Z"),
                instant("2026-08-18T09:19:11Z"),
            ],
        )
        .expect("baseline replays");
        for (index, root) in rollback_repository.roots.iter().enumerate() {
            step = request(step, &format!("{}.root.json", index + 2))
                .respond(MetadataResponse::Found(root.clone().into_boxed_slice()))
                .unwrap_or_else(|_| panic!("{label} root transition authenticates"));
        }
        let timestamp = request(
            request(
                step,
                &format!("{}.root.json", rollback_repository.roots.len() + 2),
            )
            .respond(MetadataResponse::ConfirmedNotFound)
            .unwrap_or_else(|_| panic!("{label} rotation terminates")),
            "timestamp.json",
        );
        assert!(
            matches!(
                timestamp.respond(MetadataResponse::Found(
                    rollback_repository.timestamp.into_boxed_slice()
                )),
                Err(TufVerifierError::RollbackOrEquivocation)
            ),
            "{label} rotation must preserve online-role floors"
        );
    }
}

#[test]
fn online_role_recovery_requires_prior_quorum_invalidation() {
    for case in online_binding_change_cases() {
        let old = profile::root(&case.old).expect("old root parses");
        let new = if case.within_profile {
            profile::root(&case.new).expect("new root parses")
        } else {
            assert!(matches!(
                profile::root(&case.new),
                Err(TufVerifierError::MalformedMetadata)
            ));
            sigstore_tuf::metadata::Metadata::<sigstore_tuf::metadata::Root>::from_slice(&case.new)
                .expect("out-of-profile root remains structural defense-in-depth evidence")
        };
        assert_eq!(
            crate::distribution::update_auth::verifier::online_role_binding_invalidated(
                &old.signed,
                &new.signed,
            )
            .expect("role bindings are well formed"),
            case.changed,
            "{}",
            case.label
        );
    }
}

#[test]
fn consistent_snapshot_names_are_derived_only_from_authenticated_parents() {
    let fixture = same_key_chain(0, true);
    let (_temp, authorization) = authorization();
    let anchor = leaked_anchor(&fixture.anchor);
    let candidate = complete_fixture(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T09:20:00Z"),
                instant("2026-08-18T09:20:01Z"),
            ],
        )
        .expect("consistent-snapshot anchor starts"),
        &fixture,
        0,
    );
    assert_eq!(candidate.snapshot().request_name(), "2.snapshot.json");
    assert_eq!(candidate.targets().request_name(), "2.targets.json");
}

#[test]
fn root_lifetime_accepts_256_rotations_and_rejects_the_257th() {
    let fixture = same_key_chain(MAX_ROOT_ROTATIONS, false);
    let (_temp, authorization) = authorization();
    let anchor = leaked_anchor(&fixture.anchor);
    let candidate = complete_fixture(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T09:30:00Z"),
                instant("2026-08-18T09:30:01Z"),
            ],
        )
        .expect("lifetime anchor starts"),
        &fixture,
        0,
    );
    assert_eq!(candidate.root_chain().len(), MAX_ROOT_ROTATIONS);

    let over_limit = same_key_chain(MAX_ROOT_ROTATIONS + 1, false);
    let (_temp, authorization) = super::authorization();
    let anchor = leaked_anchor(&over_limit.anchor);
    let mut step = begin_from_anchor_for_test(
        &authorization,
        &anchor,
        [
            instant("2026-08-18T09:30:00Z"),
            instant("2026-08-18T09:30:01Z"),
        ],
    )
    .expect("lifetime anchor starts");
    for (index, root) in over_limit.roots.iter().take(MAX_ROOT_ROTATIONS).enumerate() {
        step = request(step, &format!("{}.root.json", index + 2))
            .respond(MetadataResponse::Found(root.clone().into_boxed_slice()))
            .expect("rotation within the lifetime bound authenticates");
    }
    assert!(matches!(
        request(step, "258.root.json").respond(MetadataResponse::Found(
            over_limit.roots[MAX_ROOT_ROTATIONS]
                .clone()
                .into_boxed_slice()
        )),
        Err(TufVerifierError::RootRotationLimit)
    ));
}
use super::*;
