use crate::distribution::update_auth::tests::TIMESTAMP;

#[test]
fn durable_clock_and_exact_byte_floors_reject_rollback_and_equivocation() {
    let (_temp, authorization) = authorization();
    let anchor = EmbeddedTrustRoot::from_compiled(ROOT);
    let first = complete_static_transcript(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T10:00:00Z"),
                instant("2026-08-18T10:00:01Z"),
            ],
        )
        .expect("static anchor starts"),
    );
    commit_at_recorded_completion(&authorization, &anchor, first).expect("baseline commits");

    let selected = read_selected(&authorization)
        .expect("baseline reads")
        .expect("baseline selected");
    assert!(matches!(
        begin_from_selected_for_test(
            &authorization,
            &anchor,
            selected,
            [
                instant("2026-08-18T09:59:59Z"),
                instant("2026-08-18T10:01:00Z")
            ]
        ),
        Err(TufVerifierError::ClockRollback)
    ));

    let selected = read_selected(&authorization)
        .expect("baseline rereads")
        .expect("baseline remains selected");
    let root = request(
        begin_from_selected_for_test(
            &authorization,
            &anchor,
            selected,
            [
                instant("2026-08-18T10:01:00Z"),
                instant("2026-08-18T10:01:01Z"),
            ],
        )
        .expect("baseline replay succeeds"),
        "2.root.json",
    );
    let timestamp = request(
        root.respond(MetadataResponse::ConfirmedNotFound)
            .expect("root chain terminates"),
        "timestamp.json",
    );
    let (equivocating_timestamp, _, _) = static_lower_roles(2, "2998-01-01T00:00:00Z");
    assert!(matches!(
        timestamp.respond(MetadataResponse::Found(
            equivocating_timestamp.into_boxed_slice()
        )),
        Err(TufVerifierError::RollbackOrEquivocation)
    ));

    let selected = read_selected(&authorization)
        .expect("baseline reads after equivocation")
        .expect("baseline remains selected after equivocation");
    let root = request(
        begin_from_selected_for_test(
            &authorization,
            &anchor,
            selected,
            [
                instant("2026-08-18T10:02:00Z"),
                instant("2026-08-18T10:02:01Z"),
            ],
        )
        .expect("baseline still replays"),
        "2.root.json",
    );
    let timestamp = request(
        root.respond(MetadataResponse::ConfirmedNotFound)
            .expect("root chain terminates"),
        "timestamp.json",
    );
    let semantically_equal: serde_json::Value =
        serde_json::from_slice(TIMESTAMP).expect("retained timestamp JSON");
    let reformatted =
        serde_json::to_vec_pretty(&semantically_equal).expect("reformatted timestamp JSON");
    assert_ne!(reformatted, TIMESTAMP);
    assert!(matches!(
        timestamp.respond(MetadataResponse::Found(reformatted.into_boxed_slice())),
        Err(TufVerifierError::RollbackOrEquivocation)
    ));
}

#[test]
fn expired_committed_metadata_remains_a_floor_across_restart() {
    let fixture = same_key_chain_with_expiry(0, false, "2026-08-18T12:00:00Z");
    let (_temp, authorization) = authorization();
    let anchor = leaked_anchor(&fixture.anchor);
    let baseline = complete_fixture(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T11:00:00Z"),
                instant("2026-08-18T11:00:01Z"),
            ],
        )
        .expect("baseline starts while metadata is fresh"),
        &fixture,
        0,
    );
    commit_at_recorded_completion(&authorization, &anchor, baseline)
        .expect("fresh baseline commits");

    let root = request(
        begin_from_selected_for_test(
            &authorization,
            &anchor,
            read_selected(&authorization)
                .expect("expired baseline reads structurally")
                .expect("expired baseline remains selected"),
            [
                instant("2026-08-18T12:01:00Z"),
                instant("2026-08-18T12:01:01Z"),
            ],
        )
        .expect("historical replay retains the expired role as a floor"),
        "2.root.json",
    );
    let timestamp = request(
        root.respond(MetadataResponse::ConfirmedNotFound)
            .expect("root remains fresh"),
        "timestamp.json",
    );
    let (rollback, _, _) = same_key_lower_roles(1, "2999-01-01T00:00:00Z");
    assert!(matches!(
        timestamp.respond(MetadataResponse::Found(rollback.into_boxed_slice())),
        Err(TufVerifierError::RollbackOrEquivocation)
    ));
}

#[test]
fn every_lower_role_floor_rejects_rollback_and_same_version_new_bytes() {
    for label in ["timestamp", "snapshot", "targets"] {
        let baseline = format!("{label}-baseline");
        let changed = format!("{label}-changed");
        let floor = RoleFloor::new(7, baseline.as_bytes());
        floor
            .require(7, baseline.as_bytes())
            .expect("exact equal bytes preserve the floor");
        assert!(matches!(
            floor.require(6, baseline.as_bytes()),
            Err(TufVerifierError::RollbackOrEquivocation)
        ));
        assert!(matches!(
            floor.require(7, changed.as_bytes()),
            Err(TufVerifierError::RollbackOrEquivocation)
        ));
    }
}

#[test]
fn stale_candidate_is_replayed_against_the_live_locked_floor_before_mutation() {
    let (_temp, authorization) = authorization();
    let anchor = EmbeddedTrustRoot::from_compiled(ROOT);
    let baseline = complete_static_transcript(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T10:10:00Z"),
                instant("2026-08-18T10:10:01Z"),
            ],
        )
        .expect("static anchor starts"),
    );
    commit_at_recorded_completion(&authorization, &anchor, baseline).expect("baseline commits");

    let stale = complete_static_transcript(
        begin_from_selected_for_test(
            &authorization,
            &anchor,
            read_selected(&authorization)
                .expect("baseline reads")
                .expect("baseline selected"),
            [
                instant("2026-08-18T10:11:00Z"),
                instant("2026-08-18T10:11:01Z"),
            ],
        )
        .expect("stale attempt starts from generation one"),
    );
    let winner = complete_static_transcript(
        begin_from_selected_for_test(
            &authorization,
            &anchor,
            read_selected(&authorization)
                .expect("baseline rereads")
                .expect("baseline remains selected"),
            [
                instant("2026-08-18T10:12:00Z"),
                instant("2026-08-18T10:12:01Z"),
            ],
        )
        .expect("winning attempt starts from generation one"),
    );
    commit_at_recorded_completion(&authorization, &anchor, winner)
        .expect("winner advances the live floor");

    assert!(matches!(
        commit_at_recorded_completion(&authorization, &anchor, stale),
        Err(TufVerifierError::ClockRollback)
    ));
    assert_eq!(
        read_selected(&authorization)
            .expect("journal remains readable")
            .expect("winner remains selected")
            .sequence(),
        2
    );
}

#[test]
fn structurally_valid_selected_bytes_cannot_cross_installation_state_roots() {
    let (_first_temp, first_authorization) = authorization();
    let anchor = EmbeddedTrustRoot::from_compiled(ROOT);
    let candidate = complete_static_transcript(
        begin_from_anchor_for_test(
            &first_authorization,
            &anchor,
            [
                instant("2026-08-18T10:20:00Z"),
                instant("2026-08-18T10:20:01Z"),
            ],
        )
        .expect("first root starts"),
    );
    commit_at_recorded_completion(&first_authorization, &anchor, candidate)
        .expect("first installation commits");
    let copied_structural_bytes = read_selected(&first_authorization)
        .expect("first journal reads")
        .expect("first generation selected");

    let (_second_temp, second_authorization) = super::authorization();
    assert!(matches!(
        begin_from_selected_for_test(
            &second_authorization,
            &anchor,
            copied_structural_bytes,
            [
                instant("2026-08-18T10:21:00Z"),
                instant("2026-08-18T10:21:01Z")
            ]
        ),
        Err(TufVerifierError::Journal(_))
    ));
}

#[test]
fn completion_time_rechecks_every_role_and_rejects_expiry_equality() {
    let fixture = same_key_chain_with_expiry(0, false, "2026-08-18T10:30:00.5Z");
    let (_temp, authorization) = authorization();
    let anchor = leaked_anchor(&fixture.anchor);
    let step = begin_from_anchor_for_test(
        &authorization,
        &anchor,
        [
            instant("2026-08-18T10:30:00.400000000Z"),
            instant("2026-08-18T10:30:00.500000000Z"),
        ],
    )
    .expect("verification starts before expiry");
    let root = request(step, "2.root.json");
    let timestamp = request(
        root.respond(MetadataResponse::ConfirmedNotFound)
            .expect("root remains fresh"),
        "timestamp.json",
    );
    let snapshot = request(
        timestamp
            .respond(MetadataResponse::Found(
                fixture.timestamp.into_boxed_slice(),
            ))
            .expect("timestamp is fresh at start"),
        "snapshot.json",
    );
    let targets = request(
        snapshot
            .respond(MetadataResponse::Found(fixture.snapshot.into_boxed_slice()))
            .expect("snapshot is fresh at start"),
        "targets.json",
    );
    assert!(matches!(
        targets.respond(MetadataResponse::Found(fixture.targets.into_boxed_slice())),
        Err(TufVerifierError::ExpiredMetadata)
    ));
}

#[test]
fn completion_clock_reversal_rejects_an_otherwise_valid_transcript() {
    let (_temp, authorization) = authorization();
    let anchor = EmbeddedTrustRoot::from_compiled(ROOT);
    let root = request(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T10:40:01Z"),
                instant("2026-08-18T10:40:00Z"),
            ],
        )
        .expect("attempt starts"),
        "2.root.json",
    );
    let timestamp = request(
        root.respond(MetadataResponse::ConfirmedNotFound)
            .expect("root chain terminates"),
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
            .respond(MetadataResponse::Found(
                crate::distribution::update_auth::tests::SNAPSHOT.into(),
            ))
            .expect("snapshot authenticates"),
        "targets.json",
    );
    assert!(matches!(
        targets.respond(MetadataResponse::Found(
            crate::distribution::update_auth::tests::TARGETS.into()
        )),
        Err(TufVerifierError::ClockRollback)
    ));
}
use super::*;
