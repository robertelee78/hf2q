use super::*;

pub(in crate::distribution::update_auth) fn online_key_rotation_recovery(
) -> (RepositoryFixture, RepositoryFixture, RepositoryFixture) {
    let root = TestKey::seeded("recovery-root", 0x91);
    let old_timestamp = TestKey::seeded("recovery-timestamp-old", 0x92);
    let old_snapshot = TestKey::seeded("recovery-snapshot-old", 0x93);
    let new_timestamp = TestKey::seeded("recovery-timestamp-new", 0x94);
    let new_snapshot = TestKey::seeded("recovery-snapshot-new", 0x95);
    let targets = TestKey::seeded("recovery-targets", 0x96);

    let anchor = envelope(
        root_value_for_roles(
            1,
            &[&root],
            &[&old_timestamp],
            &[&old_snapshot],
            &[&targets],
            false,
        ),
        &[&root],
    );
    let bridge_root = envelope(
        root_value_for_roles(
            2,
            &[&root],
            &[&old_timestamp],
            &[&old_snapshot],
            &[&targets],
            false,
        ),
        &[&root],
    );
    let rotated_root = envelope(
        root_value_for_roles(
            3,
            &[&root],
            &[&new_timestamp],
            &[&new_snapshot],
            &[&targets],
            false,
        ),
        &[&root],
    );
    let (old_ts, old_snap, old_targets) = lower_roles_for_roles(
        100,
        100,
        10,
        EXPIRES,
        &[&old_timestamp],
        &[&old_snapshot],
        &[&targets],
    );
    let (new_ts, new_snap, new_targets) = lower_roles_for_roles(
        2,
        2,
        11,
        EXPIRES,
        &[&new_timestamp],
        &[&new_snapshot],
        &[&targets],
    );
    let (rollback_ts, rollback_snap, rollback_targets) = lower_roles_for_roles(
        1,
        1,
        9,
        EXPIRES,
        &[&new_timestamp],
        &[&new_snapshot],
        &[&targets],
    );

    (
        RepositoryFixture {
            anchor: anchor.clone(),
            roots: Vec::new(),
            timestamp: old_ts,
            snapshot: old_snap,
            targets: old_targets,
            consistent_snapshot: false,
            metadata_version: 100,
        },
        RepositoryFixture {
            anchor: anchor.clone(),
            roots: vec![bridge_root.clone(), rotated_root.clone()],
            timestamp: new_ts,
            snapshot: new_snap,
            targets: new_targets,
            consistent_snapshot: false,
            metadata_version: 2,
        },
        RepositoryFixture {
            anchor,
            roots: vec![bridge_root, rotated_root],
            timestamp: rollback_ts,
            snapshot: rollback_snap,
            targets: rollback_targets,
            consistent_snapshot: false,
            metadata_version: 1,
        },
    )
}

pub(in crate::distribution::update_auth) fn unrelated_root_rotation_with_lower_rollback(
) -> (RepositoryFixture, RepositoryFixture) {
    let old_root = TestKey::seeded("unrelated-root-old", 0xa1);
    let new_root = TestKey::seeded("unrelated-root-new", 0xa2);
    let timestamp = TestKey::seeded("unrelated-timestamp", 0xa3);
    let snapshot = TestKey::seeded("unrelated-snapshot", 0xa4);
    let targets = TestKey::seeded("unrelated-targets", 0xa5);
    let anchor = envelope(
        root_value_for_roles(
            1,
            &[&old_root],
            &[&timestamp],
            &[&snapshot],
            &[&targets],
            false,
        ),
        &[&old_root],
    );
    let rotated_root = envelope(
        root_value_for_roles(
            2,
            &[&new_root],
            &[&timestamp],
            &[&snapshot],
            &[&targets],
            false,
        ),
        &[&old_root, &new_root],
    );
    let (old_ts, old_snap, old_targets) = lower_roles_for_roles(
        100,
        100,
        10,
        EXPIRES,
        &[&timestamp],
        &[&snapshot],
        &[&targets],
    );
    let (rollback_ts, rollback_snap, rollback_targets) =
        lower_roles_for_roles(1, 1, 11, EXPIRES, &[&timestamp], &[&snapshot], &[&targets]);
    (
        RepositoryFixture {
            anchor: anchor.clone(),
            roots: Vec::new(),
            timestamp: old_ts,
            snapshot: old_snap,
            targets: old_targets,
            consistent_snapshot: false,
            metadata_version: 100,
        },
        RepositoryFixture {
            anchor,
            roots: vec![rotated_root],
            timestamp: rollback_ts,
            snapshot: rollback_snap,
            targets: rollback_targets,
            consistent_snapshot: false,
            metadata_version: 1,
        },
    )
}

pub(in crate::distribution::update_auth) fn targets_key_rotation_with_lower_rollback(
) -> (RepositoryFixture, RepositoryFixture) {
    let root = TestKey::seeded("targets-rotation-root", 0xb1);
    let timestamp = TestKey::seeded("targets-rotation-timestamp", 0xb2);
    let snapshot = TestKey::seeded("targets-rotation-snapshot", 0xb3);
    let old_targets = TestKey::seeded("targets-rotation-old", 0xb4);
    let new_targets = TestKey::seeded("targets-rotation-new", 0xb5);
    let anchor = envelope(
        root_value_for_roles(
            1,
            &[&root],
            &[&timestamp],
            &[&snapshot],
            &[&old_targets],
            false,
        ),
        &[&root],
    );
    let rotated_root = envelope(
        root_value_for_roles(
            2,
            &[&root],
            &[&timestamp],
            &[&snapshot],
            &[&new_targets],
            false,
        ),
        &[&root],
    );
    let (old_ts, old_snap, old_target_role) = lower_roles_for_roles(
        100,
        100,
        10,
        EXPIRES,
        &[&timestamp],
        &[&snapshot],
        &[&old_targets],
    );
    let (rollback_ts, rollback_snap, rollback_target_role) = lower_roles_for_roles(
        1,
        1,
        11,
        EXPIRES,
        &[&timestamp],
        &[&snapshot],
        &[&new_targets],
    );
    (
        RepositoryFixture {
            anchor: anchor.clone(),
            roots: Vec::new(),
            timestamp: old_ts,
            snapshot: old_snap,
            targets: old_target_role,
            consistent_snapshot: false,
            metadata_version: 100,
        },
        RepositoryFixture {
            anchor,
            roots: vec![rotated_root],
            timestamp: rollback_ts,
            snapshot: rollback_snap,
            targets: rollback_target_role,
            consistent_snapshot: false,
            metadata_version: 1,
        },
    )
}

pub(in crate::distribution::update_auth) fn transient_online_rotation_with_lower_rollback(
) -> (RepositoryFixture, RepositoryFixture) {
    let root = TestKey::seeded("transient-root", 0xd1);
    let timestamp_a = TestKey::seeded("transient-timestamp-a", 0xd2);
    let snapshot_a = TestKey::seeded("transient-snapshot-a", 0xd3);
    let timestamp_b = TestKey::seeded("transient-timestamp-b", 0xd4);
    let snapshot_b = TestKey::seeded("transient-snapshot-b", 0xd5);
    let targets = TestKey::seeded("transient-targets", 0xd6);
    let anchor = envelope(
        root_value_for_roles(
            1,
            &[&root],
            &[&timestamp_a],
            &[&snapshot_a],
            &[&targets],
            false,
        ),
        &[&root],
    );
    let bridge = envelope(
        root_value_for_roles(
            2,
            &[&root],
            &[&timestamp_b],
            &[&snapshot_b],
            &[&targets],
            false,
        ),
        &[&root],
    );
    let final_root = envelope(
        root_value_for_roles(
            3,
            &[&root],
            &[&timestamp_a],
            &[&snapshot_a],
            &[&targets],
            false,
        ),
        &[&root],
    );
    let (old_ts, old_snap, old_targets) = lower_roles_for_roles(
        100,
        100,
        10,
        EXPIRES,
        &[&timestamp_a],
        &[&snapshot_a],
        &[&targets],
    );
    let (rollback_ts, rollback_snap, rollback_targets) = lower_roles_for_roles(
        1,
        1,
        11,
        EXPIRES,
        &[&timestamp_a],
        &[&snapshot_a],
        &[&targets],
    );
    (
        RepositoryFixture {
            anchor: anchor.clone(),
            roots: Vec::new(),
            timestamp: old_ts,
            snapshot: old_snap,
            targets: old_targets,
            consistent_snapshot: false,
            metadata_version: 100,
        },
        RepositoryFixture {
            anchor,
            roots: vec![bridge, final_root],
            timestamp: rollback_ts,
            snapshot: rollback_snap,
            targets: rollback_targets,
            consistent_snapshot: false,
            metadata_version: 1,
        },
    )
}
