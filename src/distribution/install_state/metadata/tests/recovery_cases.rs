use super::*;
use std::os::unix::fs::PermissionsExt;

fn stage_unselected_successor(root: &Path, root_version: u64) {
    let failed = commit_candidate_for_test(
        authorization(root),
        candidate_at(
            root,
            "2026-08-17T20:00:00Z",
            "2026-08-17T20:00:01Z",
            root_version,
            3,
        ),
        FaultPlan {
            barrier: Some(Barrier::MetadataPrecommitSync),
        },
    );
    assert!(failed.is_err(), "staging barrier must leave residue");
    assert!(read_selected(&authorization(root)).is_err());
}

fn commit_predecessor(root: &Path) -> Vec<u8> {
    commit_candidate_for_test(
        authorization(root),
        candidate_at(root, "2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3),
        FaultPlan::default(),
    )
    .expect("commit selected predecessor");
    std::fs::read(root.join("update/metadata/current.json")).expect("selected selector bytes")
}

fn stage_second_generation(root: &Path) {
    assert!(commit_candidate_for_test(
        authorization(root),
        candidate_at(root, "2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 3, 4,),
        FaultPlan {
            barrier: Some(Barrier::MetadataPrecommitSync),
        },
    )
    .is_err());
    assert!(read_selected(&authorization(root)).is_err());
}

#[test]
fn every_successor_discard_barrier_is_exactly_retryable() {
    const ROOT_VERSION: u64 = 2;
    for barrier in successor_discard_barriers(ROOT_VERSION as usize - 1) {
        let parent = tempfile::tempdir().expect("tempdir");
        let root = test_root(&parent);
        stage_unselected_successor(&root, ROOT_VERSION);

        assert!(discard_unselected_for_test(
            authorization(&root),
            FaultPlan {
                barrier: Some(barrier),
            },
        )
        .is_err());
        let repaired = discard_unselected_for_test(authorization(&root), FaultPlan::default())
            .unwrap_or_else(|error| panic!("repair after {barrier:?}: {error}"));
        assert!(matches!(
            repaired,
            super::super::MetadataRestartCleanup::Clean
                | super::super::MetadataRestartCleanup::DiscardedUnselected { sequence: 1 }
        ));
        assert!(read_selected(&authorization(&root))
            .expect("ordinary reader recovers")
            .is_none());
        assert_eq!(
            std::fs::read_dir(root.join("update/metadata/generations"))
                .expect("generation inventory")
                .count(),
            0
        );
    }
}

#[test]
fn multi_root_discard_exercises_every_write_prefix_position() {
    const ROOT_VERSION: u64 = 10;
    for barrier in successor_discard_barriers(ROOT_VERSION as usize - 1) {
        let parent = tempfile::tempdir().expect("tempdir");
        let root = test_root(&parent);
        stage_unselected_successor(&root, ROOT_VERSION);
        let _ = discard_unselected_for_test(
            authorization(&root),
            FaultPlan {
                barrier: Some(barrier),
            },
        );
        discard_unselected_for_test(authorization(&root), FaultPlan::default())
            .unwrap_or_else(|error| panic!("multi-root repair after {barrier:?}: {error}"));
        assert!(read_selected(&authorization(&root))
            .expect("multi-root discard restores ordinary reader")
            .is_none());
    }
}

#[test]
fn every_successor_discard_barrier_preserves_the_selected_floor() {
    for barrier in successor_discard_barriers(2) {
        let parent = tempfile::tempdir().expect("tempdir");
        let root = test_root(&parent);
        let selected_selector = commit_predecessor(&root);
        let selected_receipt = std::fs::read(
            root.join("update/metadata/generations/00000000000000000001/generation.json"),
        )
        .expect("selected receipt bytes");
        stage_second_generation(&root);

        assert!(discard_unselected_for_test(
            authorization(&root),
            FaultPlan {
                barrier: Some(barrier),
            },
        )
        .is_err());
        let repaired = discard_unselected_for_test(authorization(&root), FaultPlan::default())
            .unwrap_or_else(|error| panic!("successor repair after {barrier:?}: {error}"));
        assert!(matches!(
            repaired,
            super::super::MetadataRestartCleanup::Clean
                | super::super::MetadataRestartCleanup::DiscardedUnselected { sequence: 2 }
        ));
        assert_eq!(
            std::fs::read(root.join("update/metadata/current.json"))
                .expect("selected selector remains"),
            selected_selector,
            "barrier {barrier:?}",
        );
        assert_eq!(
            std::fs::read(
                root.join("update/metadata/generations/00000000000000000001/generation.json"),
            )
            .expect("selected receipt remains"),
            selected_receipt,
            "barrier {barrier:?}",
        );
        assert_eq!(
            read_selected(&authorization(&root))
                .expect("selected floor reads after discard")
                .expect("predecessor remains selected")
                .sequence,
            1
        );

        assert_eq!(
            commit_candidate_for_test(
                authorization(&root),
                candidate_at(&root, "2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 3, 4,),
                FaultPlan::default(),
            )
            .unwrap_or_else(|error| panic!("fresh sequence-two retry after {barrier:?}: {error}")),
            MetadataCommitOutcome::Committed { sequence: 2 }
        );
    }
}

#[test]
fn truncated_final_role_write_prefix_is_discarded_without_authentication() {
    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    let pending = root.join("update/metadata/generations/.pending-00000000000000000001");
    let root_chain = pending.join("root-chain");
    std::fs::create_dir_all(&root_chain).expect("pending write prefix");
    for directory in [
        root.clone(),
        root.join("update"),
        root.join("update/metadata"),
        root.join("update/metadata/generations"),
        pending.clone(),
        root_chain,
    ] {
        std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700))
            .expect("private directory mode");
    }
    for (name, bytes) in [
        ("anchor-root.json", b"anchor".as_slice()),
        ("trusted-root.json", b"root".as_slice()),
        ("timestamp.json", b"timestamp".as_slice()),
        ("snapshot.json", b"snapshot".as_slice()),
        ("targets.json", b"partial-target".as_slice()),
    ] {
        let path = pending.join(name);
        std::fs::write(&path, bytes).expect("partial role bytes");
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .expect("private role mode");
    }

    assert_eq!(
        discard_unselected_for_test(authorization(&root), FaultPlan::default())
            .expect("derived partial write prefix can be discarded"),
        super::super::MetadataRestartCleanup::DiscardedUnselected { sequence: 1 }
    );
    assert!(!pending.exists());
    assert!(read_selected(&authorization(&root))
        .expect("ordinary reader restored")
        .is_none());
}

#[test]
fn non_prefix_pending_inventory_fails_closed_without_deletion() {
    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    let pending = root.join("update/metadata/generations/.pending-00000000000000000001");
    std::fs::create_dir_all(&pending).expect("hostile pending directory");
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).expect("root mode");
    for directory in [
        root.join("update"),
        root.join("update/metadata"),
        root.join("update/metadata/generations"),
        pending.clone(),
    ] {
        std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700))
            .expect("private directory mode");
    }
    std::fs::write(pending.join("targets.json"), b"hostile suffix").expect("out-of-order file");
    std::fs::set_permissions(
        pending.join("targets.json"),
        std::fs::Permissions::from_mode(0o600),
    )
    .expect("private file mode");

    assert!(discard_unselected_for_test(authorization(&root), FaultPlan::default()).is_err());
    assert_eq!(
        std::fs::read(pending.join("targets.json")).expect("evidence remains"),
        b"hostile suffix"
    );
}

#[test]
fn corrupt_published_successor_and_selector_are_preserved_fail_closed() {
    for relative in [
        "update/metadata/.current-00000000000000000001.json",
        "update/metadata/generations/00000000000000000001/targets.json",
    ] {
        let parent = tempfile::tempdir().expect("tempdir");
        let root = test_root(&parent);
        stage_unselected_successor(&root, 2);
        let path = root.join(relative);
        let original = std::fs::read(&path).expect("staged evidence");
        let mut corrupted = original.clone();
        corrupted[0] ^= 1;
        std::fs::write(&path, &corrupted).expect("corrupt staged evidence");
        assert!(discard_unselected_for_test(authorization(&root), FaultPlan::default()).is_err());
        assert_eq!(
            std::fs::read(&path).expect("corruption preserved"),
            corrupted
        );
        assert!(root
            .join("update/metadata/generations/00000000000000000001")
            .exists());
    }
}

#[test]
fn discard_reopens_the_named_namespace_before_deletion() {
    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    stage_unselected_successor(&root, 2);
    let metadata = root.join("update/metadata");
    let detached = root.join("update/metadata-detached");
    let result = discard_unselected_with_hook_for_test(authorization(&root), || {
        std::fs::rename(&metadata, &detached).expect("detach metadata namespace");
        std::fs::create_dir(&metadata).expect("replacement metadata");
        std::fs::set_permissions(&metadata, std::fs::Permissions::from_mode(0o700))
            .expect("replacement mode");
        let generations = metadata.join("generations");
        std::fs::create_dir(&generations).expect("replacement generations");
        std::fs::set_permissions(&generations, std::fs::Permissions::from_mode(0o700))
            .expect("replacement generations mode");
    });
    assert!(result.is_err());
    assert!(detached.join("generations/00000000000000000001").exists());
    assert_eq!(
        std::fs::read_dir(metadata.join("generations"))
            .expect("replacement remains empty")
            .count(),
        0
    );
}
