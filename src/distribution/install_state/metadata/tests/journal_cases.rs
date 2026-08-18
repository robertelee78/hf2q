use super::*;

#[test]
fn initial_and_successor_commits_are_bounded_and_idempotent() {
    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    let first = candidate_at(&root, "2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3);
    assert_eq!(
        commit_candidate_for_test(authorization(&root), first, FaultPlan::default())
            .expect("commit first"),
        MetadataCommitOutcome::Committed { sequence: 1 }
    );
    let selected = read_selected(&authorization(&root))
        .expect("read first")
        .expect("selection");
    assert_eq!(selected.sequence, 1);
    assert_eq!(selected.root_chain.len(), 1);
    assert!(!selected.anchor_root.is_empty());
    assert!(!selected.trusted_root.is_empty());
    assert!(!selected.timestamp.is_empty());
    assert!(!selected.snapshot.is_empty());
    assert!(!selected.targets.is_empty());

    let second = candidate_at(&root, "2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 3, 4);
    assert_eq!(
        commit_candidate_for_test(authorization(&root), second, FaultPlan::default())
            .expect("commit second"),
        MetadataCommitOutcome::Committed { sequence: 2 }
    );
    let generations = root.join("update/metadata/generations");
    assert_eq!(
        std::fs::read_dir(&generations)
            .expect("generations")
            .map(|entry| entry.expect("entry").file_name())
            .collect::<Vec<_>>(),
        vec![std::ffi::OsString::from("00000000000000000002")]
    );
    let selected = read_selected(&authorization(&root))
        .expect("read second")
        .expect("selection");
    assert_eq!(selected.sequence, 2);
    assert_eq!(selected.root_chain.len(), 2);

    let retry = candidate_at(&root, "2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 3, 4);
    assert_eq!(
        commit_candidate_for_test(authorization(&root), retry, FaultPlan::default())
            .expect("idempotent retry"),
        MetadataCommitOutcome::AlreadyCommitted { sequence: 2 }
    );
}

#[test]
fn every_initial_transaction_barrier_is_exactly_retryable() {
    for barrier in initial_transaction_barriers() {
        let parent = tempfile::tempdir().expect("tempdir");
        let root = test_root(&parent);
        let failed = commit_candidate_for_test(
            authorization(&root),
            candidate_at(&root, "2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3),
            FaultPlan {
                barrier: Some(barrier),
            },
        );
        assert!(failed.is_err(), "barrier {barrier:?} must fail");
        let committed = matches!(
            failed,
            Err(MetadataJournalError::CommittedDurabilityUnknown { .. })
        );
        assert_eq!(
            committed,
            matches!(
                barrier,
                Barrier::SelectorCommit
                    | Barrier::SelectorFullSync
                    | Barrier::GenerationPostcommitSync
                    | Barrier::MetadataPostcommitSync
                    | Barrier::UpdatePostcommitSync
                    | Barrier::RootPostcommitSync
            ),
            "barrier {barrier:?} committed classification"
        );
        let retry = commit_candidate_for_test(
            authorization(&root),
            candidate_at(&root, "2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3),
            FaultPlan::default(),
        )
        .expect("exact retry");
        assert!(matches!(
            retry,
            MetadataCommitOutcome::Committed { sequence: 1 }
                | MetadataCommitOutcome::AlreadyCommitted { sequence: 1 }
        ));
        assert_eq!(
            read_selected(&authorization(&root))
                .expect("read repaired state")
                .expect("selected")
                .sequence,
            1
        );
    }
}

#[test]
fn every_successor_transaction_barrier_is_exactly_retryable() {
    for barrier in successor_transaction_barriers() {
        let parent = tempfile::tempdir().expect("tempdir");
        let root = test_root(&parent);
        commit_candidate_for_test(
            authorization(&root),
            candidate_at(&root, "2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3),
            FaultPlan::default(),
        )
        .expect("commit predecessor");
        let failed = commit_candidate_for_test(
            authorization(&root),
            candidate_at(&root, "2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 3, 4),
            FaultPlan {
                barrier: Some(barrier),
            },
        );
        assert!(failed.is_err(), "barrier {barrier:?} must fail");
        let committed = matches!(
            failed,
            Err(MetadataJournalError::CommittedDurabilityUnknown { sequence: 2, .. })
        );
        assert_eq!(
            committed,
            matches!(
                barrier,
                Barrier::SelectorCommit
                    | Barrier::SelectorFullSync
                    | Barrier::GenerationPostcommitSync
                    | Barrier::MetadataPostcommitSync
                    | Barrier::UpdatePostcommitSync
                    | Barrier::RootPostcommitSync
                    | Barrier::PredecessorPruneRename
                    | Barrier::PredecessorPruneEntryRemoval(_)
                    | Barrier::PredecessorPruneRemoval
                    | Barrier::PredecessorPruneFullSync
            ),
            "barrier {barrier:?} committed classification"
        );
        let retry = commit_candidate_for_test(
            authorization(&root),
            candidate_at(&root, "2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 3, 4),
            FaultPlan::default(),
        )
        .unwrap_or_else(|error| panic!("retry after {barrier:?} failed: {error}"));
        assert!(matches!(
            retry,
            MetadataCommitOutcome::Committed { sequence: 2 }
                | MetadataCommitOutcome::AlreadyCommitted { sequence: 2 }
        ));
        assert_eq!(
            read_selected(&authorization(&root))
                .expect("read repaired successor")
                .expect("selected successor")
                .sequence,
            2
        );
        assert_eq!(
            std::fs::read_dir(root.join("update/metadata/generations"))
                .expect("bounded generations")
                .count(),
            1
        );
    }
}

#[test]
fn selected_successor_cleanup_failures_repair_without_update_exhaustion() {
    let mut barriers = vec![Barrier::PredecessorPruneRename];
    barriers.extend(prune_entry_barriers(1));
    barriers.push(Barrier::PredecessorPruneRemoval);
    barriers.push(Barrier::PredecessorPruneFullSync);
    for barrier in barriers {
        let parent = tempfile::tempdir().expect("tempdir");
        let root = test_root(&parent);
        commit_candidate_for_test(
            authorization(&root),
            candidate_at(&root, "2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3),
            FaultPlan::default(),
        )
        .expect("first");
        let failed = commit_candidate_for_test(
            authorization(&root),
            candidate_at(&root, "2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 3, 4),
            FaultPlan {
                barrier: Some(barrier),
            },
        );
        assert!(matches!(
            failed,
            Err(MetadataJournalError::CommittedDurabilityUnknown { sequence: 2, .. })
        ));
        assert_eq!(
            read_selected(&authorization(&root))
                .expect("selected remains readable")
                .expect("selection")
                .sequence,
            2
        );
        let repair = commit_candidate_for_test(
            authorization(&root),
            candidate_at(&root, "2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 3, 4),
            FaultPlan::default(),
        )
        .expect("cleanup retry");
        assert_eq!(
            repair,
            MetadataCommitOutcome::AlreadyCommitted { sequence: 2 }
        );
        assert_eq!(
            std::fs::read_dir(root.join("update/metadata/generations"))
                .expect("generations")
                .count(),
            1
        );
    }
}

#[test]
fn every_multi_root_cleanup_prefix_is_exactly_retryable() {
    const ROOT_VERSION: u64 = 10;
    const ROOT_HISTORY_ENTRIES: usize = ROOT_VERSION as usize - 1;

    for barrier in prune_entry_barriers(ROOT_HISTORY_ENTRIES) {
        let parent = tempfile::tempdir().expect("tempdir");
        let root = test_root(&parent);
        commit_candidate_for_test(
            authorization(&root),
            candidate_at(
                &root,
                "2026-08-17T20:00:00Z",
                "2026-08-17T20:00:01Z",
                ROOT_VERSION,
                1,
            ),
            FaultPlan::default(),
        )
        .expect("commit multi-root predecessor");

        assert!(matches!(
            commit_candidate_for_test(
                authorization(&root),
                candidate_at(
                    &root,
                    "2026-08-17T20:00:01Z",
                    "2026-08-17T20:00:02Z",
                    ROOT_VERSION,
                    2,
                ),
                FaultPlan {
                    barrier: Some(barrier),
                },
            ),
            Err(MetadataJournalError::CommittedDurabilityUnknown { sequence: 2, .. })
        ));

        assert_eq!(
            commit_candidate_for_test(
                authorization(&root),
                candidate_at(
                    &root,
                    "2026-08-17T20:00:01Z",
                    "2026-08-17T20:00:02Z",
                    ROOT_VERSION,
                    2,
                ),
                FaultPlan::default(),
            )
            .unwrap_or_else(|error| panic!("retry after {barrier:?} failed: {error}")),
            MetadataCommitOutcome::AlreadyCommitted { sequence: 2 }
        );
        assert_eq!(
            std::fs::read_dir(root.join("update/metadata/generations"))
                .expect("bounded generations")
                .count(),
            1
        );
    }
}

#[test]
fn prune_barrier_count_is_exact_for_each_exercised_history_shape() {
    for (root_version, root_history_entries) in [(2_u64, 1_usize), (10, 9)] {
        let parent = tempfile::tempdir().expect("tempdir");
        let root = test_root(&parent);
        commit_candidate_for_test(
            authorization(&root),
            candidate_at(
                &root,
                "2026-08-17T20:00:00Z",
                "2026-08-17T20:00:01Z",
                root_version,
                1,
            ),
            FaultPlan::default(),
        )
        .expect("commit predecessor");

        let first_unreachable_step = prune_entry_barriers(root_history_entries).count() + 1;
        assert_eq!(
            commit_candidate_for_test(
                authorization(&root),
                candidate_at(
                    &root,
                    "2026-08-17T20:00:01Z",
                    "2026-08-17T20:00:02Z",
                    root_version,
                    2,
                ),
                FaultPlan {
                    barrier: Some(Barrier::PredecessorPruneEntryRemoval(
                        first_unreachable_step,
                    )),
                },
            )
            .expect("past-the-end barrier must not trip"),
            MetadataCommitOutcome::Committed { sequence: 2 }
        );
    }
}

#[test]
fn corrupted_surviving_multi_root_suffix_fails_closed() {
    const ROOT_VERSION: u64 = 10;

    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    commit_candidate_for_test(
        authorization(&root),
        candidate_at(
            &root,
            "2026-08-17T20:00:00Z",
            "2026-08-17T20:00:01Z",
            ROOT_VERSION,
            1,
        ),
        FaultPlan::default(),
    )
    .expect("commit multi-root predecessor");
    assert!(matches!(
        commit_candidate_for_test(
            authorization(&root),
            candidate_at(
                &root,
                "2026-08-17T20:00:01Z",
                "2026-08-17T20:00:02Z",
                ROOT_VERSION,
                2,
            ),
            FaultPlan {
                barrier: Some(Barrier::PredecessorPruneEntryRemoval(3)),
            },
        ),
        Err(MetadataJournalError::CommittedDurabilityUnknown { sequence: 2, .. })
    ));

    let survivor = root.join(
        "update/metadata/generations/.prune-00000000000000000001/\
         root-chain/00000000000000000005.root.json",
    );
    std::fs::write(&survivor, b"corrupt").expect("corrupt surviving root");
    assert!(matches!(
        commit_candidate_for_test(
            authorization(&root),
            candidate_at(
                &root,
                "2026-08-17T20:00:01Z",
                "2026-08-17T20:00:02Z",
                ROOT_VERSION,
                2,
            ),
            FaultPlan::default(),
        ),
        Err(MetadataJournalError::Invalid(
            "stored metadata bytes do not match their receipt descriptor"
        ))
    ));
    assert!(
        survivor.exists(),
        "hostile cleanup residue must be preserved for diagnosis"
    );
}

#[test]
fn ordinary_reader_rejects_unselected_published_successor() {
    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    let failed = commit_candidate_for_test(
        authorization(&root),
        candidate_at(&root, "2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3),
        FaultPlan {
            barrier: Some(Barrier::GenerationPublish),
        },
    );
    assert!(failed.is_err());
    assert!(matches!(
        read_selected(&authorization(&root)),
        Err(MetadataJournalError::Invalid(
            "metadata successor transaction requires lock-held recovery"
        ))
    ));
    commit_candidate_for_test(
        authorization(&root),
        candidate_at(&root, "2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3),
        FaultPlan::default(),
    )
    .expect("lock-held recovery");
}

#[test]
fn metadata_commit_shares_the_nonblocking_installation_lock() {
    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    let _held = LockedInstallation::acquire(&root).expect("hold lock");
    assert!(matches!(
        commit_candidate_for_test(
            authorization(&root),
            candidate_at(&root, "2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3,),
            FaultPlan::default(),
        ),
        Err(MetadataJournalError::InstallState(
            super::super::InstallStateError::Busy
        ))
    ));
}

#[test]
fn selected_journal_is_bound_to_the_authorized_installation_identity() {
    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    commit_candidate_for_test(
        authorization(&root),
        candidate_at(&root, "2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3),
        FaultPlan::default(),
    )
    .expect("commit");
    let wrong_identity = MetadataStateAuthorization::for_test(
        ExplicitRootAuthorization::new(&root).expect("root authorization"),
        "a70ee078-5f20-45f6-bf42-bfcd1a992382",
    );
    assert!(matches!(
        read_selected(&wrong_identity),
        Err(MetadataJournalError::Invalid(
            "metadata generation belongs to a different installation state root"
        ))
    ));
}

#[test]
fn new_candidate_finishes_prior_cleanup_before_staging_successor() {
    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    commit_candidate_for_test(
        authorization(&root),
        candidate_at(&root, "2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3),
        FaultPlan::default(),
    )
    .expect("first");
    assert!(matches!(
        commit_candidate_for_test(
            authorization(&root),
            candidate_at(&root, "2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 3, 4),
            FaultPlan {
                barrier: Some(Barrier::PredecessorPruneEntryRemoval(3)),
            },
        ),
        Err(MetadataJournalError::CommittedDurabilityUnknown { sequence: 2, .. })
    ));

    assert_eq!(
        commit_candidate_for_test(
            authorization(&root),
            candidate_at(&root, "2026-08-17T20:00:02Z", "2026-08-17T20:00:03Z", 4, 5),
            FaultPlan::default(),
        )
        .expect("third candidate cleans and commits"),
        MetadataCommitOutcome::Committed { sequence: 3 }
    );
    assert_eq!(
        read_selected(&authorization(&root))
            .expect("read third")
            .expect("selection")
            .sequence,
        3
    );
    assert_eq!(
        std::fs::read_dir(root.join("update/metadata/generations"))
            .expect("generations")
            .count(),
        1
    );
}

#[test]
fn selected_generation_is_reverified_before_predecessor_cleanup() {
    use std::os::unix::fs::PermissionsExt;

    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    commit_candidate_for_test(
        authorization(&root),
        candidate_at(&root, "2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3),
        FaultPlan::default(),
    )
    .expect("first");
    assert!(matches!(
        commit_candidate_for_test(
            authorization(&root),
            candidate_at(&root, "2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 3, 4),
            FaultPlan {
                barrier: Some(Barrier::PredecessorPruneRename),
            },
        ),
        Err(MetadataJournalError::CommittedDurabilityUnknown { sequence: 2, .. })
    ));

    let generations = root.join("update/metadata/generations");
    let cleanup_receipt = generations.join(".prune-00000000000000000001/generation.json");
    std::fs::set_permissions(
        generations.join("00000000000000000002/targets.json"),
        std::fs::Permissions::from_mode(0o644),
    )
    .expect("corrupt selected generation mode");
    assert!(commit_candidate_for_test(
        authorization(&root),
        candidate_at(&root, "2026-08-17T20:00:02Z", "2026-08-17T20:00:03Z", 4, 5),
        FaultPlan::default(),
    )
    .is_err());
    assert!(
        cleanup_receipt.is_file(),
        "predecessor cleanup must wait for selected-generation repair"
    );
}

#[test]
fn cleanup_reopens_the_named_namespace_before_deletion() {
    use std::os::unix::fs::PermissionsExt;

    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    commit_candidate_for_test(
        authorization(&root),
        candidate_at(&root, "2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3),
        FaultPlan::default(),
    )
    .expect("first");
    assert!(matches!(
        commit_candidate_for_test(
            authorization(&root),
            candidate_at(&root, "2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 3, 4),
            FaultPlan {
                barrier: Some(Barrier::PredecessorPruneRename),
            },
        ),
        Err(MetadataJournalError::CommittedDurabilityUnknown { sequence: 2, .. })
    ));

    let metadata = root.join("update/metadata");
    let detached = root.join("update/detached-metadata");
    let result = commit_candidate_with_hook_for_test(
        authorization(&root),
        candidate_at(&root, "2026-08-17T20:00:02Z", "2026-08-17T20:00:03Z", 4, 5),
        || {
            std::fs::rename(&metadata, &detached).expect("detach verified metadata namespace");
            std::fs::create_dir(&metadata).expect("replace metadata namespace");
            std::fs::set_permissions(&metadata, std::fs::Permissions::from_mode(0o700))
                .expect("private replacement mode");
        },
    );
    assert!(matches!(
        result,
        Err(MetadataJournalError::CommittedDurabilityUnknown { sequence: 2, .. })
    ));
    assert!(detached
        .join("generations/.prune-00000000000000000001")
        .is_dir());
}

#[test]
fn cleanup_revalidates_the_live_selector_before_deletion() {
    use std::os::unix::fs::PermissionsExt;

    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    commit_candidate_for_test(
        authorization(&root),
        candidate_at(&root, "2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3),
        FaultPlan::default(),
    )
    .expect("first");
    assert!(matches!(
        commit_candidate_for_test(
            authorization(&root),
            candidate_at(&root, "2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 3, 4),
            FaultPlan {
                barrier: Some(Barrier::PredecessorPruneRename),
            },
        ),
        Err(MetadataJournalError::CommittedDurabilityUnknown { sequence: 2, .. })
    ));

    let metadata = root.join("update/metadata");
    let prune = metadata.join("generations/.prune-00000000000000000001");
    let result = cleanup_selected_with_hook_for_test(authorization(&root), || {
        let replacement = metadata.join("replacement-current.json");
        std::fs::write(&replacement, b"{}\n").expect("write hostile replacement selector");
        std::fs::set_permissions(&replacement, std::fs::Permissions::from_mode(0o600))
            .expect("private replacement mode");
        std::fs::rename(&replacement, metadata.join("current.json"))
            .expect("replace selected metadata selector");
    });
    assert!(result.is_err());
    assert!(prune.join("generation.json").is_file());
}

#[test]
fn precommit_rejects_namespace_inode_and_file_attribute_swaps() {
    use std::os::unix::fs::{symlink, PermissionsExt};

    #[derive(Clone, Copy, Debug)]
    enum Mutation {
        MetadataDirectory,
        GenerationsDirectory,
        PublishedGeneration,
        StagedSelector,
        PayloadSymlink,
        PayloadHardlink,
        PayloadMode,
    }

    for mutation in [
        Mutation::MetadataDirectory,
        Mutation::GenerationsDirectory,
        Mutation::PublishedGeneration,
        Mutation::StagedSelector,
        Mutation::PayloadSymlink,
        Mutation::PayloadHardlink,
        Mutation::PayloadMode,
    ] {
        let parent = tempfile::tempdir().expect("tempdir");
        let root = test_root(&parent);
        commit_candidate_for_test(
            authorization(&root),
            candidate_at(&root, "2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3),
            FaultPlan::default(),
        )
        .expect("commit predecessor");
        let metadata = root.join("update/metadata");
        let generations = metadata.join("generations");
        let generation = generations.join("00000000000000000002");
        let pending_selector = metadata.join(".current-00000000000000000002.json");
        let result = commit_candidate_with_precommit_hook_for_test(
            authorization(&root),
            candidate_at(&root, "2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 3, 4),
            || match mutation {
                Mutation::MetadataDirectory => {
                    std::fs::rename(&metadata, root.join("detached-metadata"))
                        .expect("detach metadata directory");
                    std::fs::create_dir(&metadata).expect("replace metadata directory");
                    std::fs::set_permissions(&metadata, std::fs::Permissions::from_mode(0o700))
                        .expect("private metadata replacement");
                }
                Mutation::GenerationsDirectory => {
                    std::fs::rename(&generations, metadata.join("detached-generations"))
                        .expect("detach generations directory");
                    std::fs::create_dir(&generations).expect("replace generations directory");
                    std::fs::set_permissions(&generations, std::fs::Permissions::from_mode(0o700))
                        .expect("private generations replacement");
                }
                Mutation::PublishedGeneration => {
                    std::fs::rename(&generation, generations.join("detached-generation"))
                        .expect("detach published generation");
                    std::fs::create_dir(&generation).expect("replace published generation");
                    std::fs::set_permissions(&generation, std::fs::Permissions::from_mode(0o700))
                        .expect("private generation replacement");
                }
                Mutation::StagedSelector => {
                    let bytes = std::fs::read(&pending_selector).expect("staged selector bytes");
                    let replacement = metadata.join("replacement-selector.json");
                    std::fs::write(&replacement, bytes).expect("replacement selector");
                    std::fs::set_permissions(&replacement, std::fs::Permissions::from_mode(0o600))
                        .expect("private selector replacement");
                    std::fs::rename(replacement, &pending_selector)
                        .expect("replace staged selector inode");
                }
                Mutation::PayloadSymlink => {
                    std::fs::remove_file(generation.join("targets.json"))
                        .expect("remove targets role");
                    symlink("trusted-root.json", generation.join("targets.json"))
                        .expect("replace targets with symlink");
                }
                Mutation::PayloadHardlink => {
                    std::fs::hard_link(
                        generation.join("targets.json"),
                        root.join("hardlinked-targets.json"),
                    )
                    .expect("create hostile hardlink");
                }
                Mutation::PayloadMode => {
                    std::fs::set_permissions(
                        generation.join("targets.json"),
                        std::fs::Permissions::from_mode(0o644),
                    )
                    .expect("weaken payload mode");
                }
            },
        );
        assert!(result.is_err(), "mutation {mutation:?} must fail closed");
    }
}
