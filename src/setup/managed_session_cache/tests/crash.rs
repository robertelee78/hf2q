#[test]
fn managed_session_store_destructive_barrier_faults_require_exact_recovery() {
    for barrier in [
        ManagedCacheBarrier::BeforeObjectDelete,
        ManagedCacheBarrier::ObjectDeleted,
    ] {
        let temp = TempDir::new().unwrap();
        let state_root = root(&temp, barrier.as_str());
        configure(&state_root, "8MiB");
        let first = ManagedCheckpointKey::from_canonical_receipt(b"delete-first");
        let second = ManagedCheckpointKey::from_canonical_receipt(b"delete-second");
        let first_payload = vec![0x31; 4 * 1024 * 1024];
        let second_payload = vec![0x32; 4 * 1024 * 1024];
        let store = open_store(&state_root).unwrap();
        with_room(|| store.commit(first, &first_payload)).unwrap();
        let error = with_room(|| {
            commit_with_test_hook(&store, second, &second_payload, |seen| {
                if seen == barrier {
                    Err(ManagedSessionCacheError::Filesystem("injected".to_owned()))
                } else {
                    Ok(())
                }
            })
        })
        .unwrap_err();
        assert!(matches!(
            error,
            ManagedSessionCacheError::CommittedDurabilityUnknown(_)
        ));
        assert!(matches!(
            store.restore(first),
            Err(ManagedSessionCacheError::RecoveryRequired)
        ));
        drop(store);
        let reopened = open_store(&state_root).unwrap();
        assert_eq!(reopened.restore(first).unwrap(), None);
        assert_eq!(reopened.restore(second).unwrap(), None);
        with_room(|| reopened.commit(second, &second_payload)).unwrap();
        assert_eq!(reopened.restore(second).unwrap(), Some(second_payload));
    }

    for barrier in [
        ManagedCacheBarrier::BeforeCatalogHistoryPrune,
        ManagedCacheBarrier::CatalogHistoryPruned,
    ] {
        let temp = TempDir::new().unwrap();
        let state_root = root(&temp, barrier.as_str());
        configure(&state_root, "64MiB");
        let store = open_store(&state_root).unwrap();
        for value in 0..4 {
            let key = ManagedCheckpointKey::from_canonical_receipt(
                format!("fault-history-{value}").as_bytes(),
            );
            with_room(|| store.commit(key, b"checkpoint")).unwrap();
        }
        let fifth = ManagedCheckpointKey::from_canonical_receipt(b"fault-history-4");
        let error = with_room(|| {
            commit_with_test_hook(&store, fifth, b"checkpoint", |seen| {
                if seen == barrier {
                    Err(ManagedSessionCacheError::Filesystem("injected".to_owned()))
                } else {
                    Ok(())
                }
            })
        })
        .unwrap_err();
        if barrier == ManagedCacheBarrier::CatalogHistoryPruned {
            assert!(matches!(
                error,
                ManagedSessionCacheError::CommittedDurabilityUnknown(_)
            ));
            assert!(matches!(
                store.restore(fifth),
                Err(ManagedSessionCacheError::RecoveryRequired)
            ));
        }
        drop(store);
        let reopened = open_store(&state_root).unwrap();
        assert_eq!(reopened.restore(fifth).unwrap(), None);
        with_room(|| reopened.commit(fifth, b"checkpoint")).unwrap();
        assert_eq!(
            reopened.restore(fifth).unwrap(),
            Some(b"checkpoint".to_vec())
        );
    }
}

#[test]
fn managed_session_store_faults_recover_old_or_new_catalog_state() {
    use ManagedCacheBarrier::*;
    for barrier in [
        AuthorizationValidated,
        ObjectPartialSynced,
        BeforeObjectPublish,
        ObjectPublished,
        ObjectDirectorySynced,
        CatalogPartialSynced,
        BeforeCatalogPublish,
        CatalogPublished,
        CatalogDirectorySynced,
        EndpointSynced,
    ] {
        let temp = TempDir::new().unwrap();
        let state_root = root(&temp, barrier.as_str());
        configure(&state_root, "64MiB");
        let key = ManagedCheckpointKey::from_canonical_receipt(barrier.as_str().as_bytes());
        let store = open_store(&state_root).unwrap();
        let error = with_room(|| {
            commit_with_test_hook(&store, key, b"checkpoint", |seen| {
                if seen == barrier {
                    Err(ManagedSessionCacheError::Filesystem(format!(
                        "injected at {seen:?}"
                    )))
                } else {
                    Ok(())
                }
            })
        })
        .unwrap_err();
        if matches!(
            barrier,
            CatalogPublished | CatalogDirectorySynced | EndpointSynced
        ) {
            assert!(matches!(
                error,
                ManagedSessionCacheError::CommittedDurabilityUnknown(_)
            ));
        }
        drop(store);
        let reopened = open_store(&state_root).unwrap();
        let observed = reopened.restore(key).unwrap();
        if matches!(
            barrier,
            AuthorizationValidated
                | ObjectPartialSynced
                | BeforeObjectPublish
                | ObjectPublished
                | ObjectDirectorySynced
                | CatalogPartialSynced
                | BeforeCatalogPublish
        ) {
            assert_eq!(observed, None, "barrier {barrier:?}");
        } else {
            assert_eq!(
                observed,
                Some(b"checkpoint".to_vec()),
                "barrier {barrier:?}"
            );
        }
        with_room(|| reopened.commit(key, b"checkpoint")).unwrap();
        assert_eq!(reopened.restore(key).unwrap(), Some(b"checkpoint".to_vec()));
    }
}

#[test]
fn managed_session_store_sigabrt_at_every_commit_barrier_recovers() {
    const CHILD: &str = "HF2Q_MANAGED_CACHE_CRASH_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let root = PathBuf::from(std::env::var_os("HF2Q_MANAGED_CACHE_ROOT").unwrap());
        let store = open_store(&root).unwrap();
        let barrier = std::env::var("HF2Q_MANAGED_CACHE_ABORT_AT").unwrap();
        let key = ManagedCheckpointKey::from_canonical_receipt(barrier.as_bytes());
        with_room(|| store.commit(key, b"checkpoint")).unwrap();
        return;
    }

    use ManagedCacheBarrier::*;
    for barrier in [
        AuthorizationValidated,
        ObjectPartialSynced,
        BeforeObjectPublish,
        ObjectPublished,
        ObjectDirectorySynced,
        CatalogPartialSynced,
        BeforeCatalogPublish,
        CatalogPublished,
        CatalogDirectorySynced,
        EndpointSynced,
    ] {
        let temp = TempDir::new().unwrap();
        let state_root = root(&temp, barrier.as_str());
        configure(&state_root, "64MiB");
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .arg(
                "setup::managed_session_cache::tests::managed_session_store_sigabrt_at_every_commit_barrier_recovers",
            )
            .arg("--exact")
            .env(CHILD, "1")
            .env("HF2Q_MANAGED_CACHE_ROOT", &state_root)
            .env("HF2Q_MANAGED_CACHE_ABORT_AT", barrier.as_str())
            .status()
            .unwrap();
        assert_eq!(status.signal(), Some(libc::SIGABRT), "barrier {barrier:?}");
        let store = open_store(&state_root).unwrap();
        let key = ManagedCheckpointKey::from_canonical_receipt(barrier.as_str().as_bytes());
        let observed = store.restore(key).unwrap();
        if matches!(
            barrier,
            AuthorizationValidated
                | ObjectPartialSynced
                | BeforeObjectPublish
                | ObjectPublished
                | ObjectDirectorySynced
                | CatalogPartialSynced
                | BeforeCatalogPublish
        ) {
            assert_eq!(observed, None, "barrier {barrier:?}");
        } else if barrier == CatalogPublished {
            assert!(
                observed.is_none() || observed == Some(b"checkpoint".to_vec()),
                "barrier {barrier:?}"
            );
        } else {
            assert_eq!(
                observed,
                Some(b"checkpoint".to_vec()),
                "barrier {barrier:?}"
            );
        }
        with_room(|| store.commit(key, b"checkpoint")).unwrap();
        assert_eq!(store.restore(key).unwrap(), Some(b"checkpoint".to_vec()));
    }
}

#[test]
fn managed_session_store_sigabrt_at_destructive_barriers_recovers() {
    const CHILD: &str = "HF2Q_MANAGED_CACHE_DESTRUCTIVE_CRASH_CHILD";
    if let Some(kind) = std::env::var_os(CHILD) {
        let root = PathBuf::from(std::env::var_os("HF2Q_MANAGED_CACHE_ROOT").unwrap());
        let store = open_store(&root).unwrap();
        if kind == "delete" {
            let second = ManagedCheckpointKey::from_canonical_receipt(b"abort-delete-second");
            let payload = vec![0x62; 4 * 1024 * 1024];
            with_room(|| store.commit(second, &payload)).unwrap();
        } else {
            let fifth = ManagedCheckpointKey::from_canonical_receipt(b"abort-history-4");
            with_room(|| store.commit(fifth, b"checkpoint")).unwrap();
        }
        return;
    }

    for barrier in [
        ManagedCacheBarrier::BeforeObjectDelete,
        ManagedCacheBarrier::ObjectDeleted,
        ManagedCacheBarrier::BeforeCatalogHistoryPrune,
        ManagedCacheBarrier::CatalogHistoryPruned,
    ] {
        let delete = matches!(
            barrier,
            ManagedCacheBarrier::BeforeObjectDelete | ManagedCacheBarrier::ObjectDeleted
        );
        let temp = TempDir::new().unwrap();
        let state_root = root(&temp, barrier.as_str());
        configure(&state_root, if delete { "8MiB" } else { "64MiB" });
        if delete {
            let store = open_store(&state_root).unwrap();
            let first = ManagedCheckpointKey::from_canonical_receipt(b"abort-delete-first");
            let payload = vec![0x61; 4 * 1024 * 1024];
            with_room(|| store.commit(first, &payload)).unwrap();
        } else {
            let store = open_store(&state_root).unwrap();
            for value in 0..4 {
                let key = ManagedCheckpointKey::from_canonical_receipt(
                    format!("abort-history-{value}").as_bytes(),
                );
                with_room(|| store.commit(key, b"checkpoint")).unwrap();
            }
        }
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .arg(
                "setup::managed_session_cache::tests::managed_session_store_sigabrt_at_destructive_barriers_recovers",
            )
            .arg("--exact")
            .env(CHILD, if delete { "delete" } else { "history" })
            .env("HF2Q_MANAGED_CACHE_ROOT", &state_root)
            .env("HF2Q_MANAGED_CACHE_ABORT_AT", barrier.as_str())
            .status()
            .unwrap();
        assert_eq!(status.signal(), Some(libc::SIGABRT), "barrier {barrier:?}");
        let reopened = open_store(&state_root).unwrap();
        if delete {
            let first = ManagedCheckpointKey::from_canonical_receipt(b"abort-delete-first");
            let second = ManagedCheckpointKey::from_canonical_receipt(b"abort-delete-second");
            assert_eq!(reopened.restore(first).unwrap(), None);
            assert_eq!(reopened.restore(second).unwrap(), None);
            let payload = vec![0x62; 4 * 1024 * 1024];
            with_room(|| reopened.commit(second, &payload)).unwrap();
            assert_eq!(reopened.restore(second).unwrap(), Some(payload));
        } else {
            let fifth = ManagedCheckpointKey::from_canonical_receipt(b"abort-history-4");
            assert_eq!(reopened.restore(fifth).unwrap(), None);
            with_room(|| reopened.commit(fifth, b"checkpoint")).unwrap();
            assert_eq!(
                reopened.restore(fifth).unwrap(),
                Some(b"checkpoint".to_vec())
            );
        }
    }
}

#[test]
fn managed_session_store_recovers_umask_filtered_creation_crashes() {
    const CHILD: &str = "HF2Q_MANAGED_CACHE_UMASK_CRASH_CHILD";
    if let Some(kind) = std::env::var_os(CHILD) {
        unsafe { libc::umask(0o777) };
        let root = PathBuf::from(std::env::var_os("HF2Q_MANAGED_CACHE_ROOT").unwrap());
        if kind == "top-directory" || kind == "lock-file" {
            open_store(&root).unwrap();
        } else {
            let store = open_store(&root).unwrap();
            let key = ManagedCheckpointKey::from_canonical_receipt(b"umask-crash-checkpoint");
            with_room(|| store.commit(key, b"checkpoint")).unwrap();
        }
        return;
    }

    for (kind, barrier) in [
        (
            "top-directory",
            ManagedCacheBarrier::DirectoryCreatedBeforeMode,
        ),
        ("lock-file", ManagedCacheBarrier::FileCreatedBeforeMode),
        (
            "shard-directory",
            ManagedCacheBarrier::DirectoryCreatedBeforeMode,
        ),
        ("object-partial", ManagedCacheBarrier::FileCreatedBeforeMode),
    ] {
        let temp = TempDir::new().unwrap();
        let state_root = root(&temp, kind);
        configure(&state_root, "64MiB");
        let sessions = state_root.join("cache/sessions");
        if kind == "lock-file" {
            for name in ["pending", "objects", "catalogs", "quarantine"] {
                let path = sessions.join(name);
                fs::create_dir(&path).unwrap();
                fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
            }
        } else if kind == "shard-directory" || kind == "object-partial" {
            drop(open_store(&state_root).unwrap());
        }
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .arg(
                "setup::managed_session_cache::tests::managed_session_store_recovers_umask_filtered_creation_crashes",
            )
            .arg("--exact")
            .env(CHILD, kind)
            .env("HF2Q_MANAGED_CACHE_ROOT", &state_root)
            .env("HF2Q_MANAGED_CACHE_ABORT_AT", barrier.as_str())
            .status()
            .unwrap();
        assert_eq!(status.signal(), Some(libc::SIGABRT), "case {kind}");

        let store = open_store(&state_root).unwrap();
        if kind == "shard-directory" || kind == "object-partial" {
            let key = ManagedCheckpointKey::from_canonical_receipt(b"umask-crash-checkpoint");
            assert_eq!(store.restore(key).unwrap(), None);
            with_room(|| store.commit(key, b"checkpoint")).unwrap();
            assert_eq!(store.restore(key).unwrap(), Some(b"checkpoint".to_vec()));
        }
        assert_eq!(
            fs::metadata(&sessions).unwrap().permissions().mode() & 0o7777,
            0o700
        );
    }
}
