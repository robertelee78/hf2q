#[test]
fn managed_session_store_serializes_processes_with_one_named_lock() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "lock");
    configure(&state_root, "64MiB");
    let first = open_store(&state_root).unwrap();
    assert!(matches!(
        open_store(&state_root),
        Err(ManagedSessionCacheError::Busy)
    ));
    drop(first);
    open_store(&state_root).unwrap();
}

#[test]
fn managed_session_store_rejects_root_session_and_leaf_replacement() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "replacement");
    configure(&state_root, "64MiB");
    let store = open_store(&state_root).unwrap();
    let sessions = state_root.join("cache/sessions");
    let retained = state_root.join("cache/retained-sessions");
    fs::rename(&sessions, &retained).unwrap();
    fs::create_dir(&sessions).unwrap();
    fs::set_permissions(&sessions, fs::Permissions::from_mode(0o700)).unwrap();
    let key = ManagedCheckpointKey::from_canonical_receipt(b"replacement");
    assert!(matches!(
        with_room(|| store.commit(key, b"checkpoint")),
        Err(ManagedSessionCacheError::StaleAuthorization(_))
            | Err(ManagedSessionCacheError::InvalidLayout(_))
    ));
    assert!(!sessions.join("catalogs").exists());
}

#[test]
fn managed_session_store_fails_closed_on_corruption_and_preserves_evidence() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "corrupt");
    configure(&state_root, "64MiB");
    let key = ManagedCheckpointKey::from_canonical_receipt(b"corrupt");
    {
        let store = open_store(&state_root).unwrap();
        with_room(|| store.commit(key, b"checkpoint")).unwrap();
    }
    let objects = state_root.join("cache/sessions/objects");
    let shard = fs::read_dir(&objects)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let object = fs::read_dir(&shard)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let mut bytes = fs::read(&object).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    fs::write(&object, &bytes).unwrap();
    fs::set_permissions(&object, fs::Permissions::from_mode(0o600)).unwrap();
    let orphan_shard = objects.join("ff");
    fs::create_dir(&orphan_shard).unwrap();
    fs::set_permissions(&orphan_shard, fs::Permissions::from_mode(0o700)).unwrap();
    let orphan = orphan_shard.join(format!("{}.checkpoint", "f".repeat(64)));
    fs::write(&orphan, b"orphan-evidence").unwrap();
    fs::set_permissions(&orphan, fs::Permissions::from_mode(0o600)).unwrap();
    let orphan_inode = fs::metadata(&orphan).unwrap().ino();
    assert!(open_store(&state_root).is_err());
    assert_eq!(fs::read(&object).unwrap(), bytes);
    assert_eq!(fs::read(&orphan).unwrap(), b"orphan-evidence");
    assert_eq!(fs::metadata(&orphan).unwrap().ino(), orphan_inode);
}

#[test]
fn managed_session_store_rejects_unknown_residue_without_deleting_it() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "hostile-residue");
    configure(&state_root, "64MiB");
    {
        open_store(&state_root).unwrap();
    }
    let evidence = state_root.join("cache/sessions/pending/evidence");
    fs::write(&evidence, b"hostile").unwrap();
    fs::set_permissions(&evidence, fs::Permissions::from_mode(0o600)).unwrap();
    assert!(open_store(&state_root).is_err());
    assert_eq!(fs::read(&evidence).unwrap(), b"hostile");
}

#[test]
fn managed_session_store_enforces_the_exact_quarantine_entry_cap() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "quarantine-cap");
    configure(&state_root, "64MiB");
    drop(open_store(&state_root).unwrap());
    let quarantine = state_root.join("cache/sessions/quarantine");
    for sequence in 0..128 {
        let path = quarantine.join(format!(
            "{sequence:020}-corrupt-{}.checkpoint",
            format!("{sequence:064x}")
        ));
        fs::write(&path, []).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    }
    drop(open_store(&state_root).unwrap());
    let overflow = quarantine.join(format!(
        "{:020}-corrupt-{}.checkpoint",
        128,
        format!("{:064x}", 128)
    ));
    fs::write(&overflow, b"evidence").unwrap();
    fs::set_permissions(&overflow, fs::Permissions::from_mode(0o600)).unwrap();
    assert!(matches!(
        open_store(&state_root),
        Err(ManagedSessionCacheError::InvalidLayout(_))
    ));
    assert_eq!(fs::read(&overflow).unwrap(), b"evidence");
}

#[test]
fn managed_session_store_preflights_fresh_hostile_residue_without_mutation() {
    for nested in [false, true] {
        let temp = TempDir::new().unwrap();
        let state_root = root(&temp, if nested { "nested" } else { "root" });
        configure(&state_root, "64MiB");
        let sessions = state_root.join("cache/sessions");
        let evidence = if nested {
            let pending = sessions.join("pending");
            fs::create_dir(&pending).unwrap();
            fs::set_permissions(&pending, fs::Permissions::from_mode(0o700)).unwrap();
            pending.join("evidence")
        } else {
            sessions.join("evidence")
        };
        fs::write(&evidence, b"hostile").unwrap();
        fs::set_permissions(&evidence, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(open_store(&state_root).is_err());
        assert_eq!(fs::read(&evidence).unwrap(), b"hostile");
        for name in [
            ".managed-session-cache.lock",
            "objects",
            "catalogs",
            "quarantine",
        ] {
            assert!(!sessions.join(name).exists());
        }
    }
}

#[test]
fn managed_session_store_preserves_hostile_reserved_mode_and_link_residue() {
    for case in ["directory-mode", "lock-mode", "symlink", "hardlink"] {
        let temp = TempDir::new().unwrap();
        let state_root = root(&temp, case);
        configure(&state_root, "64MiB");
        let sessions = state_root.join("cache/sessions");
        let mut external = None;
        match case {
            "directory-mode" => {
                let pending = sessions.join("pending");
                fs::create_dir(&pending).unwrap();
                fs::set_permissions(&pending, fs::Permissions::from_mode(0o755)).unwrap();
            }
            "lock-mode" => {
                let lock = sessions.join(".managed-session-cache.lock");
                fs::write(&lock, []).unwrap();
                fs::set_permissions(&lock, fs::Permissions::from_mode(0o644)).unwrap();
            }
            "symlink" => {
                let target = temp.path().join("symlink-target");
                fs::create_dir(&target).unwrap();
                fs::set_permissions(&target, fs::Permissions::from_mode(0o700)).unwrap();
                symlink(&target, sessions.join("pending")).unwrap();
                external = Some(target);
            }
            "hardlink" => {
                let target = temp.path().join("hardlink-target");
                fs::write(&target, []).unwrap();
                fs::set_permissions(&target, fs::Permissions::from_mode(0o600)).unwrap();
                fs::hard_link(&target, sessions.join(".managed-session-cache.lock")).unwrap();
                external = Some(target);
            }
            _ => unreachable!(),
        }

        let before = snapshot_tree(&sessions);
        let external_before = external
            .as_ref()
            .map(|path| fs::symlink_metadata(path).unwrap());
        assert!(matches!(
            with_room(|| open_store(&state_root)),
            Err(ManagedSessionCacheError::InvalidLayout(_))
        ));
        assert_eq!(snapshot_tree(&sessions), before, "case {case}");

        match case {
            "symlink" => {
                let link = sessions.join("pending");
                let before = external_before.unwrap();
                assert!(fs::symlink_metadata(&link)
                    .unwrap()
                    .file_type()
                    .is_symlink());
                assert_eq!(fs::read_link(&link).unwrap(), external.unwrap());
                let after = fs::metadata(fs::read_link(link).unwrap()).unwrap();
                assert_eq!((after.ino(), after.mode()), (before.ino(), before.mode()));
            }
            "hardlink" => {
                let lock = fs::metadata(sessions.join(".managed-session-cache.lock")).unwrap();
                let target = fs::metadata(external.unwrap()).unwrap();
                assert_eq!(lock.ino(), target.ino());
                assert_eq!(lock.nlink(), 2);
                assert_eq!(target.nlink(), 2);
                assert_eq!(target.mode() & 0o7777, 0o600);
            }
            _ => {}
        }
    }
}

#[test]
fn managed_session_store_possible_commit_poison_requires_reopen() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "poison");
    configure(&state_root, "64MiB");
    let key = ManagedCheckpointKey::from_canonical_receipt(b"poison");
    let store = open_store(&state_root).unwrap();
    let error = with_room(|| {
        commit_with_test_hook(&store, key, b"checkpoint", |barrier| {
            if barrier == ManagedCacheBarrier::CatalogPublished {
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
        with_room(|| store.commit(key, b"checkpoint")),
        Err(ManagedSessionCacheError::RecoveryRequired)
    ));
    assert!(matches!(
        store.restore(key),
        Err(ManagedSessionCacheError::RecoveryRequired)
    ));
    drop(store);
    let reopened = open_store(&state_root).unwrap();
    assert_eq!(reopened.restore(key).unwrap(), Some(b"checkpoint".to_vec()));
}

#[test]
fn managed_session_store_rejects_same_byte_inode_swaps_at_publication_hooks() {
    for barrier in [
        ManagedCacheBarrier::ObjectPublished,
        ManagedCacheBarrier::CatalogPublished,
        ManagedCacheBarrier::CatalogDirectorySynced,
    ] {
        let temp = TempDir::new().unwrap();
        let state_root = root(&temp, barrier.as_str());
        configure(&state_root, "64MiB");
        let key = ManagedCheckpointKey::from_canonical_receipt(barrier.as_str().as_bytes());
        let store = open_store(&state_root).unwrap();
        let sessions = state_root.join("cache/sessions");
        let error = with_room(|| {
            commit_with_test_hook(&store, key, b"checkpoint", |seen| {
                if seen != barrier {
                    return Ok(());
                }
                let parent = if barrier == ManagedCacheBarrier::ObjectPublished {
                    let objects = sessions.join("objects");
                    fs::read_dir(objects)
                        .unwrap()
                        .next()
                        .unwrap()
                        .unwrap()
                        .path()
                } else {
                    sessions.join("catalogs")
                };
                let final_path = fs::read_dir(&parent)
                    .unwrap()
                    .next()
                    .unwrap()
                    .unwrap()
                    .path();
                let bytes = fs::read(&final_path).unwrap();
                let evidence = state_root.join(format!("evidence-{}", barrier.as_str()));
                fs::rename(&final_path, &evidence).unwrap();
                fs::write(&final_path, bytes).unwrap();
                fs::set_permissions(&final_path, fs::Permissions::from_mode(0o600)).unwrap();
                Ok(())
            })
        })
        .unwrap_err();
        if barrier == ManagedCacheBarrier::ObjectPublished {
            assert!(matches!(error, ManagedSessionCacheError::InvalidLayout(_)));
        } else {
            assert!(matches!(
                error,
                ManagedSessionCacheError::CommittedDurabilityUnknown(_)
            ));
        }
    }
}

#[test]
fn managed_session_store_rejects_object_and_shard_swaps_before_catalog_commit() {
    for replace_shard in [false, true] {
        let temp = TempDir::new().unwrap();
        let state_root = root(
            &temp,
            if replace_shard {
                "replace-object-shard"
            } else {
                "replace-object-leaf"
            },
        );
        configure(&state_root, "64MiB");
        let key =
            ManagedCheckpointKey::from_canonical_receipt(state_root.as_os_str().as_encoded_bytes());
        let store = open_store(&state_root).unwrap();
        let objects = state_root.join("cache/sessions/objects");
        let error = with_room(|| {
            commit_with_test_hook(&store, key, b"checkpoint", |barrier| {
                if barrier != ManagedCacheBarrier::BeforeCatalogPublish {
                    return Ok(());
                }
                let shard = fs::read_dir(&objects)
                    .unwrap()
                    .next()
                    .unwrap()
                    .unwrap()
                    .path();
                let object = fs::read_dir(&shard)
                    .unwrap()
                    .next()
                    .unwrap()
                    .unwrap()
                    .path();
                let bytes = fs::read(&object).unwrap();
                if replace_shard {
                    let evidence = state_root.join("retained-object-shard");
                    fs::rename(&shard, &evidence).unwrap();
                    fs::create_dir(&shard).unwrap();
                    fs::set_permissions(&shard, fs::Permissions::from_mode(0o700)).unwrap();
                    let replacement = shard.join(object.file_name().unwrap());
                    fs::write(&replacement, bytes).unwrap();
                    fs::set_permissions(&replacement, fs::Permissions::from_mode(0o600)).unwrap();
                } else {
                    let evidence = state_root.join("retained-object-leaf");
                    fs::rename(&object, &evidence).unwrap();
                    fs::write(&object, bytes).unwrap();
                    fs::set_permissions(&object, fs::Permissions::from_mode(0o600)).unwrap();
                }
                Ok(())
            })
        })
        .unwrap_err();
        assert!(matches!(error, ManagedSessionCacheError::InvalidLayout(_)));
    }
}
