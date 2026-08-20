#[test]
fn managed_session_catalog_v1_golden_is_canonical_and_pinned() {
    let catalog = CatalogV1::empty()
        .with_entry(CatalogEntryV1::new("0".repeat(64), "1".repeat(64), 84))
        .unwrap();
    let expected = include_bytes!("../../testdata/managed_session_catalog_v1.json");
    let bytes = catalog.to_canonical_bytes().unwrap();
    assert_eq!(bytes, expected);
    assert_eq!(
        sha256_hex(&bytes),
        "629d13bb3f66e8535f77b58e55c8d7e4a5e37d8c39736c01bde7df6fe37e9e44"
    );
    assert_eq!(CatalogV1::parse_exact(&bytes).unwrap(), catalog);
}

#[test]
fn managed_session_store_maps_enospc_and_edquot_to_storage_full() {
    for (errno, raw) in [
        (rustix::io::Errno::NOSPC, libc::ENOSPC),
        (rustix::io::Errno::DQUOT, libc::EDQUOT),
    ] {
        assert!(matches!(
            super::unix::test_storage_error_mapping(errno),
            ManagedSessionCacheError::StorageFull
        ));
        assert!(matches!(
            super::transaction::test_std_io_error_mapping(std::io::Error::from_raw_os_error(raw)),
            ManagedSessionCacheError::StorageFull
        ));
    }
}

#[test]
fn managed_session_store_round_trips_and_recovers_only_cataloged_objects() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "round-trip");
    configure(&state_root, "64MiB");
    let key = ManagedCheckpointKey::from_canonical_receipt(b"receipt-a");
    {
        let store = open_store(&state_root).unwrap();
        let outcome = with_room(|| store.commit(key, b"checkpoint-a")).unwrap();
        assert!(matches!(outcome, ManagedCommitOutcome::Published));
        assert_eq!(store.restore(key).unwrap(), Some(b"checkpoint-a".to_vec()));
        let outcome = with_room(|| store.commit(key, b"checkpoint-a")).unwrap();
        assert!(matches!(outcome, ManagedCommitOutcome::AlreadyPresent));
        assert_eq!(format!("{store:?}"), "ManagedSessionCache(<redacted>)");
    }
    let reopened = open_store(&state_root).unwrap();
    assert_eq!(
        reopened.restore(key).unwrap(),
        Some(b"checkpoint-a".to_vec())
    );
    assert!(!state_root
        .join("cache/sessions/pending")
        .join(OBJECT_PARTIAL)
        .exists());
    assert!(!state_root
        .join("cache/sessions/pending")
        .join(CATALOG_PARTIAL)
        .exists());
}

#[test]
fn managed_session_policy_absent_or_disabled_creates_no_managed_state() {
    let temp = TempDir::new().unwrap();
    let absent = root(&temp, "absent");
    assert!(matches!(
        authorize_session_cache_policy(&absent).unwrap(),
        SessionCachePolicyAuthorization::Absent
    ));
    assert!(!absent.exists());

    let disabled = root(&temp, "disabled");
    let args = SetupArgs {
        session_cache: Some(SessionCacheChoice::Off),
        session_cache_limit: None,
        state_root: Some(disabled.clone()),
    };
    execute(
        args,
        &FakeProbe,
        false,
        &mut Cursor::new(Vec::<u8>::new()),
        &mut Vec::new(),
    )
    .unwrap();
    assert!(matches!(
        authorize_session_cache_policy(&disabled).unwrap(),
        SessionCachePolicyAuthorization::Disabled(_)
    ));
    for name in [
        ".managed-session-cache.lock",
        "pending",
        "objects",
        "catalogs",
        "quarantine",
    ] {
        assert!(!disabled.join("cache/sessions").join(name).exists());
    }
}

#[test]
fn managed_session_store_tiny_positive_limit_creates_no_managed_state() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "tiny-positive-limit");
    configure(&state_root, "1B");
    assert!(matches!(
        open_store(&state_root),
        Err(ManagedSessionCacheError::QuotaExceeded)
    ));
    for name in [
        ".managed-session-cache.lock",
        "pending",
        "objects",
        "catalogs",
        "quarantine",
    ] {
        assert!(!state_root.join("cache/sessions").join(name).exists());
    }
}

#[test]
fn managed_session_store_existing_layout_tiny_limit_is_read_only() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "existing-tiny-limit");
    configure(&state_root, "64MiB");
    let key = ManagedCheckpointKey::from_canonical_receipt(b"existing-tiny-limit");
    {
        let store = open_store(&state_root).unwrap();
        with_room(|| store.commit(key, b"checkpoint")).unwrap();
    }
    configure(&state_root, "1B");
    let sessions = state_root.join("cache/sessions");
    fs::set_permissions(sessions.join("pending"), fs::Permissions::from_mode(0o000)).unwrap();
    let before = snapshot_tree(&sessions);
    assert!(matches!(
        open_store(&state_root),
        Err(ManagedSessionCacheError::QuotaExceeded)
    ));
    assert_eq!(snapshot_tree(&sessions), before);
}

#[test]
fn managed_session_store_low_space_open_creates_no_managed_state() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "low-space-open");
    configure(&state_root, "64MiB");
    assert!(matches!(
        with_test_volume_space(500 * GIB, 20 * GIB, 4096, || open_store(&state_root)),
        Err(ManagedSessionCacheError::FreeSpaceFloor)
    ));
    for name in [
        ".managed-session-cache.lock",
        "pending",
        "objects",
        "catalogs",
        "quarantine",
    ] {
        assert!(!state_root.join("cache/sessions").join(name).exists());
    }
}

#[test]
fn managed_session_store_low_space_reopens_existing_state_read_only() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "low-space-existing");
    configure(&state_root, "64MiB");
    let key = ManagedCheckpointKey::from_canonical_receipt(b"low-space-existing");
    {
        let store = open_store(&state_root).unwrap();
        with_room(|| store.commit(key, b"checkpoint")).unwrap();
    }
    let store = with_test_volume_space(500 * GIB, 1, 4096, || open_store(&state_root)).unwrap();
    assert_eq!(store.restore(key).unwrap(), Some(b"checkpoint".to_vec()));
}

#[test]
fn managed_session_store_refuses_oversized_checkpoint_before_temp_creation() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "oversized");
    configure(&state_root, "4MiB");
    let store = open_store(&state_root).unwrap();
    let key = ManagedCheckpointKey::from_canonical_receipt(b"oversized");
    let payload = vec![0x5a; 5 * 1024 * 1024];
    assert!(matches!(
        with_room(|| store.commit(key, &payload)),
        Err(ManagedSessionCacheError::QuotaExceeded)
    ));
    assert!(!state_root
        .join("cache/sessions/pending")
        .join(OBJECT_PARTIAL)
        .exists());
    assert!(store.restore(key).unwrap().is_none());
}

#[test]
fn managed_session_store_rejects_hostile_sparse_catalog_before_payload_allocation() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "hostile-sparse-catalog");
    configure(&state_root, "64MiB");
    drop(open_store(&state_root).unwrap());

    let sessions = state_root.join("cache/sessions");
    let digest = "1".repeat(64);
    let shard = sessions.join("objects/11");
    fs::create_dir(&shard).unwrap();
    fs::set_permissions(&shard, fs::Permissions::from_mode(0o700)).unwrap();
    let object = shard.join(format!("{digest}.checkpoint"));
    let file = fs::File::create(&object).unwrap();
    file.set_len(64 * 1024 * 1024 + 1).unwrap();
    fs::set_permissions(&object, fs::Permissions::from_mode(0o600)).unwrap();

    let catalog = CatalogV1::empty()
        .with_entry(CatalogEntryV1::new(
            "0".repeat(64),
            digest,
            64 * 1024 * 1024 + 1,
        ))
        .unwrap();
    let bytes = catalog.to_canonical_bytes().unwrap();
    let catalog_path = sessions.join("catalogs").join(format!(
        "{:020}-{}.catalog",
        catalog.generation(),
        sha256_hex(&bytes)
    ));
    fs::write(&catalog_path, &bytes).unwrap();
    fs::set_permissions(&catalog_path, fs::Permissions::from_mode(0o600)).unwrap();

    assert!(matches!(
        open_store(&state_root),
        Err(ManagedSessionCacheError::QuotaExceeded)
    ));
    assert_eq!(fs::metadata(&object).unwrap().len(), 64 * 1024 * 1024 + 1);
    assert_eq!(fs::read(&catalog_path).unwrap(), bytes);
}

#[test]
fn managed_session_store_rejects_aggregate_sparse_catalog_before_object_reads() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "aggregate-sparse-catalog");
    configure(&state_root, "64MiB");
    drop(open_store(&state_root).unwrap());

    let sessions = state_root.join("cache/sessions");
    let object_bytes = 40 * 1024 * 1024;
    let mut catalog = CatalogV1::empty();
    let mut evidence = Vec::new();
    for (key_digit, digest_digit) in [('0', '1'), ('2', '3')] {
        let digest = digest_digit.to_string().repeat(64);
        let shard = sessions.join("objects").join(&digest[..2]);
        fs::create_dir(&shard).unwrap();
        fs::set_permissions(&shard, fs::Permissions::from_mode(0o700)).unwrap();
        let object = shard.join(format!("{digest}.checkpoint"));
        let file = fs::File::create(&object).unwrap();
        file.set_len(object_bytes).unwrap();
        fs::set_permissions(&object, fs::Permissions::from_mode(0o600)).unwrap();
        evidence.push((object.clone(), fs::metadata(&object).unwrap().ino()));
        catalog = catalog
            .with_entry(CatalogEntryV1::new(
                key_digit.to_string().repeat(64),
                digest,
                object_bytes,
            ))
            .unwrap();
    }
    let bytes = catalog.to_canonical_bytes().unwrap();
    let catalog_path = sessions.join("catalogs").join(format!(
        "{:020}-{}.catalog",
        catalog.generation(),
        sha256_hex(&bytes)
    ));
    fs::write(&catalog_path, &bytes).unwrap();
    fs::set_permissions(&catalog_path, fs::Permissions::from_mode(0o600)).unwrap();

    assert!(matches!(
        open_store(&state_root),
        Err(ManagedSessionCacheError::QuotaExceeded)
    ));
    for (path, inode) in evidence {
        assert_eq!(fs::metadata(&path).unwrap().ino(), inode);
        assert_eq!(fs::metadata(&path).unwrap().len(), object_bytes);
    }
    assert_eq!(fs::read(&catalog_path).unwrap(), bytes);
}

#[test]
fn managed_session_store_enforces_one_aggregate_cap_across_logical_namespaces() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "aggregate");
    configure(&state_root, "8MiB");
    let store = open_store(&state_root).unwrap();
    let first = ManagedCheckpointKey::from_canonical_receipt(b"qwen-model-a/config-a");
    let second = ManagedCheckpointKey::from_canonical_receipt(b"qwen-model-b/config-b");
    let first_payload = vec![0x11; 4 * 1024 * 1024];
    let second_payload = vec![0x22; 4 * 1024 * 1024];
    with_room(|| store.commit(first, &first_payload)).unwrap();
    with_room(|| store.commit(second, &second_payload)).unwrap();
    assert!(store.restore(first).unwrap().is_none());
    assert_eq!(store.restore(second).unwrap(), Some(second_payload));
}

#[test]
fn managed_session_store_replaces_one_key_at_a_tight_aggregate_cap() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "tight-replacement");
    configure(&state_root, "8MiB");
    let key = ManagedCheckpointKey::from_canonical_receipt(b"tight-replacement");
    let first = vec![0x81; 4 * 1024 * 1024];
    let second = vec![0x82; 4 * 1024 * 1024];
    {
        let store = open_store(&state_root).unwrap();
        with_room(|| store.commit(key, &first)).unwrap();
        with_room(|| store.commit(key, &second)).unwrap();
        assert_eq!(store.restore(key).unwrap(), Some(second.clone()));
    }
    let reopened = open_store(&state_root).unwrap();
    assert_eq!(reopened.restore(key).unwrap(), Some(second));
}

#[test]
fn managed_session_store_rebinds_multiple_objects_in_one_shard() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "same-shard");
    configure(&state_root, "64MiB");
    let mut first_by_shard = std::collections::BTreeMap::new();
    let (first, second) = (0u64..10_000)
        .find_map(|value| {
            let receipt = format!("same-shard-{value}");
            let key = ManagedCheckpointKey::from_canonical_receipt(receipt.as_bytes());
            let object = encode_object(key, b"checkpoint").unwrap();
            let shard = object.digest[..2].to_owned();
            first_by_shard.insert(shard, key).map(|first| (first, key))
        })
        .expect("pigeonhole search must find one shared two-hex shard");
    let store = open_store(&state_root).unwrap();
    with_room(|| store.commit(first, b"checkpoint")).unwrap();
    with_room(|| store.commit(second, b"checkpoint")).unwrap();
    assert_eq!(store.restore(first).unwrap(), Some(b"checkpoint".to_vec()));
    assert_eq!(store.restore(second).unwrap(), Some(b"checkpoint".to_vec()));
}

#[test]
fn managed_session_store_evicts_before_crossing_the_free_space_floor() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "free-floor-eviction");
    configure(&state_root, "16MiB");
    let first = ManagedCheckpointKey::from_canonical_receipt(b"free-floor-first");
    let second = ManagedCheckpointKey::from_canonical_receipt(b"free-floor-second");
    let payload = vec![0x45; 4 * 1024 * 1024];
    let store = open_store(&state_root).unwrap();
    with_room(|| store.commit(first, &payload)).unwrap();
    let seen = std::cell::RefCell::new(Vec::new());
    with_test_volume_space(100 * GIB, 20 * GIB + 1024 * 1024, 4096, || {
        commit_with_test_hook(&store, second, &payload, |barrier| {
            seen.borrow_mut().push(barrier);
            Ok(())
        })
    })
    .unwrap();
    assert!(store.restore(first).unwrap().is_none());
    assert_eq!(store.restore(second).unwrap(), Some(payload));
    let seen = seen.into_inner();
    assert!(seen.contains(&ManagedCacheBarrier::BeforeObjectDelete));
    assert!(seen.contains(&ManagedCacheBarrier::ObjectDeleted));
}

#[test]
fn managed_session_store_prunes_catalog_history_through_exact_barriers() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "catalog-history");
    configure(&state_root, "64MiB");
    let store = open_store(&state_root).unwrap();
    for value in 0..4 {
        let key =
            ManagedCheckpointKey::from_canonical_receipt(format!("history-{value}").as_bytes());
        with_room(|| store.commit(key, b"checkpoint")).unwrap();
    }
    let fifth = ManagedCheckpointKey::from_canonical_receipt(b"history-4");
    let seen = std::cell::RefCell::new(Vec::new());
    with_room(|| {
        commit_with_test_hook(&store, fifth, b"checkpoint", |barrier| {
            seen.borrow_mut().push(barrier);
            Ok(())
        })
    })
    .unwrap();
    let seen = seen.into_inner();
    assert!(seen.contains(&ManagedCacheBarrier::BeforeCatalogHistoryPrune));
    assert!(seen.contains(&ManagedCacheBarrier::CatalogHistoryPruned));
    assert_eq!(
        fs::read_dir(state_root.join("cache/sessions/catalogs"))
            .unwrap()
            .count(),
        4
    );
}

#[test]
fn managed_session_store_checks_descriptor_volume_free_space_before_every_write() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "space-floor");
    configure(&state_root, "64MiB");
    let store = open_store(&state_root).unwrap();
    let key = ManagedCheckpointKey::from_canonical_receipt(b"space-floor");
    let result = with_test_volume_space(100 * GIB, 19 * GIB, 4096, || {
        store.commit(key, b"checkpoint")
    });
    assert!(matches!(
        result,
        Err(ManagedSessionCacheError::FreeSpaceFloor)
    ));
    assert!(store.restore(key).unwrap().is_none());
}

#[test]
fn managed_session_store_cleans_an_ineligible_object_after_the_floor_drops() {
    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "post-object-floor-drop");
    configure(&state_root, "64MiB");
    let store = open_store(&state_root).unwrap();
    let key = ManagedCheckpointKey::from_canonical_receipt(b"post-object-floor-drop");
    let error = with_room(|| {
        commit_with_test_hook(&store, key, b"checkpoint", |barrier| {
            if barrier == ManagedCacheBarrier::ObjectDirectorySynced {
                set_test_volume_space(100 * GIB, 19 * GIB, 4096);
            }
            Ok(())
        })
    })
    .unwrap_err();
    assert!(matches!(error, ManagedSessionCacheError::FreeSpaceFloor));
    assert_eq!(store.restore(key).unwrap(), None);
    assert_eq!(
        fs::read_dir(state_root.join("cache/sessions/pending"))
            .unwrap()
            .count(),
        0
    );
    assert_eq!(
        fs::read_dir(state_root.join("cache/sessions/catalogs"))
            .unwrap()
            .count(),
        0
    );
    let object_files: usize = fs::read_dir(state_root.join("cache/sessions/objects"))
        .unwrap()
        .map(|entry| fs::read_dir(entry.unwrap().path()).unwrap().count())
        .sum();
    assert_eq!(object_files, 0);
    with_room(|| store.commit(key, b"checkpoint")).unwrap();
}
