#[test]
fn managed_session_store_storage_full_equivalence_classes_recover_exactly() {
    for fault in [
        TestIoFault::ObjectWrite,
        TestIoFault::ObjectFullSync,
        TestIoFault::ObjectFinalFullSync,
        TestIoFault::ObjectDirectorySync,
        TestIoFault::ObjectPendingDirectorySync,
        TestIoFault::CatalogWrite,
        TestIoFault::CatalogFullSync,
        TestIoFault::CatalogDirectorySync,
        TestIoFault::CatalogFinalFullSync,
        TestIoFault::PendingDirectorySync,
        TestIoFault::EndpointLockFullSync,
    ] {
        let temp = TempDir::new().unwrap();
        let state_root = root(&temp, &format!("storage-full-{fault:?}"));
        configure(&state_root, "64MiB");
        let key = ManagedCheckpointKey::from_canonical_receipt(format!("{fault:?}").as_bytes());
        let store = open_store(&state_root).unwrap();
        let error = with_room(|| with_test_io_fault(fault, || store.commit(key, b"checkpoint")))
            .unwrap_err();
        if matches!(
            fault,
            TestIoFault::CatalogDirectorySync
                | TestIoFault::CatalogFinalFullSync
                | TestIoFault::PendingDirectorySync
                | TestIoFault::EndpointLockFullSync
        ) {
            assert!(matches!(
                error,
                ManagedSessionCacheError::CommittedDurabilityUnknown(_)
            ));
            assert!(matches!(
                store.restore(key),
                Err(ManagedSessionCacheError::RecoveryRequired)
            ));
        } else {
            assert!(matches!(error, ManagedSessionCacheError::StorageFull));
            assert_eq!(store.restore(key).unwrap(), None);
            assert_eq!(
                fs::read_dir(state_root.join("cache/sessions/pending"))
                    .unwrap()
                    .count(),
                0,
                "fault {fault:?}"
            );
            let object_files: usize = fs::read_dir(state_root.join("cache/sessions/objects"))
                .unwrap()
                .map(|entry| fs::read_dir(entry.unwrap().path()).unwrap().count())
                .sum();
            assert_eq!(object_files, 0, "fault {fault:?}");
        }
        drop(store);
        let reopened = open_store(&state_root).unwrap();
        let observed = reopened.restore(key).unwrap();
        if matches!(
            fault,
            TestIoFault::CatalogDirectorySync
                | TestIoFault::CatalogFinalFullSync
                | TestIoFault::PendingDirectorySync
                | TestIoFault::EndpointLockFullSync
        ) {
            assert!(observed.is_none() || observed == Some(b"checkpoint".to_vec()));
        } else {
            assert_eq!(observed, None);
        }
        with_room(|| reopened.commit(key, b"checkpoint")).unwrap();
        assert_eq!(reopened.restore(key).unwrap(), Some(b"checkpoint".to_vec()));
    }

    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "storage-full-deletion-directory-sync");
    configure(&state_root, "8MiB");
    let first = ManagedCheckpointKey::from_canonical_receipt(b"storage-delete-first");
    let second = ManagedCheckpointKey::from_canonical_receipt(b"storage-delete-second");
    let payload = vec![0x71; 4 * 1024 * 1024];
    let store = open_store(&state_root).unwrap();
    with_room(|| store.commit(first, &payload)).unwrap();
    let error = with_room(|| {
        with_test_io_fault(TestIoFault::DeletionDirectorySync, || {
            store.commit(second, &payload)
        })
    })
    .unwrap_err();
    assert!(matches!(
        error,
        ManagedSessionCacheError::CommittedDurabilityUnknown(_)
    ));
    drop(store);
    let reopened = open_store(&state_root).unwrap();
    assert_eq!(reopened.restore(first).unwrap(), None);
    assert_eq!(reopened.restore(second).unwrap(), None);
    with_room(|| reopened.commit(second, &payload)).unwrap();
    assert_eq!(reopened.restore(second).unwrap(), Some(payload));

    let temp = TempDir::new().unwrap();
    let state_root = root(&temp, "storage-full-catalog-history-sync");
    configure(&state_root, "64MiB");
    let store = open_store(&state_root).unwrap();
    for value in 0..4 {
        let key = ManagedCheckpointKey::from_canonical_receipt(
            format!("storage-history-{value}").as_bytes(),
        );
        with_room(|| store.commit(key, b"checkpoint")).unwrap();
    }
    let fifth = ManagedCheckpointKey::from_canonical_receipt(b"storage-history-4");
    let error = with_room(|| {
        with_test_io_fault(TestIoFault::CatalogHistoryDirectorySync, || {
            store.commit(fifth, b"checkpoint")
        })
    })
    .unwrap_err();
    assert!(matches!(
        error,
        ManagedSessionCacheError::CommittedDurabilityUnknown(_)
    ));
    drop(store);
    let reopened = open_store(&state_root).unwrap();
    assert_eq!(reopened.restore(fifth).unwrap(), None);
    with_room(|| reopened.commit(fifth, b"checkpoint")).unwrap();
    assert_eq!(
        reopened.restore(fifth).unwrap(),
        Some(b"checkpoint".to_vec())
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires HF2Q_MANAGED_CACHE_APFS_ENOSPC_ROOT on a disposable constrained APFS volume"]
fn managed_session_store_real_apfs_enospc_cleans_and_retries() {
    use std::io::Write;

    let state_root = PathBuf::from(
        std::env::var_os("HF2Q_MANAGED_CACHE_APFS_ENOSPC_ROOT")
            .expect("protected APFS ENOSPC root must be supplied"),
    );
    assert!(!state_root.exists());
    configure(&state_root, "64MiB");
    let store = with_room(|| open_store(&state_root)).unwrap();
    let key = ManagedCheckpointKey::from_canonical_receipt(b"real-apfs-enospc");
    let volume_root = state_root.parent().unwrap();
    let reserve_path = volume_root.join("hf2q-managed-cache-enospc-reserve");
    let filler_path = volume_root.join("hf2q-managed-cache-enospc-filler");
    let mut reserve = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&reserve_path)
        .unwrap();
    let mut filler = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&filler_path)
        .unwrap();
    let mut chunk = vec![0u8; 1024 * 1024];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..16 {
        for byte in &mut chunk {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state as u8;
        }
        reserve.write_all(&chunk).unwrap();
    }
    reserve.sync_all().unwrap();
    assert!(reserve.metadata().unwrap().blocks().saturating_mul(512) >= 16 * 1024 * 1024);
    loop {
        for byte in &mut chunk {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state as u8;
        }
        match filler.write_all(&chunk).and_then(|()| filler.sync_all()) {
            Ok(()) => {}
            Err(error)
                if matches!(
                    error.raw_os_error(),
                    Some(libc::ENOSPC) | Some(libc::EDQUOT)
                ) =>
            {
                break;
            }
            Err(error) => panic!("unexpected APFS filler error: {error}"),
        }
    }
    let filled = filler.metadata().unwrap().len();
    assert!(filled > 4 * 1024 * 1024);
    assert_eq!(
        filler.metadata().unwrap().dev(),
        store.directories.pending.device()
    );
    assert!(filler.metadata().unwrap().blocks().saturating_mul(512) > 4 * 1024 * 1024);
    drop(reserve);
    fs::remove_file(&reserve_path).unwrap();
    fs::File::open(volume_root).unwrap().sync_all().unwrap();

    let stat = super::unix::volume_space(&store.directories.pending).unwrap();
    let available_before_commit = stat.f_bavail.saturating_mul(stat.f_frsize);
    let mut payload = vec![0u8; 32 * 1024 * 1024];
    for byte in &mut payload {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte = state as u8;
    }
    assert!(available_before_commit > 4 * 1024 * 1024);
    assert!(available_before_commit < payload.len() as u64);
    reset_test_object_write_failure_len();
    let commit = with_room(|| store.commit(key, &payload));
    assert!(
        matches!(commit, Err(ManagedSessionCacheError::StorageFull)),
        "real APFS exhaustion returned {commit:?} with {available_before_commit} bytes reported available"
    );
    let partial_len = take_test_object_write_failure_len()
        .expect("real APFS exhaustion must interrupt the object write itself");
    assert!(
        partial_len > super::OBJECT_HEADER_BYTES as u64,
        "real APFS write stopped before its payload prefix at {partial_len} bytes"
    );
    assert!(
        partial_len < super::OBJECT_HEADER_BYTES as u64 + payload.len() as u64,
        "real APFS write unexpectedly completed all {partial_len} object bytes"
    );
    assert_eq!(store.restore(key).unwrap(), None);
    assert_eq!(
        fs::read_dir(state_root.join("cache/sessions/pending"))
            .unwrap()
            .count(),
        0
    );
    drop(store);
    drop(filler);
    fs::remove_file(&filler_path).unwrap();

    let reopened = with_room(|| open_store(&state_root)).unwrap();
    assert_eq!(reopened.restore(key).unwrap(), None);
    with_room(|| reopened.commit(key, &payload)).unwrap();
    assert_eq!(reopened.restore(key).unwrap(), Some(payload));
}
