use std::fs;
use std::io::Cursor;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};

use tempfile::TempDir;

use super::super::host::{HostObservation, HostProbe};
use super::super::runtime_policy::{
    authorize_session_cache_policy, SessionCachePolicyAuthorization,
};
use super::super::schema::{ConfiguredShell, HardwareProfileV1};
use super::super::{execute, SetupError};
use super::catalog::{sha256_hex, CatalogEntryV1, CatalogV1};
use super::transaction::{commit_with_test_hook, with_test_volume_space};
use super::{
    ManagedCacheBarrier, ManagedCheckpointKey, ManagedCommitOutcome, ManagedSessionCache,
    ManagedSessionCacheError, CATALOG_PARTIAL, OBJECT_PARTIAL,
};
use crate::cli::{SessionCacheChoice, SetupArgs};

const GIB: u64 = 1024 * 1024 * 1024;

pub(super) fn abort_at_managed_cache_barrier(barrier: ManagedCacheBarrier) {
    if std::env::var("HF2Q_MANAGED_CACHE_ABORT_AT").as_deref() == Ok(barrier.as_str()) {
        unsafe { libc::raise(libc::SIGABRT) };
        std::process::abort();
    }
}

struct FakeProbe;

impl HostProbe for FakeProbe {
    fn observe(&self, _state_root: &Path) -> Result<HostObservation, SetupError> {
        Ok(HostObservation {
            hardware: HardwareProfileV1 {
                target: "aarch64-apple-darwin".to_owned(),
                chip_model: "Apple M5 Max".to_owned(),
                unified_memory_bytes: 128 * GIB,
                metal_device_name: "Apple M5 Max".to_owned(),
                metal_recommended_working_set_bytes: 96 * GIB,
            },
            macos_version: "15.6.1".to_owned(),
            configured_shell: ConfiguredShell::Zsh,
            performance_level0_name: "Super".to_owned(),
            performance_level0_cores: 4,
            performance_level1_name: "Performance".to_owned(),
            performance_level1_cores: 12,
            open_file_soft_limit: 10240,
            volume_total_bytes: 500 * GIB,
            volume_available_bytes: 200 * GIB,
        })
    }
}

fn root(temp: &TempDir, name: &str) -> PathBuf {
    temp.path().canonicalize().unwrap().join(name)
}

fn configure(root: &Path, limit: &str) {
    let args = SetupArgs {
        session_cache: Some(SessionCacheChoice::On),
        session_cache_limit: Some(limit.to_owned()),
        state_root: Some(root.to_owned()),
    };
    execute(
        args,
        &FakeProbe,
        false,
        &mut Cursor::new(Vec::<u8>::new()),
        &mut Vec::new(),
    )
    .unwrap();
}

fn open_store(root: &Path) -> Result<ManagedSessionCache, ManagedSessionCacheError> {
    let authorization = authorize_session_cache_policy(root).unwrap();
    let SessionCachePolicyAuthorization::Enabled(authorization) = authorization else {
        panic!("positive setup policy must mint enabled authority");
    };
    authorization.into_managed_store()
}

fn with_room<T>(action: impl FnOnce() -> T) -> T {
    with_test_volume_space(500 * GIB, 200 * GIB, 4096, action)
}

#[test]
fn managed_session_catalog_v1_golden_is_canonical_and_pinned() {
    let catalog = CatalogV1::empty()
        .with_entry(CatalogEntryV1::new("0".repeat(64), "1".repeat(64), 84))
        .unwrap();
    let expected = concat!(
        "{\"kind\":\"hf2q.managed-session-cache.catalog\",",
        "\"schema_version\":1,\"generation\":1,\"entries\":[{",
        "\"key_sha256\":\"0000000000000000000000000000000000000000000000000000000000000000\",",
        "\"object_sha256\":\"1111111111111111111111111111111111111111111111111111111111111111\",",
        "\"object_bytes\":84,\"last_committed_generation\":1}]}\n"
    )
    .as_bytes();
    let bytes = catalog.to_canonical_bytes().unwrap();
    assert_eq!(bytes, expected);
    assert_eq!(
        sha256_hex(&bytes),
        "629d13bb3f66e8535f77b58e55c8d7e4a5e37d8c39736c01bde7df6fe37e9e44"
    );
    assert_eq!(CatalogV1::parse_exact(&bytes).unwrap(), catalog);
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
    assert!(open_store(&state_root).is_err());
    assert_eq!(fs::read(&object).unwrap(), bytes);
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
