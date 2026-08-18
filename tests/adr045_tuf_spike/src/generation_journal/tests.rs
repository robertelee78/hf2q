use std::fs;
use std::os::unix::fs::symlink;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use jiff::Timestamp;
use tempfile::TempDir;

use super::{
    commit_candidate, read_selected_sequence, Barrier, CommitOutcome, FaultAction, FaultPlan,
    JournalError, LockedJournal, ALL_BARRIERS,
};
use crate::candidates::VerifiedMetadataEvidence;
use crate::model::{sha256, CapturedRole};

struct PrivateRoot {
    _temp: TempDir,
    path: PathBuf,
}

impl PrivateRoot {
    fn path(&self) -> &Path {
        &self.path
    }
}

fn private_root() -> PrivateRoot {
    let temp = TempDir::new().expect("temporary journal root");
    fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o700))
        .expect("private journal root mode");
    let path = temp
        .path()
        .canonicalize()
        .expect("canonical temporary root");
    PrivateRoot { _temp: temp, path }
}

fn role(name: &str, version: u64, bytes: &[u8]) -> CapturedRole {
    CapturedRole {
        request_name: name.to_string(),
        version,
        raw: bytes.to_vec(),
        raw_sha256: sha256(bytes),
    }
}

fn candidate(version: u64) -> VerifiedMetadataEvidence {
    candidate_at(version, "2026-08-17T20:00:00Z", None)
}

fn candidate_at(
    version: u64,
    update_start: &str,
    targets_override: Option<&[u8]>,
) -> VerifiedMetadataEvidence {
    let root = role("1.root.json", 1, b"root-v1");
    VerifiedMetadataEvidence::test_only(
        "https://updates.invalid/stable/metadata/",
        "stable",
        update_start.parse::<Timestamp>().expect("fixed timestamp"),
        root.clone(),
        Vec::new(),
        [
            root,
            role(
                "timestamp.json",
                version,
                format!("timestamp-v{version}").as_bytes(),
            ),
            role(
                "snapshot.json",
                version,
                format!("snapshot-v{version}").as_bytes(),
            ),
            role(
                "targets.json",
                version,
                targets_override.unwrap_or(match version {
                    1 => b"targets-v1",
                    2 => b"targets-v2",
                    3 => b"targets-v3",
                    _ => b"targets-vN",
                }),
            ),
        ],
    )
}

fn rotated_candidate() -> VerifiedMetadataEvidence {
    let prior = role("1.root.json", 1, b"root-v1");
    let root = role("2.root.json", 2, b"root-v2-dual-signed");
    VerifiedMetadataEvidence::test_only(
        "https://updates.invalid/stable/metadata/",
        "stable",
        "2026-08-17T20:00:00Z"
            .parse::<Timestamp>()
            .expect("fixed timestamp"),
        prior,
        vec![root.clone()],
        [
            root,
            role("timestamp.json", 2, b"timestamp-v2"),
            role("snapshot.json", 2, b"snapshot-v2"),
            role("targets.json", 2, b"targets-v2"),
        ],
    )
}

fn second_rotated_candidate() -> VerifiedMetadataEvidence {
    let prior = role("2.root.json", 2, b"root-v2-dual-signed");
    let root = role("3.root.json", 3, b"root-v3-dual-signed");
    VerifiedMetadataEvidence::test_only(
        "https://updates.invalid/stable/metadata/",
        "stable",
        "2026-08-17T20:01:00Z"
            .parse::<Timestamp>()
            .expect("fixed timestamp"),
        prior,
        vec![root.clone()],
        [
            root,
            role("timestamp.json", 3, b"timestamp-v3"),
            role("snapshot.json", 3, b"snapshot-v3"),
            role("targets.json", 3, b"targets-v3"),
        ],
    )
}

#[test]
fn immutable_generations_advance_one_selector() {
    let root = private_root();
    assert_eq!(read_selected_sequence(root.path()).unwrap(), None);
    assert_eq!(
        commit_candidate(root.path(), &candidate(1), FaultPlan::default()).unwrap(),
        CommitOutcome::Committed { sequence: 1 }
    );
    assert_eq!(read_selected_sequence(root.path()).unwrap(), Some(1));
    assert_eq!(
        commit_candidate(root.path(), &candidate(1), FaultPlan::default()).unwrap(),
        CommitOutcome::AlreadyCommitted { sequence: 1 }
    );
    assert_eq!(
        commit_candidate(root.path(), &candidate(2), FaultPlan::default()).unwrap(),
        CommitOutcome::Committed { sequence: 2 }
    );
    assert_eq!(read_selected_sequence(root.path()).unwrap(), Some(2));
}

#[test]
fn first_durable_generation_can_bind_a_verified_root_rotation() {
    let root = private_root();
    assert_eq!(
        commit_candidate(root.path(), &rotated_candidate(), FaultPlan::default()).unwrap(),
        CommitOutcome::Committed { sequence: 1 }
    );
    assert_eq!(read_selected_sequence(root.path()).unwrap(), Some(1));
    assert_eq!(
        commit_candidate(root.path(), &rotated_candidate(), FaultPlan::default()).unwrap(),
        CommitOutcome::AlreadyCommitted { sequence: 1 }
    );
    assert_eq!(
        commit_candidate(
            root.path(),
            &second_rotated_candidate(),
            FaultPlan::default(),
        )
        .unwrap(),
        CommitOutcome::Committed { sequence: 2 }
    );
    assert_eq!(read_selected_sequence(root.path()).unwrap(), Some(2));
}

#[test]
fn clock_and_role_floors_never_decrease_or_equivocate() {
    let root = private_root();
    commit_candidate(root.path(), &candidate(2), FaultPlan::default()).unwrap();

    assert!(matches!(
        commit_candidate(root.path(), &candidate(1), FaultPlan::default()),
        Err(JournalError::Invalid(_))
    ));
    assert_eq!(read_selected_sequence(root.path()).unwrap(), Some(1));

    let changed = candidate_at(2, "2026-08-17T20:00:00Z", Some(b"changed-envelope"));
    assert!(matches!(
        commit_candidate(root.path(), &changed, FaultPlan::default()),
        Err(JournalError::Invalid(_))
    ));

    let older_clock = candidate_at(3, "2026-08-17T19:59:59Z", None);
    assert!(matches!(
        commit_candidate(root.path(), &older_clock, FaultPlan::default()),
        Err(JournalError::Invalid(_))
    ));
    assert_eq!(read_selected_sequence(root.path()).unwrap(), Some(1));
}

#[test]
fn corrupt_selected_state_fails_without_rollback_fallback() {
    let root = private_root();
    commit_candidate(root.path(), &candidate(1), FaultPlan::default()).unwrap();
    commit_candidate(root.path(), &candidate(2), FaultPlan::default()).unwrap();
    let current = root.path().join("update/metadata/current.json");
    fs::remove_file(&current).unwrap();
    symlink("generations/00000000000000000001/generation.json", &current).unwrap();
    assert!(matches!(
        read_selected_sequence(root.path()),
        Err(JournalError::Invalid(_))
    ));
}

#[test]
fn selected_history_rejects_missing_predecessors_and_future_cruft() {
    let root = private_root();
    commit_candidate(root.path(), &candidate(1), FaultPlan::default()).unwrap();
    commit_candidate(root.path(), &candidate(2), FaultPlan::default()).unwrap();
    fs::rename(
        root.path()
            .join("update/metadata/generations/00000000000000000001"),
        root.path().join("removed-generation"),
    )
    .unwrap();
    assert!(read_selected_sequence(root.path()).is_err());

    let root = private_root();
    commit_candidate(root.path(), &candidate(1), FaultPlan::default()).unwrap();
    let future = root
        .path()
        .join("update/metadata/generations/00000000000000000003");
    fs::create_dir(&future).unwrap();
    fs::set_permissions(&future, fs::Permissions::from_mode(0o700)).unwrap();
    assert!(matches!(
        read_selected_sequence(root.path()),
        Err(JournalError::Invalid(_))
    ));
}

#[test]
fn selected_history_rejects_unbounded_sequence_before_range_construction() {
    let root = private_root();
    let metadata = root.path().join("update/metadata");
    let generations = metadata.join("generations");
    fs::create_dir_all(&generations).unwrap();
    for directory in [root.path().join("update"), metadata.clone(), generations] {
        fs::set_permissions(directory, fs::Permissions::from_mode(0o700)).unwrap();
    }
    let selector = format!(
        "{{\"schema_version\":0,\"sequence\":{},\"generation_sha256\":\"{}\"}}\n",
        super::schema::MAX_RETAINED_GENERATIONS + 1,
        "00".repeat(32)
    );
    let current = metadata.join("current.json");
    fs::write(&current, selector).unwrap();
    fs::set_permissions(current, fs::Permissions::from_mode(0o600)).unwrap();

    assert!(read_selected_sequence(root.path()).is_err());
}

#[test]
fn selected_receipt_rejects_duplicate_root_descriptors() {
    let root = private_root();
    commit_candidate(root.path(), &rotated_candidate(), FaultPlan::default()).unwrap();
    let generation = root
        .path()
        .join("update/metadata/generations/00000000000000000001");
    let receipt_path = generation.join("generation.json");
    let receipt = fs::read_to_string(&receipt_path).unwrap();
    let start = receipt.find("\"root_chain\":[").unwrap() + "\"root_chain\":[".len();
    let end = receipt[start..].find("],\"root\"").unwrap() + start;
    let descriptor = &receipt[start..end];
    let hostile = format!("{},{}", &receipt[..end], descriptor) + &receipt[end..];
    fs::write(&receipt_path, hostile.as_bytes()).unwrap();
    fs::set_permissions(&receipt_path, fs::Permissions::from_mode(0o600)).unwrap();

    let current_path = root.path().join("update/metadata/current.json");
    let current = fs::read_to_string(&current_path).unwrap();
    let old_digest = hex::encode(sha256(receipt.as_bytes()));
    let new_digest = hex::encode(sha256(hostile.as_bytes()));
    let current = current.replace(&old_digest, &new_digest);
    fs::write(&current_path, current).unwrap();
    fs::set_permissions(&current_path, fs::Permissions::from_mode(0o600)).unwrap();
    assert!(matches!(
        read_selected_sequence(root.path()),
        Err(JournalError::Invalid(_))
    ));
}

#[test]
fn selected_generation_rejects_extra_entries_and_hardlinks() {
    let root = private_root();
    commit_candidate(root.path(), &candidate(1), FaultPlan::default()).unwrap();
    let generation = root
        .path()
        .join("update/metadata/generations/00000000000000000001");
    fs::write(generation.join("unexpected"), b"hostile").unwrap();
    fs::set_permissions(
        generation.join("unexpected"),
        fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    assert!(matches!(
        read_selected_sequence(root.path()),
        Err(JournalError::Invalid(_))
    ));

    let root = private_root();
    commit_candidate(root.path(), &candidate(1), FaultPlan::default()).unwrap();
    let generation = root
        .path()
        .join("update/metadata/generations/00000000000000000001");
    let outside_hardlink = root.path().join("outside-hardlink");
    fs::hard_link(generation.join("timestamp.json"), outside_hardlink).unwrap();
    assert!(matches!(
        read_selected_sequence(root.path()),
        Err(JournalError::Invalid(_))
    ));
}

#[test]
fn live_namespace_reopen_rejects_update_metadata_and_generation_swaps() {
    for component in ["update", "metadata", "generations"] {
        let root = private_root();
        let journal = LockedJournal::open(root.path()).expect("journal opens");
        let path = match component {
            "update" => root.path().join("update"),
            "metadata" => root.path().join("update/metadata"),
            "generations" => root.path().join("update/metadata/generations"),
            _ => unreachable!(),
        };
        let detached = path.with_extension("detached");
        fs::rename(&path, &detached).expect("detach verified directory");
        fs::create_dir(&path).expect("replace live directory");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).expect("replacement mode");
        assert!(matches!(
            journal.reopen_live_namespace(),
            Err(JournalError::Invalid(_) | JournalError::Missing)
        ));
    }
}

#[test]
fn staged_selector_identity_rejects_same_byte_name_replacement() {
    let root = private_root();
    let journal = LockedJournal::open(root.path()).expect("journal opens");
    let name = ".current-00000000000000000001.json";
    let bytes = b"selector bytes\n";
    let staged = super::file::write_or_resume_private_file(&journal.metadata, name, bytes)
        .expect("stage selector");
    let identity = super::unix::regular_file_identity(&staged, journal.metadata.device())
        .expect("selector identity");
    fs::remove_file(root.path().join("update/metadata").join(name)).expect("remove staged name");
    fs::write(root.path().join("update/metadata").join(name), bytes).expect("replace staged name");
    fs::set_permissions(
        root.path().join("update/metadata").join(name),
        fs::Permissions::from_mode(0o600),
    )
    .expect("replacement mode");
    assert!(matches!(
        super::unix::verify_named_identity(&journal.metadata, name, identity),
        Err(JournalError::Invalid(_))
    ));
}

#[test]
fn exact_file_prefixes_resume_without_truncation_or_transaction_cruft() {
    let root = private_root();
    let pending = root
        .path()
        .join("update/metadata/generations/.pending-00000000000000000001");
    fs::create_dir_all(pending.join("root-chain")).unwrap();
    fs::set_permissions(
        root.path().join("update"),
        fs::Permissions::from_mode(0o700),
    )
    .unwrap();
    fs::set_permissions(
        root.path().join("update/metadata"),
        fs::Permissions::from_mode(0o700),
    )
    .unwrap();
    fs::set_permissions(
        root.path().join("update/metadata/generations"),
        fs::Permissions::from_mode(0o700),
    )
    .unwrap();
    fs::set_permissions(&pending, fs::Permissions::from_mode(0o700)).unwrap();
    fs::set_permissions(
        pending.join("root-chain"),
        fs::Permissions::from_mode(0o700),
    )
    .unwrap();
    fs::write(pending.join("trusted-root-before.json"), b"roo").unwrap();
    fs::set_permissions(
        pending.join("trusted-root-before.json"),
        fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    let partial_root = pending
        .join("root-chain")
        .join("00000000000000000002.root.json");
    fs::write(&partial_root, b"root-v2").unwrap();
    fs::set_permissions(&partial_root, fs::Permissions::from_mode(0o600)).unwrap();
    assert_eq!(
        commit_candidate(root.path(), &rotated_candidate(), FaultPlan::default()).unwrap(),
        CommitOutcome::Committed { sequence: 1 }
    );
    assert_eq!(read_selected_sequence(root.path()).unwrap(), Some(1));

    let root = private_root();
    assert!(matches!(
        commit_candidate(
            root.path(),
            &candidate(1),
            FaultPlan {
                barrier: Some(Barrier::GenerationsSync),
                action: Some(FaultAction::ReturnError),
            },
        ),
        Err(JournalError::Injected(Barrier::GenerationsSync))
    ));
    let receipt = fs::read(
        root.path()
            .join("update/metadata/generations/00000000000000000001/generation.json"),
    )
    .unwrap();
    let selector = format!(
        "{{\"schema_version\":0,\"sequence\":1,\"generation_sha256\":\"{}\"}}\n",
        hex::encode(sha256(&receipt))
    );
    let pending_selector = root
        .path()
        .join("update/metadata/.current-00000000000000000001.json");
    fs::write(
        &pending_selector,
        &selector.as_bytes()[..selector.len() / 2],
    )
    .unwrap();
    fs::set_permissions(&pending_selector, fs::Permissions::from_mode(0o600)).unwrap();
    assert_eq!(
        commit_candidate(root.path(), &candidate(1), FaultPlan::default()).unwrap(),
        CommitOutcome::Committed { sequence: 1 }
    );
    assert!(!pending_selector.exists());
    assert_eq!(read_selected_sequence(root.path()).unwrap(), Some(1));
}

#[test]
fn every_injected_barrier_recovers_to_complete_generation() {
    for barrier in ALL_BARRIERS {
        let root = private_root();
        let first = commit_candidate(
            root.path(),
            &candidate(1),
            FaultPlan {
                barrier: Some(*barrier),
                action: Some(FaultAction::ReturnError),
            },
        );
        assert!(first.is_err(), "{barrier:?} must inject a failure");
        let selected = read_selected_sequence(root.path());
        if barrier_leaves_ambiguous_published_generation(*barrier) {
            assert!(selected.is_err(), "{barrier:?}");
        } else if barrier_precedes_commit(*barrier) {
            assert_eq!(selected.unwrap(), None, "{barrier:?}");
        } else {
            assert_eq!(selected.unwrap(), Some(1), "{barrier:?}");
            assert!(matches!(
                first,
                Err(JournalError::CommittedDurabilityUnknown { sequence: 1, .. })
            ));
        }
        let retry = commit_candidate(root.path(), &candidate(1), FaultPlan::default())
            .unwrap_or_else(|error| panic!("retry after {barrier:?} failed: {error}"));
        assert!(matches!(
            retry,
            CommitOutcome::Committed { sequence: 1 }
                | CommitOutcome::AlreadyCommitted { sequence: 1 }
        ));
        assert_eq!(read_selected_sequence(root.path()).unwrap(), Some(1));
    }
}

#[test]
fn every_update_barrier_preserves_old_or_complete_new_generation() {
    for barrier in ALL_BARRIERS {
        let root = private_root();
        commit_candidate(root.path(), &candidate(1), FaultPlan::default()).unwrap();
        let update = commit_candidate(
            root.path(),
            &candidate(2),
            FaultPlan {
                barrier: Some(*barrier),
                action: Some(FaultAction::ReturnError),
            },
        );
        assert!(update.is_err(), "{barrier:?} must inject a failure");
        let selected = read_selected_sequence(root.path());
        if barrier_leaves_ambiguous_published_generation(*barrier) {
            assert!(selected.is_err(), "{barrier:?}");
        } else if barrier_precedes_commit(*barrier) {
            assert_eq!(selected.unwrap(), Some(1), "{barrier:?}");
        } else {
            assert_eq!(selected.unwrap(), Some(2), "{barrier:?}");
            assert!(matches!(
                update,
                Err(JournalError::CommittedDurabilityUnknown { sequence: 2, .. })
            ));
        }
        let retry = commit_candidate(root.path(), &candidate(2), FaultPlan::default())
            .unwrap_or_else(|error| panic!("update retry after {barrier:?} failed: {error}"));
        assert!(matches!(
            retry,
            CommitOutcome::Committed { sequence: 2 }
                | CommitOutcome::AlreadyCommitted { sequence: 2 }
        ));
        assert_eq!(read_selected_sequence(root.path()).unwrap(), Some(2));
    }
}

fn barrier_precedes_commit(barrier: Barrier) -> bool {
    matches!(
        barrier,
        Barrier::PendingDirectory
            | Barrier::RootChainFiles
            | Barrier::RootChainSync
            | Barrier::TrustedRootBeforeFile
            | Barrier::TrustedRootFile
            | Barrier::TimestampFile
            | Barrier::SnapshotFile
            | Barrier::TargetsFile
            | Barrier::ReceiptFile
            | Barrier::PendingDirectorySync
            | Barrier::GenerationPublish
            | Barrier::GenerationsSync
            | Barrier::SelectorFile
            | Barrier::MetadataPrecommitSync
    )
}

fn barrier_leaves_ambiguous_published_generation(barrier: Barrier) -> bool {
    matches!(
        barrier,
        Barrier::GenerationPublish | Barrier::GenerationsSync
    )
}

#[test]
fn replayed_old_selector_is_not_update_authority_and_exact_retry_repairs_it() {
    let root = private_root();
    commit_candidate(root.path(), &candidate(1), FaultPlan::default()).unwrap();
    let selector_path = root.path().join("update/metadata/current.json");
    let old_selector = fs::read(&selector_path).unwrap();
    commit_candidate(root.path(), &candidate(2), FaultPlan::default()).unwrap();

    fs::write(&selector_path, old_selector).unwrap();
    assert!(read_selected_sequence(root.path()).is_err());

    assert_eq!(
        commit_candidate(root.path(), &candidate(2), FaultPlan::default()).unwrap(),
        CommitOutcome::Committed { sequence: 2 }
    );
    assert_eq!(read_selected_sequence(root.path()).unwrap(), Some(2));
}

#[test]
fn cross_process_lock_is_nonblocking_and_released_on_kill() {
    let root = private_root();
    let ready = root.path().join("worker-ready");
    let mut child = spawn_helper(
        "generation_journal::tests::lock_worker_process",
        &[
            ("HF2Q_SPIKE_LOCK_ROOT", root.path()),
            ("HF2Q_SPIKE_LOCK_READY", &ready),
        ],
    );
    wait_for(&ready);
    assert!(matches!(
        commit_candidate(root.path(), &candidate(1), FaultPlan::default()),
        Err(JournalError::Busy)
    ));
    child.kill().expect("kill lock worker");
    child.wait().expect("reap lock worker");
    assert!(commit_candidate(root.path(), &candidate(1), FaultPlan::default()).is_ok());
}

#[test]
fn process_abort_at_every_barrier_is_recoverable() {
    for barrier in ALL_BARRIERS {
        let root = private_root();
        let barrier_value = PathBuf::from(barrier.name());
        let mut child = spawn_helper(
            "generation_journal::tests::crash_worker_process",
            &[
                ("HF2Q_SPIKE_CRASH_ROOT", root.path()),
                ("HF2Q_SPIKE_CRASH_BARRIER", &barrier_value),
            ],
        );
        let status = child.wait().expect("wait for crash worker");
        assert!(!status.success(), "{barrier:?} worker must abort");
        let retry = commit_candidate(root.path(), &candidate(1), FaultPlan::default())
            .unwrap_or_else(|error| panic!("recovery after process abort at {barrier:?}: {error}"));
        assert!(matches!(
            retry,
            CommitOutcome::Committed { sequence: 1 }
                | CommitOutcome::AlreadyCommitted { sequence: 1 }
        ));
        assert_eq!(read_selected_sequence(root.path()).unwrap(), Some(1));
    }
}

#[test]
fn process_abort_during_update_preserves_old_or_complete_new_generation() {
    for barrier in ALL_BARRIERS {
        let root = private_root();
        commit_candidate(root.path(), &candidate(1), FaultPlan::default()).unwrap();
        let barrier_value = PathBuf::from(barrier.name());
        let version_value = PathBuf::from("2");
        let mut child = spawn_helper(
            "generation_journal::tests::crash_worker_process",
            &[
                ("HF2Q_SPIKE_CRASH_ROOT", root.path()),
                ("HF2Q_SPIKE_CRASH_BARRIER", &barrier_value),
                ("HF2Q_SPIKE_CRASH_VERSION", &version_value),
            ],
        );
        let status = child.wait().expect("wait for update crash worker");
        assert!(!status.success(), "{barrier:?} worker must abort");
        let selected = read_selected_sequence(root.path());
        if barrier_leaves_ambiguous_published_generation(*barrier) {
            assert!(selected.is_err(), "{barrier:?}");
        } else {
            assert_eq!(
                selected.unwrap(),
                Some(if barrier_precedes_commit(*barrier) {
                    1
                } else {
                    2
                }),
                "{barrier:?}"
            );
        }
        let retry = commit_candidate(root.path(), &candidate(2), FaultPlan::default())
            .unwrap_or_else(|error| panic!("update recovery after {barrier:?}: {error}"));
        assert!(matches!(
            retry,
            CommitOutcome::Committed { sequence: 2 }
                | CommitOutcome::AlreadyCommitted { sequence: 2 }
        ));
    }
}

#[test]
fn process_abort_with_real_root_chain_files_is_recoverable() {
    let root = private_root();
    let barrier = PathBuf::from(Barrier::RootChainFiles.name());
    let first_rotation = PathBuf::from("first");
    let mut child = spawn_helper(
        "generation_journal::tests::crash_worker_process",
        &[
            ("HF2Q_SPIKE_CRASH_ROOT", root.path()),
            ("HF2Q_SPIKE_CRASH_BARRIER", &barrier),
            ("HF2Q_SPIKE_CRASH_ROTATION", &first_rotation),
        ],
    );
    assert!(!child
        .wait()
        .expect("wait for first rotation crash")
        .success());
    assert_eq!(
        commit_candidate(root.path(), &rotated_candidate(), FaultPlan::default()).unwrap(),
        CommitOutcome::Committed { sequence: 1 }
    );

    let second_rotation = PathBuf::from("second");
    let mut child = spawn_helper(
        "generation_journal::tests::crash_worker_process",
        &[
            ("HF2Q_SPIKE_CRASH_ROOT", root.path()),
            ("HF2Q_SPIKE_CRASH_BARRIER", &barrier),
            ("HF2Q_SPIKE_CRASH_ROTATION", &second_rotation),
        ],
    );
    assert!(!child
        .wait()
        .expect("wait for second rotation crash")
        .success());
    assert_eq!(
        commit_candidate(
            root.path(),
            &second_rotated_candidate(),
            FaultPlan::default(),
        )
        .unwrap(),
        CommitOutcome::Committed { sequence: 2 }
    );
    assert_eq!(read_selected_sequence(root.path()).unwrap(), Some(2));
}

#[test]
fn lock_worker_process() {
    let Ok(root) = std::env::var("HF2Q_SPIKE_LOCK_ROOT") else {
        return;
    };
    let ready = std::env::var("HF2Q_SPIKE_LOCK_READY").expect("ready path");
    let _journal = LockedJournal::open(Path::new(&root)).expect("worker takes lock");
    fs::write(ready, b"ready").expect("signal lock acquisition");
    thread::sleep(Duration::from_secs(60));
}

#[test]
fn crash_worker_process() {
    let Ok(root) = std::env::var("HF2Q_SPIKE_CRASH_ROOT") else {
        return;
    };
    let barrier = std::env::var("HF2Q_SPIKE_CRASH_BARRIER").expect("barrier name");
    let barrier = Barrier::parse(&barrier).expect("known barrier");
    let version = std::env::var("HF2Q_SPIKE_CRASH_VERSION")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1);
    let evidence = match std::env::var("HF2Q_SPIKE_CRASH_ROTATION").as_deref() {
        Ok("first") => rotated_candidate(),
        Ok("second") => second_rotated_candidate(),
        _ => candidate(version),
    };
    let _ = commit_candidate(
        Path::new(&root),
        &evidence,
        FaultPlan {
            barrier: Some(barrier),
            action: Some(FaultAction::AbortProcess),
        },
    );
    panic!("fault barrier was not reached");
}

fn spawn_helper(test_name: &str, envs: &[(&str, &Path)]) -> std::process::Child {
    let mut command = Command::new(std::env::current_exe().expect("test executable"));
    command
        .arg("--exact")
        .arg(test_name)
        .arg("--nocapture")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    for (name, value) in envs {
        command.env(name, value);
    }
    command.spawn().expect("spawn helper test process")
}

fn wait_for(path: &Path) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !path.exists() {
        assert!(Instant::now() < deadline, "helper did not become ready");
        thread::sleep(Duration::from_millis(10));
    }
}
