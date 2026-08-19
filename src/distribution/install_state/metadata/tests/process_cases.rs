use super::*;
use std::os::unix::process::ExitStatusExt;

#[test]
fn process_abort_at_every_initial_and_successor_barrier_is_recoverable() {
    for (sequence, barriers) in [
        (1, initial_transaction_barriers()),
        (2, successor_transaction_barriers()),
    ] {
        for barrier in barriers {
            let parent = tempfile::tempdir().expect("tempdir");
            let root = test_root(&parent);
            if sequence == 2 {
                commit_candidate_for_test(
                    authorization(&root),
                    candidate_at(&root, "2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3),
                    FaultPlan::default(),
                )
                .expect("commit crash-test predecessor");
            }
            let mut child = spawn_crash_worker(&root, sequence, barrier);
            let status = child.wait().expect("wait for crash worker");
            assert_eq!(
                status.signal(),
                Some(libc::SIGABRT),
                "sequence {sequence} barrier {barrier:?} must abort with SIGABRT"
            );

            let (started, completed, root_version, timestamp_version) = if sequence == 1 {
                ("2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3)
            } else {
                ("2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 3, 4)
            };
            let mut retry_candidate =
                candidate_at(&root, started, completed, root_version, timestamp_version);
            if sequence == 2 {
                retry_candidate.set_timestamp_snapshot_floor_reset_for_test(Some(2));
            }
            let retry = commit_candidate_for_test(
                authorization(&root),
                retry_candidate,
                FaultPlan::default(),
            )
            .unwrap_or_else(|error| {
                panic!("recovery after sequence {sequence} barrier {barrier:?}: {error}")
            });
            assert!(matches!(
                retry,
                MetadataCommitOutcome::Committed { sequence: actual }
                    | MetadataCommitOutcome::AlreadyCommitted { sequence: actual }
                    if actual == sequence
            ));
            let selected = read_selected(&authorization(&root))
                .expect("read recovered selection")
                .expect("recovered generation");
            assert_eq!(selected.sequence, sequence);
            if sequence == 2 {
                assert_eq!(
                    MetadataGenerationReceiptV2::parse(selected.generation_receipt())
                        .expect("selected reset receipt parses")
                        .timestamp_snapshot_floor_reset_from_root_version(),
                    Some(2),
                    "successor crash recovery preserves the sealed reset evidence"
                );
            }
            assert_eq!(
                std::fs::read_dir(root.join("update/metadata/generations"))
                    .expect("bounded recovered inventory")
                    .count(),
                1
            );
        }
    }
}

#[test]
fn process_abort_at_every_multi_root_cleanup_prefix_is_recoverable() {
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
        .expect("commit multi-root crash-test predecessor");

        let mut child = spawn_crash_worker_with_profile(&root, 2, barrier, ROOT_VERSION, 2);
        let status = child.wait().expect("wait for multi-root crash worker");
        assert_eq!(
            status.signal(),
            Some(libc::SIGABRT),
            "barrier {barrier:?} must abort with SIGABRT"
        );

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
            .unwrap_or_else(|error| panic!("recovery after {barrier:?}: {error}")),
            MetadataCommitOutcome::AlreadyCommitted { sequence: 2 }
        );
        assert_eq!(
            std::fs::read_dir(root.join("update/metadata/generations"))
                .expect("bounded recovered inventory")
                .count(),
            1
        );
    }
}

#[test]
fn process_abort_at_every_successor_discard_barrier_is_recoverable() {
    for barrier in successor_discard_barriers(1) {
        let parent = tempfile::tempdir().expect("tempdir");
        let root = test_root(&parent);
        assert!(commit_candidate_for_test(
            authorization(&root),
            candidate_at(&root, "2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3,),
            FaultPlan {
                barrier: Some(Barrier::MetadataPrecommitSync),
            },
        )
        .is_err());

        let mut child = spawn_discard_crash_worker(&root, 1, barrier);
        let status = child.wait().expect("wait for discard crash worker");
        assert_eq!(
            status.signal(),
            Some(libc::SIGABRT),
            "discard barrier {barrier:?} must abort with SIGABRT"
        );
        discard_unselected_for_test(authorization(&root), FaultPlan::default())
            .unwrap_or_else(|error| panic!("restart after discard {barrier:?}: {error}"));
        assert!(read_selected(&authorization(&root))
            .expect("ordinary reader recovers after discard crash")
            .is_none());
    }
}

#[test]
fn process_abort_discarding_a_successor_preserves_the_selected_floor() {
    for barrier in successor_discard_barriers(2) {
        let parent = tempfile::tempdir().expect("tempdir");
        let root = test_root(&parent);
        commit_candidate_for_test(
            authorization(&root),
            candidate_at(&root, "2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3),
            FaultPlan::default(),
        )
        .expect("commit discard predecessor");
        let selected_selector =
            std::fs::read(root.join("update/metadata/current.json")).expect("selected selector");
        assert!(commit_candidate_for_test(
            authorization(&root),
            candidate_at(&root, "2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 3, 4,),
            FaultPlan {
                barrier: Some(Barrier::MetadataPrecommitSync),
            },
        )
        .is_err());

        let mut child = spawn_discard_crash_worker(&root, 2, barrier);
        let status = child.wait().expect("wait for successor discard worker");
        assert_eq!(
            status.signal(),
            Some(libc::SIGABRT),
            "successor discard barrier {barrier:?} must abort"
        );
        discard_unselected_for_test(authorization(&root), FaultPlan::default())
            .unwrap_or_else(|error| panic!("restart successor discard after {barrier:?}: {error}"));
        assert_eq!(
            std::fs::read(root.join("update/metadata/current.json"))
                .expect("selected floor survives"),
            selected_selector,
        );
        assert_eq!(
            read_selected(&authorization(&root))
                .expect("selected floor reads")
                .expect("selected floor exists")
                .sequence,
            1
        );
    }
}

#[test]
fn metadata_crash_worker_process() {
    let Some(root) = std::env::var_os("HF2Q_METADATA_CRASH_ROOT") else {
        return;
    };
    let sequence = std::env::var("HF2Q_METADATA_CRASH_SEQUENCE")
        .expect("crash sequence")
        .parse::<u64>()
        .expect("numeric crash sequence");
    let barrier =
        Barrier::parse(&std::env::var("HF2Q_METADATA_CRASH_BARRIER").expect("crash barrier"))
            .expect("known crash barrier");
    let root = Path::new(&root);
    if std::env::var_os("HF2Q_METADATA_DISCARD_MODE").is_some() {
        let _ = discard_unselected_for_test(
            authorization(root),
            FaultPlan {
                barrier: Some(barrier),
            },
        );
        return;
    }
    let (started, completed, default_root_version, default_timestamp_version) = if sequence == 1 {
        ("2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3)
    } else {
        ("2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 3, 4)
    };
    let root_version = std::env::var("HF2Q_METADATA_CRASH_ROOT_VERSION")
        .ok()
        .map(|value| value.parse::<u64>().expect("numeric root version"))
        .unwrap_or(default_root_version);
    let timestamp_version = std::env::var("HF2Q_METADATA_CRASH_TIMESTAMP_VERSION")
        .ok()
        .map(|value| value.parse::<u64>().expect("numeric timestamp version"))
        .unwrap_or(default_timestamp_version);
    let mut candidate = candidate_at(root, started, completed, root_version, timestamp_version);
    if let Ok(version) = std::env::var("HF2Q_METADATA_RESET_FROM_VERSION") {
        candidate.set_timestamp_snapshot_floor_reset_for_test(Some(
            version.parse().expect("numeric reset root version"),
        ));
    }
    let _ = commit_candidate_for_test(
        authorization(root),
        candidate,
        FaultPlan {
            barrier: Some(barrier),
        },
    );
}

fn spawn_discard_crash_worker(root: &Path, sequence: u64, barrier: Barrier) -> std::process::Child {
    let mut command = Command::new(std::env::current_exe().expect("test executable"));
    command
        .arg("--exact")
        .arg("distribution::install_state::metadata::tests::process_cases::metadata_crash_worker_process")
        .arg("--nocapture")
        .env("HF2Q_METADATA_CRASH_ROOT", root)
        .env("HF2Q_METADATA_CRASH_SEQUENCE", sequence.to_string())
        .env("HF2Q_METADATA_CRASH_BARRIER", barrier.name())
        .env("HF2Q_METADATA_DISCARD_MODE", "1")
        .env("HF2Q_METADATA_ABORT_ON_FAULT", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    command.spawn().expect("spawn discard crash worker")
}

#[test]
fn shared_installation_lock_excludes_both_process_entry_paths() {
    for mode in ["activation", "metadata"] {
        let parent = tempfile::tempdir().expect("tempdir");
        let root = test_root(&parent);
        let state = authorization(&root);
        let ready = parent.path().join(format!("{mode}-ready"));
        let mut child = spawn_lock_worker(&root, &ready, mode);
        wait_for_file(&ready);
        if mode == "activation" {
            assert!(matches!(
                commit_candidate_for_test(
                    state,
                    candidate_at(&root, "2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3,),
                    FaultPlan::default(),
                ),
                Err(MetadataJournalError::InstallState(
                    super::super::InstallStateError::Busy
                ))
            ));
        } else {
            assert!(matches!(
                LockedInstallation::acquire(&root),
                Err(super::super::InstallStateError::Busy)
            ));
        }
        child.kill().expect("kill lock worker");
        child.wait().expect("reap lock worker");
        LockedInstallation::acquire(&root).expect("lock is released when worker dies");
    }
}

#[test]
fn metadata_lock_worker_process() {
    let Some(root) = std::env::var_os("HF2Q_METADATA_LOCK_ROOT") else {
        return;
    };
    let ready = std::env::var_os("HF2Q_METADATA_LOCK_READY").expect("ready path");
    let mode = std::env::var("HF2Q_METADATA_LOCK_MODE").expect("lock mode");
    let root = Path::new(&root);
    if mode == "metadata" {
        hold_metadata_lock_for_test(root, Path::new(&ready));
    } else {
        let _locked = LockedInstallation::acquire(root).expect("acquire lock");
        std::fs::write(&ready, b"ready").expect("signal activation lock");
        thread::sleep(Duration::from_secs(60));
    }
}

fn spawn_crash_worker(root: &Path, sequence: u64, barrier: Barrier) -> std::process::Child {
    let (root_version, timestamp_version) = if sequence == 1 { (2, 3) } else { (3, 4) };
    let mut command =
        crash_worker_command(root, sequence, barrier, root_version, timestamp_version);
    if sequence == 2 {
        command.env("HF2Q_METADATA_RESET_FROM_VERSION", "2");
    }
    command.spawn().expect("spawn crash worker")
}

fn spawn_crash_worker_with_profile(
    root: &Path,
    sequence: u64,
    barrier: Barrier,
    root_version: u64,
    timestamp_version: u64,
) -> std::process::Child {
    crash_worker_command(root, sequence, barrier, root_version, timestamp_version)
        .spawn()
        .expect("spawn crash worker")
}

fn crash_worker_command(
    root: &Path,
    sequence: u64,
    barrier: Barrier,
    root_version: u64,
    timestamp_version: u64,
) -> Command {
    let mut command = Command::new(std::env::current_exe().expect("test executable"));
    command
        .arg("--exact")
        .arg("distribution::install_state::metadata::tests::process_cases::metadata_crash_worker_process")
        .arg("--nocapture")
        .env("HF2Q_METADATA_CRASH_ROOT", root)
        .env("HF2Q_METADATA_CRASH_SEQUENCE", sequence.to_string())
        .env("HF2Q_METADATA_CRASH_BARRIER", barrier.name())
        .env(
            "HF2Q_METADATA_CRASH_ROOT_VERSION",
            root_version.to_string(),
        )
        .env(
            "HF2Q_METADATA_CRASH_TIMESTAMP_VERSION",
            timestamp_version.to_string(),
        )
        .env("HF2Q_METADATA_ABORT_ON_FAULT", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    command
}

fn spawn_lock_worker(root: &Path, ready: &Path, mode: &str) -> std::process::Child {
    let mut command = Command::new(std::env::current_exe().expect("test executable"));
    command
        .arg("--exact")
        .arg("distribution::install_state::metadata::tests::process_cases::metadata_lock_worker_process")
        .arg("--nocapture")
        .env("HF2Q_METADATA_LOCK_ROOT", root)
        .env("HF2Q_METADATA_LOCK_READY", ready)
        .env("HF2Q_METADATA_LOCK_MODE", mode)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    command.spawn().expect("spawn lock worker")
}

fn wait_for_file(path: &Path) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !path.exists() {
        assert!(Instant::now() < deadline, "worker did not become ready");
        thread::sleep(Duration::from_millis(10));
    }
}
