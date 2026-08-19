use std::os::unix::process::ExitStatusExt;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use super::*;

fn wait_for_file(path: &Path) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !path.exists() {
        assert!(
            Instant::now() < deadline,
            "identity worker did not become ready"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn process_abort_at_every_identity_barrier_is_exactly_recoverable() {
    for barrier in IdentityBarrier::ALL {
        let parent = tempfile::tempdir().expect("tempdir");
        let root = test_root(&parent);
        let status = Command::new(std::env::current_exe().expect("test executable"))
            .arg("--exact")
            .arg(
                "distribution::install_state::identity::tests::process_cases::identity_crash_worker_process",
            )
            .arg("--nocapture")
            .env("HF2Q_IDENTITY_CRASH_ROOT", &root)
            .env("HF2Q_IDENTITY_CRASH_BARRIER", barrier.name())
            .env("HF2Q_IDENTITY_ABORT_ON_FAULT", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .expect("identity crash worker");
        assert_eq!(
            status.signal(),
            Some(libc::SIGABRT),
            "barrier {barrier:?} must terminate specifically with SIGABRT"
        );
        let repaired = bootstrap_installation_identity_for_test(
            authorization(&root),
            TEST_ID,
            IdentityFaultPlan::default(),
        )
        .unwrap_or_else(|error| panic!("repair after {barrier:?}: {error}"))
        .into_identity();
        assert_eq!(repaired.installation_id().as_str(), TEST_ID);
        assert_eq!(
            std::fs::read(root.join("update").join(IDENTITY_FILE)).expect("repaired identity"),
            expected_bytes(&root)
        );
    }
}

#[test]
fn identity_crash_worker_process() {
    let Some(root) = std::env::var_os("HF2Q_IDENTITY_CRASH_ROOT") else {
        return;
    };
    let barrier = IdentityBarrier::parse(
        &std::env::var("HF2Q_IDENTITY_CRASH_BARRIER").expect("identity barrier"),
    )
    .expect("known identity barrier");
    let _ = bootstrap_installation_identity_for_test(
        authorization(Path::new(&root)),
        TEST_ID,
        IdentityFaultPlan::once(barrier),
    );
    panic!("identity fault barrier did not abort");
}

#[test]
fn shared_identity_lock_excludes_a_fresh_process_bootstrap() {
    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    let ready = parent.path().join("identity-lock-ready");
    let mut child = Command::new(std::env::current_exe().expect("test executable"))
        .arg("--exact")
        .arg(
            "distribution::install_state::identity::tests::process_cases::identity_lock_worker_process",
        )
        .arg("--nocapture")
        .env("HF2Q_IDENTITY_LOCK_ROOT", &root)
        .env("HF2Q_IDENTITY_LOCK_READY", &ready)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("identity lock worker");
    wait_for_file(&ready);
    assert!(matches!(
        bootstrap_installation_identity_for_test(
            authorization(&root),
            TEST_ID,
            IdentityFaultPlan::default(),
        ),
        Err(InstallStateError::Busy)
    ));
    child.kill().expect("stop identity lock worker");
    child.wait().expect("reap identity lock worker");
    assert!(matches!(
        bootstrap_installation_identity_for_test(
            authorization(&root),
            TEST_ID,
            IdentityFaultPlan::default(),
        )
        .expect("lock released"),
        InstallationIdentityBootstrap::AlreadyCreated(_)
    ));
}

#[test]
fn identity_lock_worker_process() {
    let Some(root) = std::env::var_os("HF2Q_IDENTITY_LOCK_ROOT") else {
        return;
    };
    let ready = std::env::var_os("HF2Q_IDENTITY_LOCK_READY").expect("ready path");
    let identity = bootstrap_installation_identity_for_test(
        authorization(Path::new(&root)),
        TEST_ID,
        IdentityFaultPlan::default(),
    )
    .expect("worker identity")
    .into_identity();
    let _locked = identity.lock().expect("worker identity lock");
    std::fs::write(ready, b"ready").expect("signal identity lock");
    std::thread::sleep(Duration::from_secs(60));
}
