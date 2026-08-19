use super::*;

#[cfg(target_os = "macos")]
#[test]
fn every_prepared_publication_barrier_returns_typed_state_and_exact_retry_recovers() {
    let (manifest, manifest_bytes) = manifest();
    let probe_archive = archive(&manifest_bytes, CompressionMethod::Stored);
    let probe_fixture = stable_release_repository_for_artifacts(&manifest_bytes, &probe_archive);
    let (_probe_temp, probe_authorization) = make_authorization();
    let probe_anchor = leaked_anchor(&probe_fixture.repository.anchor);
    commit_fixture(
        &probe_authorization,
        &probe_anchor,
        &probe_fixture.repository,
    );
    reset_observed_prepared_barriers();
    let probe = bundle(
        &probe_authorization,
        &probe_anchor,
        &probe_fixture.pointer,
        manifest.clone(),
        manifest_bytes.clone(),
        &probe_archive,
    );
    assert!(matches!(
        prepare_release_for_test(probe).expect("barrier probe prepares"),
        PreparedReleaseOutcome::Prepared(_)
    ));
    let barrier_count = observed_prepared_barriers();
    assert!(
        barrier_count > 20,
        "publication must expose all durability barriers"
    );

    for barrier in 1..=barrier_count {
        let archive_bytes = archive(&manifest_bytes, CompressionMethod::Stored);
        let fixture = stable_release_repository_for_artifacts(&manifest_bytes, &archive_bytes);
        let (temp, authorization) = make_authorization();
        let root = temp
            .path()
            .canonicalize()
            .expect("canonical parent")
            .join("state");
        let anchor = leaked_anchor(&fixture.repository.anchor);
        commit_fixture(&authorization, &anchor, &fixture.repository);
        reset_observed_prepared_barriers();
        fail_after_prepared_barrier(barrier);
        let release = bundle(
            &authorization,
            &anchor,
            &fixture.pointer,
            manifest.clone(),
            manifest_bytes.clone(),
            &archive_bytes,
        );
        let error = prepare_release_for_test(release)
            .expect_err("scripted prepared barrier must return an error");
        let committed = root.join("versions/0.2.0").exists();
        assert_eq!(
            is_prepared_durability_unknown(&error),
            committed,
            "barrier {barrier} must distinguish precommit from postcommit failure: {error:?}"
        );
        assert!(!root.join("activations").exists());
        assert!(!root.join("current").exists());

        reset_observed_prepared_barriers();
        let retry = bundle(
            &authorization,
            &anchor,
            &fixture.pointer,
            manifest.clone(),
            manifest_bytes.clone(),
            &archive_bytes,
        );
        let recovered = prepare_release_for_test(retry)
            .unwrap_or_else(|retry_error| panic!("barrier {barrier} retry: {retry_error:?}"));
        assert!(matches!(
            recovered,
            PreparedReleaseOutcome::Prepared(_) | PreparedReleaseOutcome::AlreadyPrepared(_)
        ));
        assert!(root
            .join("versions/0.2.0/version-installation.json")
            .is_file());
        assert!(!root.join("activations").exists());
        assert!(!root.join("current").exists());
    }
}
#[cfg(target_os = "macos")]
#[test]
fn prepared_version_crash_worker_process() {
    let Ok(root) = std::env::var("HF2Q_PREPARED_CRASH_ROOT") else {
        return;
    };
    let barrier = std::env::var("HF2Q_PREPARED_CRASH_BARRIER")
        .expect("prepared crash barrier")
        .parse::<usize>()
        .expect("numeric prepared crash barrier");
    let (manifest, manifest_bytes) = manifest();
    let archive_bytes = archive(&manifest_bytes, CompressionMethod::Deflated);
    let fixture = stable_release_repository_for_artifacts(&manifest_bytes, &archive_bytes);
    let authorization =
        MetadataStateAuthorization::for_test_path(std::path::Path::new(&root), INSTALLATION_ID);
    let anchor = leaked_anchor(&fixture.repository.anchor);
    let release = bundle(
        &authorization,
        &anchor,
        &fixture.pointer,
        manifest,
        manifest_bytes,
        &archive_bytes,
    );
    abort_after_prepared_barrier(barrier);
    let result = prepare_release_for_test(release);
    panic!("prepared crash barrier {barrier} did not abort: {result:?}");
}
#[cfg(target_os = "macos")]
#[test]
fn process_abort_at_every_prepared_barrier_is_fresh_process_recoverable() {
    let (manifest, manifest_bytes) = manifest();
    let probe_archive = archive(&manifest_bytes, CompressionMethod::Deflated);
    let probe_fixture = stable_release_repository_for_artifacts(&manifest_bytes, &probe_archive);
    let (_probe_temp, probe_authorization) = make_authorization();
    let probe_anchor = leaked_anchor(&probe_fixture.repository.anchor);
    commit_fixture(
        &probe_authorization,
        &probe_anchor,
        &probe_fixture.repository,
    );
    reset_observed_prepared_barriers();
    let probe = bundle(
        &probe_authorization,
        &probe_anchor,
        &probe_fixture.pointer,
        manifest.clone(),
        manifest_bytes.clone(),
        &probe_archive,
    );
    drop(prepare_release_for_test(probe).expect("abort barrier probe prepares"));
    let barrier_count = observed_prepared_barriers();
    assert!(barrier_count > 20);
    const WORKER: &str = "distribution::update_auth::target_set::tests::extraction_cases::prepared_publication_cases::fault_cases::prepared_version_crash_worker_process";

    for barrier in 1..=barrier_count {
        let archive_bytes = archive(&manifest_bytes, CompressionMethod::Deflated);
        let fixture = stable_release_repository_for_artifacts(&manifest_bytes, &archive_bytes);
        let (temp, authorization) = make_authorization();
        let root = temp
            .path()
            .canonicalize()
            .expect("canonical parent")
            .join("state");
        let anchor = leaked_anchor(&fixture.repository.anchor);
        commit_fixture(&authorization, &anchor, &fixture.repository);

        let status =
            crate::distribution::install_state::run_prepared_crash_worker(WORKER, &root, barrier);
        assert_eq!(
            status.signal(),
            Some(libc::SIGABRT),
            "barrier {barrier} must terminate specifically by SIGABRT"
        );
        assert!(
            root.join("update/extractions").exists()
                || root.join("update/prepared").exists()
                || root.join("versions").exists(),
            "barrier {barrier} must leave bounded inert or published evidence"
        );
        assert!(!root.join("activations").exists());
        assert!(!root.join("current").exists());

        let retry = bundle(
            &authorization,
            &anchor,
            &fixture.pointer,
            manifest.clone(),
            manifest_bytes.clone(),
            &archive_bytes,
        );
        let recovered = prepare_release_for_test(retry)
            .unwrap_or_else(|error| panic!("crash barrier {barrier} retry: {error:?}"));
        assert!(matches!(
            recovered,
            PreparedReleaseOutcome::Prepared(_) | PreparedReleaseOutcome::AlreadyPrepared(_)
        ));
        assert!(root
            .join("versions/0.2.0/version-installation.json")
            .is_file());
        assert!(!root.join("activations").exists());
        assert!(!root.join("current").exists());
    }
}
