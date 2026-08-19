use super::*;

#[cfg(target_os = "macos")]
#[test]
fn hostile_prepared_residue_is_preserved_and_never_adopted() {
    // A conflicting marker prefix remains diagnostic evidence.
    {
        let (manifest, manifest_bytes) = manifest();
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
        fail_after_prepared_barrier(1);
        let failed = bundle(
            &authorization,
            &anchor,
            &fixture.pointer,
            manifest.clone(),
            manifest_bytes.clone(),
            &archive_bytes,
        );
        assert!(prepare_release_for_test(failed).is_err());
        reset_observed_prepared_barriers();
        let prepared = root.join("update/prepared");
        let marker = std::fs::read_dir(&prepared)
            .expect("prepared marker inventory")
            .next()
            .expect("marker entry")
            .expect("marker directory entry")
            .path();
        let original = std::fs::read(&marker).expect("partial marker bytes");
        let corrupt = vec![b'x'; original.len()];
        std::fs::write(&marker, &corrupt).expect("corrupt marker prefix");
        let retry = bundle(
            &authorization,
            &anchor,
            &fixture.pointer,
            manifest,
            manifest_bytes,
            &archive_bytes,
        );
        assert!(prepare_release_for_test(retry).is_err());
        assert_eq!(std::fs::read(marker).unwrap(), corrupt);
        assert!(!root.join("versions/0.2.0").exists());
        assert!(!root.join("activations").exists());
        assert!(!root.join("current").exists());
    }

    // An extra pending-tree node cannot become part of the release inventory.
    {
        let (manifest, manifest_bytes) = manifest();
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
        reset_observed_prepared_barriers();
        fail_after_prepared_barrier(6);
        let failed = bundle(
            &authorization,
            &anchor,
            &fixture.pointer,
            manifest.clone(),
            manifest_bytes.clone(),
            &archive_bytes,
        );
        assert!(prepare_release_for_test(failed).is_err());
        reset_observed_prepared_barriers();
        let pending = std::fs::read_dir(root.join("update/prepared"))
            .expect("pending parent")
            .map(|entry| entry.unwrap().path())
            .find(|path| path.is_dir())
            .expect("pending tree");
        let hostile = pending.join("unexpected");
        std::fs::write(&hostile, b"diagnostic evidence").expect("hostile extra file");
        std::fs::set_permissions(&hostile, std::fs::Permissions::from_mode(0o600))
            .expect("hostile file mode");
        let retry = bundle(
            &authorization,
            &anchor,
            &fixture.pointer,
            manifest,
            manifest_bytes,
            &archive_bytes,
        );
        assert!(prepare_release_for_test(retry).is_err());
        assert_eq!(std::fs::read(hostile).unwrap(), b"diagnostic evidence");
        assert!(!root.join("versions/0.2.0").exists());
        assert!(!root.join("activations").exists());
        assert!(!root.join("current").exists());
    }

    // A published tree is adopted only if every exact payload still matches.
    {
        let (manifest, manifest_bytes) = manifest();
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
        let release = bundle(
            &authorization,
            &anchor,
            &fixture.pointer,
            manifest.clone(),
            manifest_bytes.clone(),
            &archive_bytes,
        );
        drop(prepare_release_for_test(release).expect("publish exact version"));
        let document = root.join("versions/0.2.0/share/doc/hf2q/README.md");
        let corrupt = vec![b'x'; PAYLOADS[1].1.len()];
        std::fs::write(&document, &corrupt).expect("corrupt published payload");
        let retry = bundle(
            &authorization,
            &anchor,
            &fixture.pointer,
            manifest,
            manifest_bytes,
            &archive_bytes,
        );
        assert!(prepare_release_for_test(retry).is_err());
        assert_eq!(std::fs::read(document).unwrap(), corrupt);
        assert!(!root.join("activations").exists());
        assert!(!root.join("current").exists());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn recovered_marker_time_cannot_be_later_than_the_fresh_tuf_replay() {
    let (manifest, manifest_bytes) = manifest();
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
    fail_after_prepared_barrier(6);
    let failed = bundle(
        &authorization,
        &anchor,
        &fixture.pointer,
        manifest.clone(),
        manifest_bytes.clone(),
        &archive_bytes,
    );
    assert!(prepare_release_for_test(failed).is_err());
    reset_observed_prepared_barriers();

    let pending = std::fs::read_dir(root.join("update/prepared"))
        .expect("pending parent")
        .map(|entry| entry.unwrap().path())
        .find(|path| path.is_dir())
        .expect("pending tree");
    let marker = pending.join("version-installation.json");
    let mut bytes = std::fs::read(&marker).expect("exact pending marker");
    let key = b"\"installed_at_unix_seconds\":";
    let start = bytes
        .windows(key.len())
        .position(|window| window == key)
        .expect("installed-at field")
        + key.len();
    let end = bytes[start..]
        .iter()
        .position(|byte| !byte.is_ascii_digit())
        .map(|length| start + length)
        .expect("installed-at terminator");
    let installed_at = std::str::from_utf8(&bytes[start..end])
        .unwrap()
        .parse::<u64>()
        .unwrap();
    let future = (installed_at + 86_400).to_string();
    assert_eq!(future.len(), end - start);
    bytes[start..end].copy_from_slice(future.as_bytes());
    std::fs::write(&marker, &bytes).expect("future canonical marker evidence");

    let retry = bundle(
        &authorization,
        &anchor,
        &fixture.pointer,
        manifest,
        manifest_bytes,
        &archive_bytes,
    );
    assert!(prepare_release_for_test(retry).is_err());
    assert_eq!(std::fs::read(marker).unwrap(), bytes);
    assert!(!root.join("versions/0.2.0").exists());
    assert!(!root.join("activations").exists());
    assert!(!root.join("current").exists());
}

#[cfg(target_os = "macos")]
#[test]
fn detached_prepared_namespace_fails_closed_and_preserves_evidence() {
    let (manifest, manifest_bytes) = manifest();
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
    fail_after_prepared_barrier(6);
    let failed = bundle(
        &authorization,
        &anchor,
        &fixture.pointer,
        manifest.clone(),
        manifest_bytes.clone(),
        &archive_bytes,
    );
    assert!(prepare_release_for_test(failed).is_err());
    reset_observed_prepared_barriers();
    let update = root.join("update");
    std::fs::rename(update.join("prepared"), update.join("prepared-detached"))
        .expect("detach prepared namespace");
    std::fs::create_dir(update.join("prepared")).expect("replacement prepared namespace");
    std::fs::set_permissions(
        update.join("prepared"),
        std::fs::Permissions::from_mode(0o700),
    )
    .expect("replacement prepared mode");
    let error = begin_artifact_fetch_for_test(
        &authorization,
        &anchor,
        [
            instant("2026-08-18T09:02:00Z"),
            instant("2026-08-18T09:02:01Z"),
        ],
    )
    .expect_err("detached prepared namespace blocks fresh authority");
    assert!(matches!(
        error,
        ArtifactFetchAuthorizationError::Authentication(TufVerifierError::Journal(
            MetadataJournalError::InstallState(
                crate::distribution::install_state::InstallStateError::InvalidLayout(_)
            )
        ))
    ));
    drop((manifest, manifest_bytes, archive_bytes));
    assert!(update.join("prepared-detached").exists());
    assert!(!root.join("versions/0.2.0").exists());
    assert!(!root.join("activations").exists());
    assert!(!root.join("current").exists());
}

#[cfg(target_os = "macos")]
#[test]
fn prepared_namespace_swap_at_commit_boundary_is_rejected_before_publication() {
    let (manifest, manifest_bytes) = manifest();
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
    let update = root.join("update");
    let hook_update = update.clone();
    set_prepared_precommit_hook(move || {
        std::fs::rename(
            hook_update.join("prepared"),
            hook_update.join("prepared-detached"),
        )
        .expect("detach prepared namespace at commit boundary");
        std::fs::create_dir(hook_update.join("prepared")).expect("replacement prepared namespace");
        std::fs::set_permissions(
            hook_update.join("prepared"),
            std::fs::Permissions::from_mode(0o700),
        )
        .expect("replacement prepared mode");
    });
    let release = bundle(
        &authorization,
        &anchor,
        &fixture.pointer,
        manifest,
        manifest_bytes,
        &archive_bytes,
    );
    let error = prepare_release_for_test(release).expect_err("namespace swap must fail");
    assert!(
        matches!(
            error,
            PreparedReleaseError::PreparedCommit(
                crate::distribution::update_auth::PreparedVersionCommitError::Publication(_)
            )
        ),
        "unexpected namespace-swap error: {error:?}"
    );
    assert!(update.join("prepared-detached").exists());
    assert!(!root.join("versions/0.2.0").exists());
    assert!(!is_prepared_durability_unknown(&error));
    assert!(!root.join("activations").exists());
    assert!(!root.join("current").exists());
}
